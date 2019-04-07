from logbook import Logger, StreamHandler, WARNING, NOTICE, INFO, DEBUG
import sys
import copy

import numpy as np
from collections import namedtuple, OrderedDict

import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")


class BaseEngine(object):
    """
    Base stateful computational backend class.
    """

    def __init__(
            self,
            name='BaseEngine',
            task=0,
            log=None,
            log_level=INFO,
            **kwargs
    ):
        self.name = name
        self.task = task

        if log is None:
            StreamHandler(sys.stdout).push_application()
            self.log_level = log_level
            self.log = Logger('{}_{}'.format(self.name, self.task), level=self.log_level)

        else:
            self.log = log
            self.log_level = None

        self.init_state = None
        self.state = None
        self.ready = False

    def start(self, *args, **kwargs):
        self.ready = True

    def stop(self, *args, **kwargs):
        self.ready = False
        pass

    def update_state(self, *args, **kwargs):
        pass


PandasStateConfig = namedtuple('PandasStateConfig', ['columns', 'depth'])


class BasePandasIterator(BaseEngine):

    def __init__(
            self,
            dataframe,
            sample_length,
            state_config,
            name='MarketIterator',
            **kwargs
    ):
        super().__init__(name=name, **kwargs)

        self.dataframe = dataframe
        self.sample_length = sample_length
        self.state_config = state_config

        self.sample_max_depth = self.get_max_depth(self.state_config)

        try:
            assert self.sample_length <= self.dataframe.shape[0] - self.sample_max_depth

        except AssertionError as e:
            msg = 'Expected sample be shorter than data length - embedding length, got: {} and {}'.format(
                self.sample_length, self.dataframe.shape[0] - self.sample_max_depth
            )
            self.log.error(msg)
            raise AssertionError(msg)

        if self.sample_length > 0:
            self.sample_start_interval = dict(
                low=self.sample_max_depth,
                high=self.dataframe.shape[0] - self.sample_length
            )
        else:
            self.sample_length = self.dataframe.shape[0] - self.sample_max_depth
            self.sample_start_interval = dict(
                low=self.sample_max_depth - 1 ,
                high=self.sample_max_depth
            )
        self.start_pointer = None
        self.iter_passed = None

    def update_start_pointer(self, start_pointer=None):
        if start_pointer is None:
            self.start_pointer = np.random.randint(**self.sample_start_interval)

        else:
            assert self.sample_start_interval['low'] <= start_pointer < self.sample_start_interval['high']
            self.start_pointer = start_pointer

    def get_max_depth(self, state_config):
        if isinstance(state_config, dict):
            depth = []
            for value in state_config.values():
                depth.append(self.get_max_depth(value))
            return max(depth)

        else:
            try:
                assert isinstance(state_config, PandasStateConfig)

            except AssertionError as e:
                self.log.error(e)
                raise TypeError(e)

            return state_config.depth

    @staticmethod
    def get_data_slice(dataframe, columns, depth, position):
        return dataframe[columns][position - depth: position]

    def get_state(self, position, state_config):
        if isinstance(state_config, dict):
            state = {key: self.get_state(position, value) for key, value in state_config.items()}
        else:
            state = self.get_data_slice(self.dataframe, state_config.columns, state_config.depth, position)

        return state

    def start(self, start_pointer=None):
        self.update_start_pointer(start_pointer)
        self.iter_passed = 0
        self.ready = True
        self.update_state()

    def update_state(self, inputs=None, *args, **kwargs):
        if self.ready:
            self.state = self.get_state(self.start_pointer + self.iter_passed, self.state_config)
            self.iter_passed += 1

            if self.iter_passed >= self.sample_length:
                self.ready = False

            self.log.debug(
                'market iteration {} of {}, ready: {}'.format(self.iter_passed, self.sample_length, self.ready)
            )

        else:
            msg = 'Attempt to iterate uninitialised data / go beyond sample length.\nHint: forgot to check .ready flag?'
            self.log.error(msg)
            raise IndexError(msg)


MarketOrder = namedtuple('MarketOrder', ['asset', 'type'])

OrderRecord = namedtuple('OrderRecord', ['type', 'size', 'result'])


class BasePortfolioManager(BaseEngine):

    def __init__(
            self,
            max_position_size,
            order_size=1,
            order_commission=0.0,
            orders=('buy', 'sell', 'close'),
            assets=('default_asset',),
            name='PortfolioManager',
            **kwargs
    ):
        super().__init__(name=name, **kwargs)

        self.max_position_size = max_position_size
        self.order_size = abs(order_size)
        self.order_commission = abs(order_commission)
        self.orders = orders
        self.assets = list(assets)

        self.portfolio = None
        self.portfolio_value = None
        self.assets_prices = None
        self.submitted_orders = None
        self.asset_just_closed = None

        self.unrealised_return = None
        self.realised_return = None
        self.last_portfolio_value = None
        self.last_realised_portfolio_value = None

        self.step_order_record = None

    def update_portfolio_value(self, market_state):
        self.assets_prices = np.concatenate([np.ones(1), market_state[self.assets].values[0, :]])
        self.portfolio_value = np.sum(np.asarray(list(self.portfolio.values())) * self.assets_prices)

    def submit_orders(self, orders):
        if not isinstance(orders, list):
            orders_list = [orders]

        else:
            orders_list = orders

        for order in orders_list:
            try:
                assert isinstance(order, MarketOrder)

            except AssertionError:
                msg = 'Expected order be instance of {}, got: {}'.format(MarketOrder, type(order))
                self.log.error(msg)
                raise TypeError(msg)

            try:
                assert order.asset in self.assets

            except AssertionError:
                msg = 'Expected order asset be in {}, got: {}'.format(self.assets, order.asset)
                self.log.error(msg)
                raise ValueError(msg)

        self.submitted_orders = orders_list

    def execute_orders(self, market_state):
        self.step_order_record = []
        while len(self.submitted_orders) > 0:
            order = self.submitted_orders.pop(-1)

            if order.type == 'buy':
                order_size = self.order_size

            elif order.type == 'sell':
                order_size = - self.order_size

            elif order.type == 'close':
                order_size = - self.portfolio[order.asset]

            else:
                msg = 'Expected order type be in {}, got: {}'.format(self.orders, order.type)
                self.log.error(msg)
                raise ValueError(msg)
            self.log.debug('order type: {}'.format(order.type))

            order_value = abs(np.squeeze(market_state[order.asset].values * order_size))
            friction_value = order_value * self.order_commission

            self.log.debug('order_value: {:.4f}, friction_value: {:.6f}'.format(order_value, friction_value))

            previous_asset_size = copy.copy(self.portfolio[order.asset])

            if abs(previous_asset_size + order_size) > self.max_position_size or order_size == 0:
                self.log.debug(
                    'Order {} \nfailed due to exceeding max. position size or zero order value.'.format(order)
                )
                order_executed = False

            else:
                order_executed = True

                self.portfolio[order.asset] += order_size

                cash_flow = (previous_asset_size - self.portfolio[order.asset]) \
                    * np.squeeze(market_state[order.asset].values)

                self.log.debug('cash_flow: {:.4f}'.format(cash_flow))

                self.portfolio['cash'] += (cash_flow - friction_value)

                if self.portfolio[order.asset] == 0:
                    self.asset_just_closed[order.asset] = True

                self.log.debug('asset_just_closed: {}'.format(self.asset_just_closed))

            self.step_order_record.append(
                OrderRecord(
                    type=order.type,
                    size=order_size,
                    result=order_executed,
                )
            )

    def reset_just_closed(self):
        for k in self.asset_just_closed.keys():
            self.asset_just_closed[k]= False

    def start(self, market_state, **kwargs):
        self.portfolio = OrderedDict(
            [
                (name, amount) for name, amount in zip(['cash'] + self.assets, np.zeros(len(self.assets) + 1))
            ]
        )
        self.asset_just_closed = OrderedDict(
            [
                (name, False) for name in self.assets
            ]
        )
        self.assets_prices = np.zeros(len(self.assets) + 1)
        self.portfolio_value = 0.0
        self.submitted_orders = []
        self.unrealised_return = 0.0
        self.realised_return = 0.0
        self.last_portfolio_value = 0.0
        self.last_realised_portfolio_value = 0.0
        self.ready = True
        self.update_state(market_state)

    def update_state(self, market_state, **kwargs):

        # Execute pending orders:
        self.reset_just_closed()
        self.execute_orders(market_state)

        # Compute state:
        self.update_portfolio_value(market_state)

        self.unrealised_return = self.portfolio_value - self.last_portfolio_value
        self.last_portfolio_value = copy.copy(self.portfolio_value)

        if np.asarray(list(self.asset_just_closed.values())).all():
            # TODO: all() --> any() for multiasset!
            self.realised_return = self.portfolio_value - self.last_realised_portfolio_value
            self.last_realised_portfolio_value = copy.copy(self.portfolio_value)

        else:
            self.realised_return = np.nan

        self.log.debug('upd. ptf: {}'.format(self.portfolio))
        self.log.debug('upd. u_ret: {}, real_ret: {}'.format(self.unrealised_return, self.realised_return))
        self.state = dict(
            portfolio=copy.deepcopy(self.portfolio),
            portfolio_value=self.portfolio_value,
            broker_value=self.portfolio_value,  # btgym compatibility
            realised_return=self.realised_return,
            unrealised_return=self.unrealised_return,
            order=self.step_order_record,
        )


class BaseTradeEngine(BaseEngine):

    def __init__(
            self,
            market_config,
            manager_config,
            market_engine_ref=BasePandasIterator,
            manager_engine_ref=BasePortfolioManager,
            name='TradeEngine',
            **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.market = market_engine_ref(name=self.name + 'Market', task=self.task, log=self.log, **market_config)
        self.manager = manager_engine_ref(name=self.name + 'Manager', task=self.task, log=self.log, **manager_config)
        self.ready = self.market.ready

    def start(self, *args, **kwargs):
        self.market.start(*args, **kwargs)
        try:
            assert 'assets' in set(self.market.state.keys())

        except AssertionError:
            msg = 'Expected iterator state contain `assets` key, found: {}'.format(set(self.market.state.keys()))
            self.log.error(msg)
            raise KeyError(msg)

        self.manager.start(market_state=self.market.state['assets'], **kwargs)
        self.state = dict(
            market=self.market.state,
            manager=self.manager.state
        )
        self.ready = self.market.ready

    def update_state(self, *args, **kwargs):
        self.market.update_state()
        self.manager.update_state(market_state=self.market.state['assets'])
        self.ready = self.market.ready
        self.state = dict(
            market=self.market.state,
            manager=self.manager.state
        )

    def submit_orders(self, *args, **kwargs):
        self.manager.submit_orders(*args, **kwargs)
