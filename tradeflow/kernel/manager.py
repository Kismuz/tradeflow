from logbook import INFO
import sys
import copy
import numpy as np
from collections import namedtuple, OrderedDict

from ..core import Kernel
from .action import MarketOrder


import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")


OrderRecord = namedtuple('OrderRecord', ['type', 'size', 'result'])


class BasePortfolioManager(Kernel):

    def __init__(
            self,
            max_position_size,
            order_size=1,
            order_commission=0.0,
            orders=('buy', 'sell', 'close'),
            assets=('default_asset',),
            name='PortfolioManager',
            pass_input_state=False,
            task=0,
            log=None,
            log_level=INFO,
    ):
        super().__init__(name=name, task=task, log=log, log_level=log_level)

        self.max_position_size = max_position_size
        self.order_size = abs(order_size)
        self.order_commission = abs(order_commission)
        self.orders = orders
        self.assets = list(assets)
        self.pass_input_state = pass_input_state

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
        self.assets_prices = np.concatenate([np.ones(1)] + [market_state[asset].values[0, :] for asset in self.assets])
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
        # self.log.debug('self.asset_just_closed: ', self.asset_just_closed)
        for k in self.asset_just_closed.keys():
            self.asset_just_closed[k]= False

    def update_state(self, input_state, reset, orders):
        if reset:
            self._start(input_state)

        else:
            self._update_state(input_state, orders)

        if self.pass_input_state:
            return self.state, input_state

        else:
            return self.state

    def _start(self, market_state):
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
        self._update_state(market_state, [])

    def _update_state(self, market_state, orders):

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
            realized_return=self.realised_return,
            unrealized_return=self.unrealised_return,
            order=self.step_order_record,
        )

        self.submit_orders(orders)