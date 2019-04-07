import copy
import time
import os
import numpy as np
from datetime import timedelta

from .base import BaseEnvironment, BaseNestedEnvironment
from ..kernels.base import BaseTradeKernel, PandasStateConfig, MarketOrder
from btgym.spaces import DictSpace, ActionDictSpace, spaces

from ray.tune.util import get_pinned_object


class TradeEnvironment(BaseEnvironment):

    def __init__(
            self,

            sample_length,
            state_columns,
            state_depth,
            asset_columns,
            max_position_size,
            order_size,
            order_commission,
            dataframe=None,
            dataframe_store_id=None,
            observation_space=None,
            action_space=None,
            action_map=None,
            skip_step=1,
            engine_class_ref=BaseTradeKernel,
            name='TradeEnvironment',
            **kwargs
    ):
        if isinstance(asset_columns, str):
            asset_columns = [asset_columns]

        if isinstance(state_columns, str):
            state_columns = [state_columns]

        if dataframe_store_id is not None:
            dataframe = get_pinned_object(dataframe_store_id)

        else:
            assert dataframe is not None, 'Either `dataframe` or `dataframe_store_id` kwarg should be provided.'

        market_config = dict(
            dataframe=dataframe,
            sample_length=sample_length,
            state_config=dict(
                observation=PandasStateConfig(
                    columns=state_columns,
                    depth=state_depth,
                ),
                assets=PandasStateConfig(
                    columns=asset_columns,
                    depth=1,
                ),
            )
        )
        manager_config = dict(
            max_position_size=max_position_size,
            order_size=order_size,
            order_commission=order_commission,
            orders=('buy', 'sell', 'close'),
            assets=asset_columns,
        )
        if observation_space is None:
            observation_space = DictSpace(
                {
                    'external': spaces.Box(low=-100, high=100, shape=[state_depth, 1, len(state_columns)], dtype='float'),
                    'internal': spaces.Box(low=-100, high=100, shape=[state_depth, 1, 3], dtype='float'),
                }
            )
        else:
            try:
                assert isinstance(observation_space, DictSpace)

            except AssertionError:
                msg = 'Expected observation space as instance of {}, got: {}'.format(DictSpace, type(observation_space))
                self.log.error(msg)
                raise TypeError(msg)

        if action_space is None:
            action_space = ActionDictSpace(
                base_actions=[0, 1, 2, 3],
                assets=asset_columns
            )
            self.action_map = {0: None, 1: 'buy', 2: 'sell', 3: 'close'}

        else:
            try:
                assert isinstance(action_space, ActionDictSpace)

            except AssertionError:
                msg = 'Expected action space as instance of {}, got: {}'.format(ActionDictSpace, type(action_space))
                self.log.error(msg)
                raise TypeError(msg)
            try:
                assert action_map is not None

            except AssertionError:
                msg = '`action_space` arg. requires `action_map` to be provided as well.'
                self.log.error(msg)
                raise TypeError(msg)

        super().__init__(
            engine_class_ref=engine_class_ref,
            engine_config=dict(
                market_config=market_config,
                manager_config=manager_config,
            ),
            action_space=action_space,
            observation_space=observation_space,
            name=name,
            **kwargs
        )
        self.skip_step = skip_step
        self.state_accum = None

    def reset(self, *args, **kwargs):
        self.reset_state_accum()
        self.engine.stop(*args, **kwargs)
        self.engine.start(*args, **kwargs)
        self.update_state_accum()
        self.done = False
        self.step_counter = 0

        return self.get_state()

    def step(self, action):
        orders = [MarketOrder(asset, self.action_map[value]) for asset, value in action.items() if value != 0]
        submitted = False

        if not self.done:
            for i in range(self.skip_step):
                self.engine.update_state()
                self.update_state_accum()
                self.done = self.get_done()

                if not submitted:
                    self.engine.submit_orders(orders)
                    submitted = True

                if self.done:
                    break

            obs = self.get_state()
            reward = self.get_reward()
            info = self.get_info()

            self.step_counter += 1
            if self.done:
                self.elapsed_time = timedelta(seconds=time.time() - self.start_time)
                self.elapsed_length = copy.copy(self.step_counter)

        else:
            msg = 'Environment either exhausted or has not been initialised. Hint: forgot to call ._reset_state()?'
            self.log.error(msg)
            raise RuntimeError(msg)

        return obs, reward, self.done, info

    def get_state(self):
        return {
            'external': self.engine.market.state['observation'].values[:, None, :],
            'internal': np.nan_to_num(np.stack([acc for acc in self.state_accum.values()], axis=-1))[:, None, :]
        }

    def reset_state_accum(self):
        accum_modes = ['unrealized_return', 'realized_return', 'normalized_exposure']
        self.state_accum = {mode: np.zeros(self.engine.market.sample_max_depth) for mode in accum_modes}

    def update_state_accum(self):
        # TODO: correct for single asset only:
        exposure = np.sum([self.engine.manager.portfolio[asset] for asset in self.engine.manager.assets])
        self.log.debug('exposure: {}'.format(exposure))

        norm_exposure = exposure / self.engine.manager.max_position_size
        self.log.debug('normalized exposure: {}'.format(norm_exposure))

        u_ret = self.engine.manager.state['unrealised_return']
        r_ret = self.engine.manager.state['realised_return']
        self.log.debug('last returns: {}/{}'.format(u_ret, r_ret))

        self.state_accum['unrealized_return'] = np.append(self.state_accum['unrealized_return'][1:], u_ret)
        self.state_accum['realized_return'] = np.append(self.state_accum['realized_return'][1:], r_ret)

        # self.state_accum['exposure'] = np.append(
        #     self.state_accum['exposure'][1:],
        #     exposure
        # )
        self.state_accum['normalized_exposure'] = np.append(
            self.state_accum['normalized_exposure'][1:],
            norm_exposure
        )

    def get_reward(self):
        u_ret = self.state_accum['unrealized_return'][-self.skip_step:]
        r_ret = self.state_accum['realized_return'][-self.skip_step:]
        self.log.debug('u_ret: {}, r_ret: {}'.format(u_ret, r_ret))
        mean_unr_returns = np.mean(np.asarray(u_ret))
        mean_real_returns = np.nanmean(np.asarray(r_ret))
        if np.isnan(mean_real_returns):
            mean_real_returns = 0.0

        return np.asarray(
            1.0 * mean_unr_returns +
            10.0 * mean_real_returns
        )

    def get_info(self):
        info = copy.deepcopy(self.engine.manager.state)
        info['assets'] = copy.deepcopy(self.engine.market.state['assets'].values)
        return [info]

    def get_done(self):
        return not copy.copy(self.engine.ready)


class BTgymCompatibleEnvironment(BaseNestedEnvironment):
    """
    Environment wrapper compatible with btgym.algorithms specs.
    Enables `switched` execution of two instances of backend environments, i. e.
    either one or another environment can run episode in given time.
    """

    def __init__(
            self,
            train_dataframe=None,
            test_dataframe=None,
            train_dataframe_store_id=None,
            test_dataframe_store_id=None,
            train_sample_length=1000,
            test_sample_length=None,
            backend_env_class_ref=TradeEnvironment,
            reward_scale=1.0,
            dump_summary_modes=('test',),  #  ('train', 'test'),
            dump_summary_path='./text_summaries',
            name='BTgymEnv',
            **kwargs
    ):
        """

        Args:
            train_dataframe:
            test_dataframe:
            train_dataframe_store_id:
            test_dataframe_store_id:
            backend_env_class_ref:
            name:
            **kwargs:
        """
        if test_sample_length is None:
            test_sample_length = train_sample_length
        env_config = {
            'train': {
                'class_ref': backend_env_class_ref,
                'kwargs': {
                    'dataframe': train_dataframe,
                    'dataframe_store_id': train_dataframe_store_id,
                    'sample_length': train_sample_length,
                },
            },
            'test': {
                'class_ref': backend_env_class_ref,
                'kwargs': {
                    'dataframe': test_dataframe,
                    'dataframe_store_id': test_dataframe_store_id,
                    'sample_length': test_sample_length,
                },
            },
        }
        btgym_obs_space_wrapper = {
            # stat?
            # internal?
            'metadata': DictSpace(
                {
                    'type': spaces.Box(
                        shape=(),
                        low=0,
                        high=1,
                        dtype=np.uint32
                    ),
                    'trial_num': spaces.Box(
                        shape=(),
                        low=0,
                        high=1,
                        dtype=np.uint32
                    ),
                    'trial_type': spaces.Box(
                        shape=(),
                        low=0,
                        high=1,
                        dtype=np.uint32
                    ),
                    'sample_num': spaces.Box(
                        shape=(),
                        low=0,
                        high=1,
                        dtype=np.uint32
                    ),
                    'first_row': spaces.Box(
                        shape=(),
                        low=0,
                        high=1,
                        dtype=np.uint32
                    ),
                    'timestamp': spaces.Box(
                        shape=(),
                        low=0,
                        high=np.finfo(np.float64).max,
                        dtype=np.float64
                    ),
                }
            )
        }
        super().__init__(config=env_config, name=name, **kwargs)
        self.log.debug('env_config: {}'.format(env_config))
        try:
            assert self.envs['train'].observation_space.shape == self.envs['test'].observation_space.shape

        except AssertionError:
            msg = 'Expected identical observation space shapes for train and test envs., got: {} and {}'.\
                format(self.envs['train'].observation_space, self.envs['test'].observation_space)
            raise AssertionError(msg)

        try:
            assert self.envs['train'].action_space.shape == self.envs['test'].action_space.shape

        except AssertionError:
            msg = 'Expected identical action space shapes for train and test envs., got: {} and {}'.\
                format(self.envs['train'].action_space, self.envs['test'].action_space)
            raise AssertionError(msg)

        self.observation_space = DictSpace(btgym_obs_space_wrapper)
        for key, value in self.envs['test'].observation_space.spaces.items():
            self.observation_space.spaces[key] = value
            self.observation_space.shape[key] = value.shape

        self.action_space = self.envs['test'].action_space
        self.mode = 'train'
        self.render_modes = ['no_image_yet']  # btgym compat.
        self.asset_names = self.envs['test'].engine.market.state_config['assets'].columns

        self.reward_scale = reward_scale
        self.state_trajectory = None

        self.dump_summary_modes = dump_summary_modes

        if not os.path.exists(dump_summary_path):
            os.makedirs(dump_summary_path)

        self.portfolio_summary_path = dump_summary_path + '/{}_{}'.format(self.name, self.task) + '_{}_{}_portfolio.npy'
        self.order_summary_path = dump_summary_path + '/{}_{}'.format(self.name, self.task) + '_{}_{}_orders.npy'

    def get_initial_action(self):
        return {asset: 0 for asset in self.asset_names}

    def reset(self, episode_config, **kwargs):
        self.log.debug('_reset_state got ep_cfg: {}, kw: {}'.format(episode_config, kwargs))
        if not episode_config['sample_type']:
            # Train environment:
            self.mode = 'train'

        else:
            self.mode = 'test'

        init_obs = self.envs[self.mode].reset()
        self.state_trajectory = [self.envs[self.mode].engine.manager.state]
        return self.wrap_observation(init_obs)

    def step(self, action):
        o, r, d, i = self.envs[self.mode].step(action)
        self.state_trajectory.append(self.envs[self.mode].engine.manager.state)
        if d:
            if self.mode in self.dump_summary_modes:
                self.dump_summary_to_npy()

        return self.wrap_observation(o), r * self.reward_scale, d, i

    def get_stat(self):
        return self.envs[self.mode].get_stat()

    def render(self, *args, **kwargs):
        return self.envs[self.mode].render({'mode': []})

    def wrap_observation(self, obs):
        """BTGym.algorithms compatibility wrapper."""
        obs['metadata'] = {
            'type': self.mode == 'test',
            'trial_num': 0,
            'trial_type': self.mode == 'test',
            'sample_num': 0,
            'first_row': 0,
            'timestamp': int(self.mode == 'test'),
        }
        return obs

    def dump_summary_to_npy(self):
        p_summary, o_summary = self.parse_state_trajectory()
        p_path = self.portfolio_summary_path.format(self.mode, time.strftime('%H:%M:%S_%d_%b_%Y'))
        np.save(p_path, p_summary)
        o_path = self.order_summary_path.format(self.mode, time.strftime('%H:%M:%S_%d_%b_%Y'))
        np.save(o_path, o_summary)
        self.log.notice('Episode summaries saved as:\n{}\n{}'.format(p_path, o_path))

    def parse_state_trajectory(self):
        portfolio_summary = {
            'portfolio': [],
            'portfolio_value': [],
            'realised_return': [],
            'unrealised_return': [],
        }
        order_summary ={
            'step': [],
            'type': [],
            'size': [],
            'result': [],
        }

        for i, state in enumerate(self.state_trajectory):
            for key, stream in portfolio_summary.items():
                try:
                    stream.append(np.squeeze(np.asarray(list(state[key].values()))))

                except (TypeError, AttributeError) as e:
                    stream.append(np.squeeze(np.asarray(state[key])))

            for order in state['order']:
                order_summary['step'].append(i)
                order_summary['size'].append(order.size)
                order_summary['result'].append(order.result)
                if order.type == 'buy':
                    order_summary['type'].append(1)
                elif order.type == 'sell':
                    order_summary['type'].append(-1)
                else:
                    order_summary['type'].append(0)

        for d in [portfolio_summary, order_summary]:
            for k, v in d.items():
                d[k] = np.squeeze(np.asarray(v))

        return portfolio_summary, order_summary



