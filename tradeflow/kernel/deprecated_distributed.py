from logbook import Logger, StreamHandler, WARNING, NOTICE, INFO, DEBUG
import sys

import numpy as np
import copy

from .base import Kernel, BaseTradeKernel, PandasStateConfig
from ..deprecared_envs.gen1 import TradeEnvironment, BTgymCompatibleEnvironment

import ray

from ray.tune.util import pin_in_object_store, get_pinned_object


# @ray.remote
class DataServer:
    def __init__(
            self,
            dataframe,
            name='RayDataServer',
            task=0,
            log=None,
            log_level=INFO,
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
        self.dataframe = ray.get(dataframe)

        self.iterations = 0
        self.pn = 0

    # @ray.method(num_return_vals=1)
    def ping(self, task):
        self.pn += 1
        m = 'returning ping from task: {} #{}'.format(task, self.pn)

        self.log.debug(m)
        return m

    # @ray.method(num_return_vals=1)
    def sample(self, sample_length):
        self.log.debug('sample #{}'.format(self.iterations))
        try:
            assert sample_length <= self.dataframe.shape[0]

        except AssertionError as e:
            e = 'Expected sample be shorter than data length, got: {} and {}'.format(
                sample_length, self.dataframe.shape[0]
            )
            self.log.error(e)
            raise AssertionError(e)

        if sample_length > 0:
            sample_start_interval = dict(
                low=0,
                high=self.dataframe.shape[0] - sample_length + 1
            )
        else:
            sample_length = self.dataframe.shape[0]
            sample_start_interval = dict(
                low=0,
                high=1,
            )
        start_pointer = np.random.randint(**sample_start_interval)
        self.log.debug(
            'sample start: {}, end: {}, len: {}'.format(start_pointer, start_pointer + sample_length, sample_length)
        )
        self.iterations += 1
        return copy.copy(self.dataframe.loc[start_pointer: start_pointer + sample_length - 1])


class DistributedPandasIterator(Kernel):

    def __init__(
            self,
            dataframe,
            sample_length,
            state_config,
            name='MarketIterator',
            **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.dataserver = dataframe
        self.log.debug('*MarketIterator got DS: {}'.format(self.dataserver))
        self.sample_length = sample_length
        self.data_length = None
        self.state_config = state_config

        self.dataframe = None
        self.start_pointer = None
        self.sample_max_depth = self.get_max_depth(self.state_config)

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

    def _start(self):
        # for i in range(4):
        #     self.log.warning('pinging ds...')
        #     #p = self.dataserver.ping.remote(task=self.task)
        #     p = self.dataserver.ping(task=self.task)
        #     self.log.warning('...ds returned pong id: {}'.format(p))
        #     #self.log.warning('...ds returned pong: {}'.format(ray.get(p)))
        #     time.sleep(.1)

        if self.sample_length > 0:
            # self.sample_id = self.dataserver.sample.remote(sample_length=self.sample_length + self.sample_max_depth)
            self.sample_id = self.dataserver.sample(sample_length=self.sample_length + self.sample_max_depth)
        else:
            # self.sample_id = self.dataserver.sample.remote(sample_length=-1)
            self.sample_id = self.dataserver.sample(sample_length=-1)

        # self.log.debug('start: got sample id: {}'.format(self.sample_id))

        # self.dataframe = ray.get(self.sample_id)
        self.dataframe = self.sample_id
        self.log.debug('got data source of shape: {}'.format(self.dataframe.values.shape))
        self.data_length = self.dataframe.values.shape[0]
        self.start_pointer = self.sample_max_depth
        self.iter_passed = 0
        self.ready = True
        self._update_state()

    def _update_state(self, inputs=None, *args, **kwargs):
        if self.ready:
            self.state = self.get_state(self.start_pointer + self.iter_passed, self.state_config)
            self.iter_passed += 1

            if self.iter_passed >= self.data_length - self.sample_max_depth:
                self.ready = False

            self.log.debug(
                'market iteration {} of {}, ready: {}'.format(
                    self.iter_passed,
                    self.data_length - self.sample_max_depth,
                    self.ready
                )
            )

        else:
            msg = 'Attempt to iterate uninitialised data / go beyond sample length.\nHint: forgot to check .ready flag?'
            self.log.error(msg)
            raise IndexError(msg)


class BaseTradeEngineD1(BaseTradeKernel):

    def __init__(self, name='TradeEngineD1', **kwargs):
        super().__init__(market_engine_ref=DistributedPandasIterator, name=name, **kwargs)


class TradeEnvironmentD1(TradeEnvironment):

    def __init__(self, name='TradeEnvironmentD1', **kwargs):
        super().__init__(engine_class_ref=BaseTradeEngineD1, name=name, **kwargs)


class BTgymCompatibleEnvironmentD1(BTgymCompatibleEnvironment):

    def __init__(self,  train_dataframe_store_id,  test_dataframe_store_id,  name='BTgymEnvD1',  **kwargs):
        super().__init__(
            train_dataframe=train_dataframe_store_id,
            test_dataframe=test_dataframe_store_id,
            backend_env_class_ref=TradeEnvironmentD1,
            name=name,
            **kwargs
        )
