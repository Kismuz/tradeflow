from logbook import INFO
import sys
import copy

import numpy as np
from collections import namedtuple
from ..core import Kernel
# from ..kernel.base import PandasStateConfig

import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")


PandasStateConfig = namedtuple('PandasStateConfig', ['columns', 'depth'])


class PandasMarketEpisodeIterator(Kernel):
    """
    Samples episodes from pandas dataset.
    """
    def __init__(
            self,
            name='MarketDataEpisodeIterator',
            task=0,
            log=None,
            log_level=INFO,
            ):
        super().__init__(name=name, task=task, log=log, log_level=log_level)
        self.dataframe = None
        self.iterations = 0
        self.pn = 0

    def update_state(self, input_state, reset, sample_length):
        self.dataframe = input_state

        if reset:
            self.state = self.sample(sample_length)

            return self.state

        else:
            return None

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


class PandasMarketStepIterator(Kernel):

    def __init__(
            self,
            state_config,
            name='MarketDataStepIterator',
            task=0,
            log=None,
            log_level=INFO,
    ):
        super().__init__(name=name, task=task, log=log, log_level=log_level)
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

            except AssertionError:
                e = 'Expected `state_config` be instance of {}, got: {}'.format(PandasStateConfig, type(state_config))
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

    def update_state(self, input_state, reset):
        if reset:
            self._start(input_state)

        self._update_state()

        return self.state

    def _start(self, dataframe):
        self.dataframe = dataframe
        self.log.debug('got data source of type: {}'.format(type(self.dataframe)))
        self.log.debug('got data source of shape: {}'.format(self.dataframe.values.shape))
        self.data_length = self.dataframe.values.shape[0]
        self.start_pointer = self.sample_max_depth
        self.iter_passed = 0
        self.ready = True

    def _update_state(self):
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
            self.state['ready'] = self.ready
            return self.state

        else:
            msg = 'Attempt to iterate uninitialised data / go beyond sample length.\nHint: forgot to check .ready flag?'
            self.log.error(msg)
            raise IndexError(msg)
