from logbook import INFO
import sys
import copy
import numpy as np
from pandas import DataFrame
from collections import OrderedDict

from btgym.spaces import DictSpace
from gym import spaces
from ..core import Kernel

import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")


class IdentityKernel(Kernel):
    """
    Maps input trough.
    """
    def __init__(
            self,
            name='Identity',
            task=0,
            log=None,
            log_level=INFO,
    ):
        super().__init__(name=name, task=task, log=log, log_level=log_level)
        self.state = None

    def update_state(self, input_state):
        self.state = input_state
        return self.state


class CheckIfDone(Kernel):
    """
    Returns termination flag.
    """
    def __init__(
            self,
            name='CheckIfDone',
            pass_input_state=False,
            task=0,
            log=None,
            log_level=INFO,
    ):
        super().__init__(name=name, task=task, log=log, log_level=log_level)
        self.pass_input_state = pass_input_state
        self.state = True

    def update_state(self, input_state):
        try:
            self.state = not input_state['ready']

        except KeyError:
            e = 'Expected key `ready` not found in input state'
            self.log.error(e)
            raise ValueError(e)

        if self.pass_input_state:
            return self.state, input_state

        else:
            return self.state


class StateToDictSpace(Kernel):
    """
    Maps dictionary of heterogeneous inputs to btgym.spaces.DictSpace.
    """
    def __init__(
            self,
            space_config,
            clip=100.0,
            name='StateToDictSpace',
            task=0,
            log=None,
            log_level=INFO,
    ):
        super().__init__(name=name, task=task, log=log, log_level=log_level)
        self.space_config = space_config
        self.clip = abs(clip)
        self.space = self.make_observation_space(self.space_config)

    def make_observation_space(self, observation_shape):
        if isinstance(observation_shape, dict):
            spec = {}
            for key, value in observation_shape.items():
                spec[key] = self.make_observation_space(value)

            space = DictSpace(spec)

        else:
            space = spaces.Box(shape=list(observation_shape), high=self.clip, low=- self.clip, dtype=np.float32,)
        return space

    @staticmethod
    def get_state(input_state, observation_space):
        if isinstance(observation_space, DictSpace):
            state = {}
            for key, space in observation_space.spaces.items():
                state[key] = StateToDictSpace.get_state(input_state[key], space)

        elif isinstance(observation_space, spaces.Box):
            state = StateToDictSpace.get_values(input_state)
            # print('i_s: {}\no_s: {}'.format(input_state, observation_space))
            # if isinstance(input_state, DataFrame):
            #     state = input_state.values
            #
            # elif isinstance(input_state, dict): # or isinstance(input_state, OrderedDict):
            #     state = np.asarray(list(input_state.values()))
            #
            # else:
            #     state = np.asarray(input_state)

        else:
            e = 'Unsupported observation space type {}'.format(type(observation_space))
            raise TypeError(e)

        return state

    @staticmethod
    def get_values(input_state):
        if isinstance(input_state, dict):
            state = {}
            for key, value in input_state.items():
                state[key] = StateToDictSpace.get_values(value)

        else:
            if isinstance(input_state, DataFrame):
                state = input_state.values

            else:
                state = np.asarray(input_state)

        return state

    def update_state(self, input_state):
        self.state = self.get_state(input_state, self.space)
        return self.state


class StateToBoxSpace(Kernel):
    """
    Maps inputs to gym.spaces.Box.
    """
    # TODO: implement: flatten and attempt to stack (maybe with replication) dictionary values along specified axis
    def __init__(
            self,
            shape,
            clip=100.0,
            dtype=np.float32,
            name='StateToBoxSpace',
            task=0,
            log=None,
            log_level=INFO,
    ):
        super().__init__(name=name, task=task, log=log, log_level=log_level)
        self.shape = shape
        self.clip = abs(clip)
        self.dtype = dtype
        self.space = spaces.Box(shape=list(self.shape), high=self.clip, low=- self.clip, dtype=self.dtype)

    @staticmethod
    def get_values(input_state):
        if isinstance(input_state, dict):
            e = 'Mapping dictionaries to gym.spaces.Box not implemented yet.'
            raise NotImplementedError(e)

        else:
            if isinstance(input_state, DataFrame):
                state = input_state.values

            else:
                state = np.asarray(input_state)

        return state

    def update_state(self, input_state):
        # print('got_state: ', input_state)
        self.state = self.get_values(input_state)
        return self.state


class StateToFlatSpace(Kernel):
    """
    Maps inputs to 1-dimensional gym.spaces.Box.
    """
    def __init__(
            self,
            shape,
            clip=100.0,
            dtype=np.float32,
            name='StateToBoxSpace',
            task=0,
            log=None,
            log_level=INFO,
    ):
        super().__init__(name=name, task=task, log=log, log_level=log_level)
        try:
            assert isinstance(shape, int)

        except AssertionError:
            e = 'Expected 1D shape as integer value, got: {}'.format(shape)
            self.log.error(e)
            raise ValueError(e)

        self.shape = shape
        self.clip = abs(clip)
        self.dtype = dtype
        self.space = spaces.Box(shape=(self.shape,), high=self.clip, low=- self.clip, dtype=self.dtype)

    @staticmethod
    def get_values(input_state):
        if isinstance(input_state, dict):
            e = 'Mapping dictionaries to gym.spaces.Box not implemented yet.'
            raise NotImplementedError(e)

        else:
            if isinstance(input_state, DataFrame):
                state = input_state.values

            else:
                state = np.asarray(input_state)

        state = state.flatten()

        return state

    def update_state(self, input_state):
        # print('got_state: ', input_state)
        self.state = self.get_values(input_state)
        return self.state