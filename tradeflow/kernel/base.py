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


class StateToObservation(Kernel):
    """
    Maps dictionary of heterogeneous input states to observation tensor.
    Supports two modes
    """
    def __init__(
            self,
            space_config,
            clip=100.0,
            name='StateToObservation',
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
                state[key] = StateToObservation.get_state(input_state[key], space)

        elif isinstance(observation_space, spaces.Box):
            state = StateToObservation.get_values(input_state)
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
                state[key] = StateToObservation.get_values(value)

        else:
            if isinstance(input_state, DataFrame):
                state = input_state.values

            else:
                state = np.asarray(input_state)

        return state

    def update_state(self, input_state):
        self.state = self.get_state(input_state, self.space)
        return self.state

    # TODO: make flatten kernel - flatten and unpack dictionary
