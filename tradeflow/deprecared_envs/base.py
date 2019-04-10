from logbook import Logger, StreamHandler, WARNING, NOTICE, INFO, DEBUG
import sys
import time
import copy
from datetime import timedelta
from collections import OrderedDict
import numpy as np


class BaseEnvironment(object):

    def __init__(
            self,
            engine_class_ref,
            engine_config,
            action_space=None,
            observation_space=None,
            name='BaseEnvironment',
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
            self.log_level = self.log.level

        self.action_space = action_space
        self.observation_space = observation_space

        self.engine = engine_class_ref(task=self.task, log=self.log, **engine_config)
        self.done = True
        self.closed = True
        self.step_counter = None
        self.start_time = 0
        self.elapsed_time = 0
        self.elapsed_length = 0

    def start(self, *args, **kwargs):
        self.closed = False

    def reset(self, *args, **kwargs):
        self.engine.stop(*args, **kwargs)
        self.engine._start(*args, **kwargs)
        self.done = False
        self.step_counter = 0

        return self.get_state()

    def step(self, action):
        if not self.done:
            self.engine._update_state(inputs=action)
            obs = self.get_state()
            reward = self.get_reward()
            info = self.get_info()
            self.done = self.get_done()
            self.step_counter += 1
            if self.done:
                self.elapsed_time = timedelta(seconds=time.time() - self.start_time)
                self.elapsed_length = copy.copy(self.step_counter)

        else:
            msg = 'Environment either exhausted or has not been initialised. Hint: forgot to call ._reset_state()?'
            self.log.error(msg)
            raise RuntimeError(msg)

        return obs, reward, self.done, info

    def render(self, mode):
        return (np.random.rand(100, 200, 3) * 255).astype(dtype=np.uint8)

    def close(self, **kwargs):
        pass

    def get_state(self):
        return 0

    def get_reward(self):
        return 0

    def get_info(self):
        return '0'

    def get_done(self):
        return 0

    def get_stat(self):
        return {
            'runtime': self.elapsed_time,
            'length': self.elapsed_length,
        }


class BaseNestedEnvironment(object):

    def __init__(self, config, name='BaseEnvironment', task=0, log=None, log_level=INFO, **kwargs):
        """

        Args:
            config:         structured configuration dictionary
            name:           strring id
            task:           numerical id
            log:            logbook.Logger instance or None
            log_level:      logbook loglevel or int
            **kwargs:       common configuration parameters
        """
        self.env_config = config
        self.name = name
        self.task = task

        if log is None:
            StreamHandler(sys.stdout).push_application()
            self.log_level = log_level
            self.log = Logger('{}_{}'.format(self.name, self.task), level=self.log_level)

        else:
            self.log = log
            self.log_level = self.log.level

        self.common_kwargs = kwargs
        try:
            self.action_space = kwargs['action_space']

        except KeyError:
            self.action_space = None

        try:
            self.observation_space = kwargs['observation_space']

        except KeyError:
            self.observation_space = None

        self.closed = True
        self.envs = self._make_envs(self.env_config, self.name)

    def _make_envs(self, config, name_suffix):
        # self.log.debug('got config: {}, named: {}'.format(config, name_suffix))
        try:
            assert isinstance(config, dict)

        except AssertionError:
            msg = 'Expected `config` as dictionary, got: {}'.format(type(config))
            self.log.error(msg)
            raise TypeError(msg)
        if 'class_ref' in set(config.keys()):
            try:
                assert 'kwargs' in set(config.keys())
                kwargs = config['kwargs']
                kwargs.update(self.common_kwargs)

            except AssertionError:
                self.log.warning('Expected `kwargs` key for {} not found.'.format(config['class_ref']))
                kwargs = {}

            return config['class_ref'](
                name=self.name + '_' + name_suffix,
                task=self.task,
                log_level=self.log_level,
                **kwargs,
            )

        else:
            return_dict = OrderedDict()
            for key, sub_config in config.items():
                return_dict[key] = self._make_envs(config=sub_config, name_suffix=key)

            return return_dict

    def map_method(self, struct, method_name, kwargs_struct=None):
        if isinstance(struct, dict):
            r_dict = {}
            for key, sub_struct in struct.items():
                if kwargs_struct is not None:
                    apply_kwargs = kwargs_struct[key]

                else:
                    apply_kwargs = None

                r_dict[key] = self.map_method(sub_struct, method_name, apply_kwargs)

            return r_dict

        else:
            if kwargs_struct is None:
                apply_kwargs = {}

            else:
                apply_kwargs = kwargs_struct

            method = getattr(struct, method_name)
            self.log.debug('got method: {} with kwargs: {}'.format(method, apply_kwargs))

            return method(**apply_kwargs)

    def start(self, kwargs_struct=None):
        self.closed = False
        return self.map_method(self.envs, 'start', kwargs_struct)

    def close(self, kwargs_struct=None):
        self.closed = True
        return self.map_method(self.envs, 'close', kwargs_struct)

    def reset(self, kwargs_struct=None):
        return self.map_method(self.envs, '_reset_state', kwargs_struct)

    def step(self, action_struct):
        return self.map_method(self.envs, 'step', action_struct)

    def render(self, modes_struct):
        return self.map_method(self.envs, 'render', modes_struct)

    def get_stat(self):
        return self.map_method(self.envs, 'get_stat')




