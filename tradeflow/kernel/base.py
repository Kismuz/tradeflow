from logbook import INFO
import sys

from collections import namedtuple
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

    def update_state(self, inputs):
        self.state = inputs
        return self.state


class CheckIfDone(Kernel):
    """
    Returns termination flag.
    """
    def __init__(
            self,
            name='CheckIfDone',
            task=0,
            log=None,
            log_level=INFO,
    ):
        super().__init__(name=name, task=task, log=log, log_level=log_level)
        self.state = True

    def update_state(self, state):
        try:
            self.state = not state['ready']

        except KeyError:
            e = 'Expected key `ready` not found in input state'
            self.log.error(e)
            raise ValueError(e)

        return self.state