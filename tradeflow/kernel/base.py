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
