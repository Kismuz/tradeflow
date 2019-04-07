from logbook import Logger, StreamHandler, WARNING, NOTICE, INFO, DEBUG
import sys

import pythonflow as pf
import numpy as np
import pandas as pd


class ResetOperation(pf.Operation):

    def __init__(
            self,
            kernel,
            name='ResetOperation',
            length=None,
            graph=None,
            dependencies=None,

            **inputs
    ):
        super().__init__(name=name, length=length, graph=graph, dependencies=dependencies, **inputs)
        self.kernel = kernel

    def _evaluate(self, **inputs):
        self.kernel.start(**inputs)
        return self.kernel.state


class UpdateStateOperation(pf.Operation):

    def __init__(
            self,
            kernel,
            name='UpdateStateOperation',
            length=None,
            graph=None,
            dependencies=None,

            **inputs
    ):
        super().__init__(name=name, length=length, graph=graph, dependencies=dependencies, **inputs)
        self.kernel = kernel

    def _evaluate(self, **inputs):
        self.kernel.update_state(**inputs)
        return self.kernel.state


class Node(object):

    def __init__(
            self,
            kernel_class_ref,
            name='BaseNode',
            task=0,
            log=None,
            log_level=INFO,
            **kernel_kwargs
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

        self.kernel = kernel_class_ref(
            log=self.log,
            task=task,
            log_level=log_level,
            name=name + '/kernel',
            **kernel_kwargs
        )

    def reset(self, length=None, graph=None, dependencies=None, **inputs):
        """
        Reset operation constructor.

        Args:
            length:
            graph:
            dependencies:
            **inputs:

        Returns:
            instance of ResetOperation
        """

        return ResetOperation(
            kernel=self.kernel,
            name=self.name + '_reset_op',
            length=length,
            graph=graph,
            dependencies=dependencies,
            **inputs
        )

    def update_state(self, length=None, graph=None, dependencies=None, **inputs):
        """
        UpdateStateOperation pf operation constructor.

        Args:
            length:
            graph:
            dependencies:
            **inputs:

        Returns:
            instance of UpdateStateOperation constructor
        """

        return UpdateStateOperation(
            kernel=self.kernel,
            name=self.name + '_update_state_op',
            length=length,
            graph=graph,
            dependencies=dependencies,
            **inputs
        )

