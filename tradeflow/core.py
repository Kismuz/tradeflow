from logbook import Logger, StreamHandler, WARNING, NOTICE, INFO, DEBUG
import sys

import pythonflow as pf


class Kernel(object):
    """
    Base stateful execution backend class.
    Todo: local /remote
    """

    def __init__(
            self,
            name='BaseEngine',
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
            self.log_level = None

        self.init_state = None
        self.state = None
        self.ready = False

    def start(self, *args, **kwargs):
        self.ready = True

    def stop(self, *args, **kwargs):
        self.ready = False
        pass

    def update_state(self, *args, **kwargs):
        pass


class StateOperation(pf.Operation):
    """
    This class implements kernel to model graph connection by providing an operation (graph node).
    """
    def __init__(
            self,
            kernel,
            reset,
            name='UpdateOrResetStateOperation',
            length=None,
            graph=None,
            dependencies=None,
            **inputs
    ):
        super().__init__(reset, name=name, length=length, graph=graph, dependencies=dependencies, **inputs)
        self.kernel = kernel

    def _evaluate(self, reset, **inputs):
        if reset:
            self.kernel.start(**inputs)

        else:
            self.kernel.update_state(**inputs)

        return self.kernel.state


class Node(object):
    """
    Connects stateful computation object (kernel) to dataflow model
    """
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

    def __call__(self, reset, length=None, graph=None, dependencies=None, **inputs):
        """
        StateOperation constructor.

        Args:
            reset:
            length:
            graph:
            dependencies:
            **inputs:

        Returns:
            instance of StateOperation
        """
        return StateOperation(
            reset=reset,
            kernel=self.kernel,
            name=self.name + '_state_op',
            length=length,
            graph=graph,
            dependencies=dependencies,
            **inputs
        )

