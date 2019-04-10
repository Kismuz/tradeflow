from logbook import Logger, StreamHandler, WARNING, NOTICE, INFO, DEBUG
import sys
from enum import Enum

import pythonflow as pf
import ray


class KernelDevice(Enum):
    """
    Defines way to carry actual kernel logic computations.
    Modes currently supported:
    1 - local in-process execution
    2 - distributed execution as ray.remote task
    """
    # TODO: add kwargs pass-through
    LOCAL = 1
    RAY = 2


class Kernel(object):
    """
    Base stateful execution backend class.
    Encapsulates actual computations to get node state.
    """

    def __init__(
            self,
            name='BaseExecutionKernel',
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

        self.init_state = None
        self.state = None
        self.ready = False

    def update_state(self, *args, **kwargs):
        return self.state


class GetStateOperation(pf.Operation):
    """
    This class implements node in-graph connectivity  by making an operation which returns actual node state.
    """
    def __init__(
            self,
            kernel,
            kernel_device,
            name='BaseUpdateOrResetStateOperation',
            length=None,
            graph=None,
            dependencies=None,
            **inputs
    ):
        super().__init__(name=name, length=length, graph=graph, dependencies=dependencies, **inputs)
        self.kernel = kernel
        self.kernel_device = kernel_device

    def _evaluate(self, **inputs):
        # Inputs can be either python objects or ray object store id's (in case dependent node's kernel were ray tasks)
        # and should be treated accordingly:
        # if current kernel is local one - we should get actual input values via ray.get() methods; pass ray Id's
        # as is otherwise:

        # self.log.debug(self.name, inputs)
        if self.kernel_device == KernelDevice.LOCAL:
            normalized_inputs = self._get_remote_inputs(**inputs)
            return self.kernel.update_state(**normalized_inputs)

        else:
            return self.kernel.update_state.remote(**inputs)

    @staticmethod
    def _get_remote_inputs(**inputs):
        """
        Substitutes remote ray.object Id's (if any) with actual values
        """
        for key, value in inputs.items():
            if isinstance(value, ray._raylet.ObjectID):
                inputs[key] = ray.get(value)

        return inputs


class Node(object):
    """
    Base model building block.
    Encapsulates stateful computation object via kernel and dataflow graph connectivity via StateOperation.
    TODO: ? maybe define dedicated State class ~ tf.Tensor-like
    """
    def __init__(
            self,
            kernel_class_ref,
            device=None,
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

        if device is None:
            # Make it LOCAL by default:
            self.kernel_device = KernelDevice(1)

        else:
            try:
                assert isinstance(device, KernelDevice)

            except AssertionError:
                e = 'Expected `device` be instance of {}, got: {}'.format(KernelDevice, device)
                self.log.error(e)
                raise TypeError(e)
            self.kernel_device = device

        # Instantiate kernel depending on execution device placement specification
        # (currently - either local or execution via Ray engine):
        if self.kernel_device == KernelDevice.RAY:
            try:
                assert ray.is_initialized()

            except AssertionError as e:
                self.log.error('Ray should be initialized before defining Node Kernel  as Ray.remote task')
                raise Exception(e)

            # Make remote ray actor out of kernel klass:
            kernel_actor_class_ref = ray.remote(kernel_class_ref)
            # TODO: add ray.remote kwargs

            self.kernel = kernel_actor_class_ref.remote(
                log=self.log,
                task=task,
                log_level=log_level,
                name=name + '/remote_kernel',
                **kernel_kwargs
            )
        elif self.kernel_device == KernelDevice.LOCAL:
            self.kernel = kernel_class_ref(
                log=self.log,
                task=task,
                log_level=log_level,
                name=name + '/kernel',
                **kernel_kwargs
            )
        else:
            raise ValueError('Unsupported KernelDevice: {}'.format(self.kernel_device))

    def __call__(self, length=None, graph=None, dependencies=None, **inputs):
        """
        StateOperation constructor. Provides graph connectivity

        Args:
            length:
            graph:
            dependencies:
            **inputs:

        Returns:
            instance of StateOperation
        """
        if self.kernel_device == KernelDevice.RAY:
            name_suffix = '_state_op_remote'

        elif self.kernel_device == KernelDevice.LOCAL:
            name_suffix = '_state_op_local'

        else:
            name_suffix = '_state_op_WTF?'
            
        return GetStateOperation(
            kernel=self.kernel,
            kernel_device=self.kernel_device,
            name=self.name + name_suffix,
            length=length,
            graph=graph,
            dependencies=dependencies,
            **inputs
        )

