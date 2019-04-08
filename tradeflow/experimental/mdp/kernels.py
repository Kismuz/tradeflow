from .core import Kernel
import numpy as np


# Toy kernel examples


class Constant(Kernel):
    def __init__(self,  value, name='ConstantKernel',log=None):
        super().__init__(name=name, log=log)
        self.value = value

    def compute(self, **inputs):
        return self.value


class RandomUniformConstant(Kernel):
    def __init__(self, low=0, high=1, size=(), name='RndUniformKernel', log=None):
        super().__init__(name=name, log=log)
        self.low = low
        self.high = high
        self.size = size

    def compute(self, **inputs):
        return np.random.uniform(low=self.low, high=self.high, size=self.size)


class Sum(Kernel):
    def __init__(self, name='SumKernel', **kwargs):
        super().__init__(name=name, **kwargs)

    def compute(self, **inputs):
        return np.sum(list(inputs.values()))


class Iterator(Kernel):
    def __init__(self, name='IteratorKernel',**kwargs):
        super().__init__(name=name, **kwargs)
        self.start = None
        self.stop = None
        self.stride = None
        self.value = None

    def initialize(self, start=0, stop=1, stride=1):
        print('iter_kernel_init:', start, stop, stride)
        self.start = start
        self.stop = stop
        self.stride = stride
        self.value = start

    def compute(self, **inputs):
        self.value += self.stride
        print('iter_kernel_comp:', self.value)
        if self.value > self.stop:
            raise StopIteration('Iterator {name} exhausted'.format(name=self.name))

        return self.value
