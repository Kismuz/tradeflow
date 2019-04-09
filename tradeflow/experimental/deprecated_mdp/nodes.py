from .core import Node
from ..deprecated_mdp import kernels as k


# Toy nodes


class Constant(Node):
    def __init__(self, value, name='ConstantNode', log=None):
        super().__init__(name=name, log=log)
        self.kernel = k.Constant(value, log=self.log)


class RandomUniformConstant(Node):
    def __init__(self, name='RndUniformConstantNode', log=None, **kwargs):
        super().__init__(name=name, log=log)
        self.kernel = k.RandomUniformConstant(log=log, **kwargs)


class Sum(Node):
    def __init__(self, name='SumNode', log=None, **inputs):
        super().__init__(name=name, log=log, **inputs)
        self.kernel = k.Sum()


class Iterator(Node):
    def __init__(self, name='IteratorNode', log=None):
        super().__init__(name=name, log=log)
        self.kernel = k.Iterator()

    def _reset_state(self, feed_dict):
        if not self._locked:
            self.kernel.initialize(**feed_dict)
            self._locked = True




