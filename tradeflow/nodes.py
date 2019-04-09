from .core import Node
from tradeflow.kernels.base import BasePandasIterator, BasePortfolioManager, ActionToMarketOrder, IdentityKernel


class Identity(Node):
    """
    Practical purpose of this node is that when placed locally after any remotedly computed node,
    it forces remote inputs evaluation and returns actual computed state.
    """
    def __init__(self, name='IdentityNode', **kwargs):
        super().__init__(
            kernel_class_ref=IdentityKernel,
            name=name,
            **kwargs
        )


class PandasMarketData(Node):
    """
    Basic iterative market data provider.
    """
    def __init__(self, name='PdMarketDataIterator', **kwargs):
        super().__init__(
            kernel_class_ref=BasePandasIterator,
            name=name,
            **kwargs
        )


class PortfolioManager(Node):
    """
    Basic broker simulator.
    """
    def __init__(self, name='PortfolioManager', **kwargs):
        super().__init__(
            kernel_class_ref=BasePortfolioManager,
            name=name,
            **kwargs
        )


class ActionToOrder(Node):
    """
    Maps abstract MDP actions to PortfolioManger specific orders
    """
    def __init__(self, name='ActionMapper', **kwargs):
        super().__init__(
            kernel_class_ref=ActionToMarketOrder,
            name=name,
            **kwargs
        )

