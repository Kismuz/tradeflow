from .core import Node
from tradeflow.kernel.base import IdentityKernel
from tradeflow.kernel.manager import ActionToMarketOrder, BasePortfolioManager
from tradeflow.kernel.iterator import PandasMarketEpisodeIterator, PandasMarketStepIterator


class Identity(Node):
    """
    Practical purpose of this node is as follows: when placed locally after any remotely executed node,
    it forces remote inputs evaluation and returns actual computed state.
    """
    def __init__(self, name='IdentityNode', **kwargs):
        super().__init__(
            kernel_class_ref=IdentityKernel,
            name=name,
            **kwargs
        )


class PandasMarketEpisode(Node):
    """
    Basic iterative market episode data provider.
    """
    def __init__(self, name='PdMarketEpisodeIterator', **kwargs):
        super().__init__(
            kernel_class_ref=PandasMarketEpisodeIterator,
            name=name,
            **kwargs
        )


class PandasMarketStep(Node):
    """
    Basic iterative market data provider.
    """
    def __init__(self, name='PdMarketDataIterator', **kwargs):
        super().__init__(
            kernel_class_ref=PandasMarketStepIterator,
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

