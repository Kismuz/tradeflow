from .core import Node
from tradeflow.kernel.base import IdentityKernel, CheckIfDone, StateToObservation
from tradeflow.kernel.manager import  BasePortfolioManager
from tradeflow.kernel.action import ActionToMarketOrder
from tradeflow.kernel.reward import ClosedTradeRewardFn
from tradeflow.kernel.iterator import PandasMarketEpisodeIterator, PandasMarketStepIterator


class Identity(Node):
    """
    Maps input to output.
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
    Basic market episode data provider.
    Randomly samples incoming dataset.
    """
    def __init__(self, name='PdMarketEpisodeIterator', **kwargs):
        super().__init__(
            kernel_class_ref=PandasMarketEpisodeIterator,
            name=name,
            **kwargs
        )


class PandasMarketStep(Node):
    """
    Basic iterative step-by-step market data provider.
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


class TradeReward(Node):
    """
    Basic reward function.
    """
    def __init__(self, name='ClosedTradeRewardFn', **kwargs):
        super().__init__(
            kernel_class_ref=ClosedTradeRewardFn,
            name=name,
            **kwargs
        )


class Done(Node):
    """
    Checks termination condition.
    """
    def __init__(self, name='CheckIfDone', **kwargs):
        super().__init__(
            kernel_class_ref=CheckIfDone,
            name=name,
            **kwargs
        )


class Observation(Node):
    """
    Maps states to [possibly nested] observation tensors
    """
    def __init__(self, name='StateToObservation', **kwargs):
        super().__init__(
            kernel_class_ref=StateToObservation,
            name=name,
            **kwargs
        )