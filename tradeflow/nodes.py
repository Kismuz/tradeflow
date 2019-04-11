from .core import Node
from tradeflow.kernel.base import IdentityKernel, CheckIfDone, StateToDictSpace, StateToBoxSpace, StateToFlatSpace
from tradeflow.kernel.manager import BasePortfolioManager
from tradeflow.kernel.action import AssetActionToMarketOrder, DiscreteActionToMarketOrder
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


class DiscreteActionToOrder(Node):
    """
    Maps  MDP actions from gym.spaces.Discrete to PortfolioManger specific orders.
    """
    def __init__(self, name='DiscreteActionMapper', **kwargs):
        super().__init__(
            kernel_class_ref=DiscreteActionToMarketOrder,
            name=name,
            **kwargs
        )


class AssetActionToOrder(Node):
    """
    Maps MDP actions from btgym.spaces.ActionDictSpace to PortfolioManger specific orders.

    """
    def __init__(self, name='AssetActionMapper', **kwargs):
        super().__init__(
            kernel_class_ref=AssetActionToMarketOrder,
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


class ToDictSpace(Node):
    """
    Maps state to instance of btgym.spaces.DictSpace
    """
    def __init__(self, name='StateToDictSpace', **kwargs):
        super().__init__(
            kernel_class_ref=StateToDictSpace,
            name=name,
            **kwargs
        )


class ToBoxSpace(Node):
    """
    Maps state to instance of gym.spaces.BoxSpace
    """
    def __init__(self, name='StateToBoxSpace', **kwargs):
        super().__init__(
            kernel_class_ref=StateToBoxSpace,
            name=name,
            **kwargs
        )


class ToFlatSpace(Node):
    """
    Maps state to single-dimensional gym.spaces.BoxSpace
    """
    def __init__(self, name='StateToFlatSpace', **kwargs):
        super().__init__(
            kernel_class_ref=StateToFlatSpace,
            name=name,
            **kwargs
        )