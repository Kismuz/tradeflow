from .core import Node
from tradeflow.kernels.base import BasePandasIterator, BasePortfolioManager, ActionToMarketOrder


class PandasMarketData(Node):

    def __init__(self, name='PdMarketDataIterator', **kwargs):
        super().__init__(
            kernel_class_ref=BasePandasIterator,
            name=name,
            **kwargs
        )


class PortfolioManager(Node):

    def __init__(self, name='PortfolioManager', **kwargs):
        super().__init__(
            kernel_class_ref=BasePortfolioManager,
            name=name,
            **kwargs
        )


class ActionToOrder(Node):

    def __init__(self, name='ActionMapper', **kwargs):
        super().__init__(
            kernel_class_ref=ActionToMarketOrder,
            name=name,
            **kwargs
        )

