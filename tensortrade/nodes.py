from .core import Node
from engines.base import BasePandasIterator, BasePortfolioManager


class PandasMarketDataNode(Node):

    def __init__(self, name='PdMarketDataIterator', **kwargs):
        super().__init__(
            kernel_class_ref=BasePandasIterator,
            name=name,
            **kwargs
        )


class PortfolioManagerNode(Node):

    def __init__(self, name='PortfolioManager', **kwargs):
        super().__init__(
            kernel_class_ref=BasePortfolioManager,
            name=name,
            **kwargs
        )

