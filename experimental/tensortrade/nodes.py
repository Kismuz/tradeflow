from .core import Node, ResetOperation, UpdateStateOperation
from ...engines.base import BasePandasIterator, BasePortfolioManager


class PandasMarketDataNode(Node):

    def __init__(self, name='PdMarketDataIterator', **kwargs):
        super().__init__(
            kernel_class_ref=BasePandasIterator,
            name=name,
            **kwargs
        )

    def reset(self, length=None, graph=None, dependencies=None, **kwargs):
        """
        Reset operation constructor.

        Args:
            length:
            graph:
            dependencies:
            **kwargs:       not used

        Returns:
            instance of ResetOperation
        """

        return ResetOperation(
            kernel=self.kernel,
            name=self.name + '_reset_op',
            length=length,
            graph=graph,
            dependencies=dependencies,
        )

    def update_state(self, length=None, graph=None, dependencies=None, **kwargs):
        """
        UpdateStateOperation pf operation constructor.

        Args:
            length:
            graph:
            dependencies:
            **kwargs:       not used

        Returns:
            instance of UpdateStateOperation constructor
        """

        return UpdateStateOperation(
            kernel=self.kernel,
            name=self.name + '_update_state_op',
            length=length,
            graph=graph,
            dependencies=dependencies,
        )


class PortfolioManagerNode(Node):

    def __init__(self, name='PortfolioManager', **kwargs):
        super().__init__(
            kernel_class_ref=BasePortfolioManager,
            name=name,
            **kwargs
        )

