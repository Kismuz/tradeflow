from logbook import INFO
from collections import namedtuple

from btgym.spaces import ActionDictSpace
from gym.spaces import Discrete

from ..core import Kernel

MarketOrder = namedtuple('MarketOrder', ['asset', 'type'])


class DiscreteActionToMarketOrder(Kernel):
    """
    Maps gym.spaces.Discrete actions to executable Market Orders.
    """
    # TODO: speed up by turning off action validation

    def __init__(
            self,
            assets,
            name='AssetActionMap',
            task=0,
            log=None,
            log_level=INFO,
    ):
        super().__init__(name=name, task=task, log=log, log_level=log_level)
        assets = list(assets)
        try:
            assert len(assets) == 1

        except AssertionError:
            e = 'This class currently supports single asset trading only'
            self.log.error(e)
            raise ValueError(e)

        self.assets = assets

        self.space = Discrete(4)
        self.action_map = {0: None, 1: 'buy', 2: 'sell', 3: 'close'}

    def update_state(self, input_state, reset):
        if reset:
            self._start(input_state)

        else:
            self._update_state(input_state)

        return self.state

    def _start(self, action):
        self.state = []

    def _update_state(self, action):
        try:
            assert self.space.contains(action)

        except AssertionError:
            e = 'Provided action `{}` is not a valid member of defined action space `{}`'.format(action, self.space)
            self.log.error(e)
            raise TypeError(e)

        if action != 0:
            self.state = [MarketOrder(self.assets[0], self.action_map[action])]

        else:
            self.state = []


class AssetActionToMarketOrder(Kernel):
    """
    Maps btgym.spaces.ActionDictSpace actions to executable Market Orders.
    """
    def __init__(
            self,
            assets,
            name='AssetActionMap',
            task=0,
            log=None,
            log_level=INFO,
    ):
        assets = list(assets)
        super().__init__(name=name, task=task, log=log, log_level=log_level)
        self.space = ActionDictSpace(
            base_actions=[0, 1, 2, 3],
            assets=assets
        )
        self.action_map = {0: None, 1: 'buy', 2: 'sell', 3: 'close'}

    def update_state(self, input_state, reset):
        if reset:
            self._start(input_state)

        else:
            self._update_state(input_state)

        return self.state

    def _start(self, action):
        self.state = []

    def _update_state(self, action):
        try:
            assert self.space.contains(action)

        except AssertionError:
            e = 'Provided action `{}` is not a valid member of defined action space`'.format(action)
            self.log.error(e)
            raise TypeError(e)

        self.state = [MarketOrder(asset, self.action_map[value]) for asset, value in action.items() if value != 0]