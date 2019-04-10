from logbook import INFO
from collections import namedtuple

from btgym.spaces import ActionDictSpace

from ..core import Kernel

MarketOrder = namedtuple('MarketOrder', ['asset', 'type'])


class ActionToMarketOrder(Kernel):
    """
    Maps abstract MDP actions to executable Market Orders.
    """

    def __init__(
            self,
            assets,
            name='ActionMap',
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

    def update_state(self, reset, action):
        if reset:
            self._start(action)

        else:
            self._update_state(action)

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