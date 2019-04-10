from logbook import INFO
import numpy as np

from ..core import Kernel


class ClosedTradeRewardFn(Kernel):
    """
    Simple single asset reward function.
    """
    def __init__(
            self,
            unrealized_pnl_weight=1.0,
            realized_pnl_weight=10.0,
            scale=1.0,
            clip=100,
            name='ClosedTradeReward',
            task=0,
            log=None,
            log_level=INFO,
    ):
        super().__init__(name=name, task=task, log=log, log_level=log_level)
        self.unrealized_pnl_weight = unrealized_pnl_weight
        self.realized_pnl_weight = realized_pnl_weight
        self.scale = scale
        self.clip = abs(clip)

    def update_state(self, reset, input_state):
        if reset:
            self.state = 0.0

        else:
            self._update_state(input_state)

        return self.state

    def _update_state(self, portfolio_state):
        try:
            u_ret = portfolio_state['unrealized_return']
            r_ret = portfolio_state['realized_return']

        except KeyError:
            e = 'Expected keys `unrealized_return` and `realized_return` not found in portfolio state'
            self.log.error(e)
            raise ValueError(e)

        self.log.debug('u_ret: {}, r_ret: {}'.format(u_ret, r_ret))

        mean_unr_returns = np.mean(np.asarray(u_ret))
        mean_real_returns = np.nanmean(np.asarray(r_ret))
        if np.isnan(mean_real_returns):
            mean_real_returns = 0.0

        self.state = np.clip(
            self.scale * (
                mean_unr_returns * self.unrealized_pnl_weight +
                mean_real_returns * self.realized_pnl_weight
            ),
            -self.clip,
            self.clip
        )

