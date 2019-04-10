from logbook import INFO, DEBUG

from tradeflow.core import KernelDevice
from tradeflow.kernel.iterator import PandasStateConfig

from tradeflow.nodes import PandasMarketEpisode, PandasMarketStep, PortfolioManager, ActionToOrder, Done, TradeReward


LOG_LEVEL = INFO

basic_nodes_config = dict(
    episode=dict(
        class_ref=PandasMarketEpisode,
        device=KernelDevice.LOCAL,
        log_level=LOG_LEVEL
    ),
    market=dict(
        class_ref=PandasMarketStep,
        state_config=dict(
            features=PandasStateConfig(
                columns= [
                    'lvolume', 'netVol', 'imb50', 'sSpread', 'lrgSpread', 'ALMA', 'brange',
                    'asize', 'bsize', 'a_cnt', 'b_cnt', 'R1', 'R1_250', 'R1_500', 'R1_1000',
                    'imb_of', 'netALMA', 'cumVolALMA',
                ],
                depth=8,
            ),
            P_VWAP=PandasStateConfig(
                columns=['P_VWAP'],
                depth=1,
            ),
        ),
        device=KernelDevice.LOCAL,
        log_level=LOG_LEVEL,
    ),
    order=dict(
        class_ref=ActionToOrder,
        assets=['P_VWAP'],
        device=KernelDevice.LOCAL,
        log_level=LOG_LEVEL
    ),
    manager=dict(
        class_ref=PortfolioManager,
        max_position_size=3,
        order_size=1,
        order_commission=0.0,
        orders=('buy', 'sell', 'close'),
        assets=['P_VWAP'],
        device=KernelDevice.LOCAL,
        log_level=LOG_LEVEL,
    ),
    reward=dict(
        class_ref=TradeReward,
        scale=1.0,
        device=KernelDevice.LOCAL,
        log_level=LOG_LEVEL,
    ),
    done=dict(
        class_ref=Done,
        device=KernelDevice.LOCAL,
        log_level=LOG_LEVEL,
    ),
)