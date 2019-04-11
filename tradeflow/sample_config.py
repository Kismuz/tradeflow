from logbook import INFO, DEBUG
import pythonflow as pf
import pandas as pd


from tradeflow import KernelDevice, PandasStateConfig
from tradeflow.nodes import PandasMarketEpisode, PandasMarketStep, PortfolioManager
from tradeflow.nodes import ActionToOrder, Done, TradeReward, ToDictSpace

from tradeflow import Environment as Env
from tradeflow.env.gym import EnvironmentConstructor


df = pd.read_csv('./data/dfk4/insample.csv', float_precision='high', skiprows=0, nrows=None)


# Maybe Put dataset to ray object store:
# df = ray.put(df)


LOG_LEVEL = INFO

FEATURES_DEPTH = 8

FEATURES_COLUMNS =  [
    'lvolume', 'netVol', 'imb50', 'sSpread', 'lrgSpread', 'ALMA', 'brange',
    'asize', 'bsize', 'a_cnt', 'b_cnt', 'R1', 'R1_250', 'R1_500', 'R1_1000',
    'imb_of', 'netALMA', 'cumVolALMA',
]


# Nodes_config sets entire environment configuration
# except computation topology and runtime inputs such as dataset and episode duration:

nodes_config = dict(
    episode=dict(
        class_ref=PandasMarketEpisode,
        device=KernelDevice.LOCAL,
        log_level=LOG_LEVEL
    ),
    market=dict(
        class_ref=PandasMarketStep,
        state_config=dict(
            features=PandasStateConfig(
                columns=FEATURES_COLUMNS,
                depth=FEATURES_DEPTH,
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
    observation=dict(
        class_ref=ToDictSpace,
        space_config = {
            'market_features': (FEATURES_DEPTH, len(FEATURES_COLUMNS)),
            'value': (1,),
            'reward': (1,),
        },
        device=KernelDevice.LOCAL,
        log_level=LOG_LEVEL,
    ),
)


def make_simple_graph(node):
    """
    Defines environment logic by building pf.Graph with provided nodes.

    Args:
        node:     dictionary of Node instances

    Returns:
        pf.Graph instance, dictionaries of input of and output handles.s
    """
    with pf.Graph() as graph:

        # Input placeholders to graph:
        is_reset = pf.placeholder(name='reset_input_flag')
        episode_duration = pf.placeholder(name='episode_duration_input')
        dataset = pf.placeholder(name='entire_dataset_input')
        action = pf.placeholder(name='incoming_mdp_action')

        # Connect nodes, define runtime logic:
        episode = node['episode'](input_state=dataset, reset=is_reset, sample_length=episode_duration)

        market_state = node['market'](input_state=episode, reset=is_reset)

        orders = node['order'](input_state=action, reset=is_reset)

        portfolio_state = node['manager'](input_state=market_state, reset=is_reset, orders=orders)

        reward = node['reward'](input_state=portfolio_state, reset=is_reset)

        done = node['done'](input_state=market_state)

        # Filter out all irrelevant fields to get observation tensor(s):
        observation_state = {
            'market_features': market_state['features'],
            'value': portfolio_state['portfolio_value'],
            'reward': reward,
        }
        # Service node, processes dictionaries, dataframes
        # and emits numpy arrays ready to feed to estimator:
        observation_state = node['observation'](input_state=observation_state)

    # Group graph input and output handles:
    graph_input = dict(
        reset=is_reset,
        dataset=dataset,
        episode_duration=episode_duration,
        action=action
    )
    graph_output = dict(
        observation=observation_state,
        reward=reward,
        done=done,
    )
    return graph, graph_input, graph_output


# Additional paramters that can be changed in environment runtime:
env_params = dict(
    dataset=df,
    episode_duration=20,
)

# Instantiate envoronment, typically should be done
# inside remote worker task via ray registration mechanism:
env = EnvironmentConstructor(Env, nodes_config, build_graph_fn=make_simple_graph)(env_params)

# Now run:
o = env.reset()

o, r, d, i = env.step(env.action_space.sample())
