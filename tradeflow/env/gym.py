import gym
import copy


class Environment(gym.Env):
    """
    Environment is basically a wrapper around pf.Graph with standard API functionality.
    """
    def __init__(
            self,
            graph,
            graph_input,
            graph_output,
            dataset,
            episode_duration,
            action_space,
            observation_space,
            name='Environment'
    ):
        # super().__init__()
        self.name = name
        self.action_space = action_space
        self.observation_space = observation_space

        self.graph = graph

        # Handles:
        self.input = graph_input
        self.output = graph_output

        # Parameters:
        self.dataset = dataset
        self.episode_duration = episode_duration

    def reset(self):
        obs_market, obs_portfolio = self.graph(
            [self.output['market_state'], self.output['portfolio_state']],
            {
                self.input['reset']: True,
                self.input['action']: None,
                self.input['dataset']: self.dataset,
                self.input['episode_duration']: self.episode_duration,
            }
        )
        return obs_market, obs_portfolio

    def step(self, action):
        obs_market, obs_portfolio, reward, done = self.graph(
            [self.output['market_state'], self.output['portfolio_state'], self.output['reward'], self.output['done']],
            {
                self.input['reset']: False,
                self.input['action']: action,
                self.input['dataset']: None,
                self.input['episode_duration']: None,
            }
        )
        return (obs_market, obs_portfolio), reward, '', done


def make_environment(env_config, env_class_ref=Environment):
    return env_class_ref(**env_config)


class EnvironmentConstructor(object):
    """
    Service class: builds mdp dataflow graph and wraps it with environment API
    """

    def __init__(self, env_class_ref, nodes_config=None, build_graph_fn=None):
        """

        Args:
            env_class_ref:      environment wrapper class
            nodes_config:       nodes configuration dict
            build_graph_fn:     callable returning pf.Graph instance,
                                dictionary of graph input handles, dictionary of graph output handles;
                                if provided, overrides bound method _build_graph
        """
        self.env_class_ref = env_class_ref
        self.nodes_config = nodes_config

        if build_graph_fn is not None:
            self._build_graph = build_graph_fn

    def __call__(self, env_parameters):
        """
        Instantiates environment object.

        Args:
            env_parameters:    env hyperparameters dict

        Returns:
            instance of env_class_ref
        """
        nodes = self._build_nodes(self.nodes_config)
        graph, graph_input, graph_output = self._build_graph(nodes)
        action_space = nodes['order'].kernel.space
        observation_space = None
        env = self.env_class_ref(
            graph=graph,
            graph_input=graph_input,
            graph_output=graph_output,
            action_space=action_space,
            observation_space=observation_space,
            **env_parameters
        )
        return env

    @staticmethod
    def _build_nodes(nodes_config):
        nodes = {}
        for name, config in nodes_config.items():
            node_config = copy.deepcopy(config)
            node_class = node_config.pop('class_ref')
            nodes[name] = node_class(**node_config)

        return nodes

    @staticmethod
    def _build_graph(nodes):
        """
        Defines graph topology with nodes provided.
        This method should be overridden to implement actual dataflow logic.

        Returns:
            pf.Graph instance, dictionary of graph input handles, dictionary of graph output handles
        """
        return None, None, None
