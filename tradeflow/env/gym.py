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

    def _evaluate_graph(self, feed_dict):
        fetches = self.graph(
            [self.output['observation'], self.output['reward'], self.output['done']], feed_dict
        )
        return fetches

    def reset(self):
        # Redundant: need to run entire graph to properly reset states.
        # todo: maybe implement specific op to reset entire graph state?
        feed_dict = {
                self.input['reset']: True,
                self.input['action']: 0,
                self.input['dataset']: self.dataset,
                self.input['episode_duration']: self.episode_duration,
            }
        observation, reward, done = self._evaluate_graph(feed_dict)
        return observation

    def step(self, action):
        feed_dict = {
                self.input['reset']: False,
                self.input['action']: action,
                self.input['dataset']: None,
                self.input['episode_duration']: None,
            }
        observation, reward, done = self._evaluate_graph(feed_dict)
        return observation, reward, done, dict()


class EnvironmentConstructor(object):
    """
    Service class: builds mdp dataflow graph and wraps it with environment API
    """
    # TODO: refract: pack all init args to env_config kwarg of __call__ method

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

    def __call__(self, env_config):
        """
        Instantiates environment object.

        Args:
            env_config:    env hyperparameters dict

        Returns:
            instance of env_class_ref
        """
        nodes = self._build_nodes(self.nodes_config)
        graph, graph_input, graph_output = self._build_graph(nodes)
        action_space = nodes['order'].kernel.space
        observation_space = nodes['observation'].kernel.space
        env = self.env_class_ref(
            graph=graph,
            graph_input=graph_input,
            graph_output=graph_output,
            action_space=action_space,
            observation_space=observation_space,
            **env_config
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
