from collections import UserDict

from logbook import StreamHandler, Logger, NullHandler
import sys


class Kernel(object):
    """
    Base Node computation container. Estimates Node state.
    Subclass for user-level kernel implementation.
    """

    def __init__(self, name='BaseKernel', log=None, **kwargs):
        self.name = name
        self.log = log

    def initialize(self, **inputs):
        pass

    def compute(self, **inputs):
        return inputs


class Node(UserDict):
    """
    Core building block for modelling markov decision process graph.
    Subclass for user-level nodes implementation.

    Methods:
        initialize:
        run:

    Attributes:
        kernel:     instance of Kernel subclass supporting `compute` and `initialize` methods
        state:      any object representing mutable persistent Node output
    """

    def __init__(self, name='Node', log=None, **inputs):
        """

        Args:
            name:       str, node identification name
            inputs:     keyword arguments holding arbitrary Python objects or instances of other Nodes,
                        inputs spec should be defined for every Node subclass and basically depends on Kernel inputs.
        """
        super().__init__(inputs)
        self.name = name
        if log is None:
            NullHandler().push_application()
            self.log = Logger()

        else:
            self.log = log

        self.kernel = Kernel()
        self.state = None
        self._initialized = False
        self._locked = None

    def initialize(self, feed_dict=None, log=None):
        if log is not Node:
            self.push_logger(log)

        self._unlock()
        self._reset_state(feed_dict)

    def push_logger(self, log):
        StreamHandler(sys.stdout).push_application()
        self.log = log
        self.kernel.log = self.log
        for key, sub_node in self.items():
            sub_node.push_logger(self.log)

    def run(self, feed_dict=None):
        self._unlock()
        fetches = self._set_state(feed_dict)
        return fetches

    def _unlock(self):
        for key, sub_node in self.items():
            sub_node._unlock()
        self._locked = False

    def _reset_state(self, feed_dict):
        if not self._locked:
            self.log.debug('{name}._reset is unlocked with feed_dict={}'.format(feed_dict, name=self.name))
            feeds = {}
            for key, sub_node in self.items():
                try:
                    feeds[key] = sub_node._reset_state(feed_dict[key])

                except (TypeError, KeyError) as e:
                    feeds[key] = sub_node._reset_state(None)

            self.kernel.initialize(**feeds)
            self._locked = True

        else:
            self.log.debug('{name}._reset is locked with feed_dict={}'.format(feed_dict, name=self.name))

        self._initialized = True

    def _set_state(self, feed_dict):
        try:
            assert self._initialized

        except AssertionError:
            e = 'Attempt to call uninitialized node: {name}.'.format(name=self.name)
            self.log.error(e)
            raise RuntimeError(e)

        if not self._locked:
            self.log.debug('{name}._set_state is unlocked with feed_dict={}'.format(feed_dict, name=self.name))
            feeds = {}
            for key, sub_node in self.items():
                try:
                    feeds[key] = sub_node._set_state(feed_dict[key])

                except (TypeError, KeyError) as e:
                    feeds[key] = sub_node._set_state(None)

            self.log.debug('{name}._set_state has gathered feeds: {}'.format(feeds, name=self.name))
            self.state = self.kernel.compute(**feeds)
            self._locked = True

        else:
            self.log.debug('{name}._set_state is locked with feed_dict={}'.format(feed_dict, name=self.name))

        return self.state


class Input(Node):
    def __init__(self, name='Input', **kwargs):
        super().__init__(name=name, **kwargs)

    def _reset_state(self, feed_dict=None):
        if not self._locked:
            self.log.debug('{name}._reset is unlocked with feed_dict={}'.format(feed_dict, name=self.name))
            self.state = feed_dict
            self._locked = True

        else:
            self.log.debug('{name}._reset is locked with feed_dict={}'.format(feed_dict, name=self.name))

        return self.state

    def _set_state(self, feed_dict):
        if not self._locked:
            self.log.debug('{name}._set_state is unlocked with feed_dict={}'.format(feed_dict, name=self.name))
            self.state = feed_dict
            self._locked = True

        else:
            self.log.debug('{name}._set_state is locked with feed_dict={}'.format(feed_dict, name=self.name))

        return self.state


class Replica(Node):
    def __init__(self, node, name='Replica', **kwargs):
        super().__init__(name=node.name + '/' + name, **kwargs)
        self.node = node
        self.state = self.node.state

    def _reset_state(self, feed_dict=None):
        self.state = self.node.state
        return self.state

    def _set_state(self, feed_dict=None):
        self.state = self.node.state
        return self.state
