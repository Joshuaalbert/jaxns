import contextlib
from abc import ABC, abstractmethod

from jaxns.distributed.zmq_rpc import RPCActor, ZMQRPCClient, RPCClientMeta


class AbstractNode(ABC):

    @abstractmethod
    def evaluate(self, u):
        """
        Evaluates the registered likelihood at the given point `u`.

        Args:
            u: a pytree primal matching the input to the model likelihood evaluation.

        Returns:
            The log-likelihood at `u`
        """
        ...


class NodeImpl(AbstractNode):

    def evaluate(self, u):
        return u


class NodeActor(RPCActor):

    # TODO: create model get endpoint in coordinator
    # Load balancer only just for dispatching requests. Need workers to load the model and inputs when they start up.
    # The load balancer doesn't know about models.

    @contextlib.contextmanager
    def yield_service(self):
        service = NodeImpl()
        try:
            yield service
        finally:
            pass


class NodeClient(ZMQRPCClient, AbstractNode, metaclass=RPCClientMeta):
    ...
