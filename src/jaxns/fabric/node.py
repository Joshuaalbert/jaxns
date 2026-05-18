from __future__ import annotations

import contextlib
import importlib
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

from jaxns.fabric.process_manager import (
    ProcessManager,
    create_random_ack_address,
    create_random_control_address,
)
from jaxns.fabric.zmq_p2p import LoadBalancer, RPCActor, RPCClientMeta, ZMQRPCClient, get_free_port
from jaxns.model import Model


class AbstractNode(ABC):

    @abstractmethod
    def evaluate(self, u):
        """
        Evaluates the registered likelihood at the given point `u`.

        Args:
            u: a pytree primal matching the input to the model likelihood evaluation.

        Returns:
            The log-likelihood at `u`.
        """
        ...


class NodeImpl(AbstractNode):
    def __init__(self, evaluate_fn: Callable[[Any], Any]):
        self._evaluate_fn = evaluate_fn

    def evaluate(self, u):
        return self._evaluate_fn(u)


class ModelNode(AbstractNode):
    def __init__(self, model: Model, args: tuple = (), params=None):
        self._model = model
        self._args = args
        self._params = params

    def evaluate(self, u):
        return self._model.log_likelihood(u, args=self._args, params=self._params, allow_nan=False)


@dataclass(frozen=True, slots=True)
class FabricAddresses:
    ctl_pub_addr: str
    ack_rep_addr: str
    frontend_addr: str
    backend_addr: str


def create_local_fabric_addresses() -> FabricAddresses:
    return FabricAddresses(
        ctl_pub_addr=create_random_control_address(),
        ack_rep_addr=create_random_ack_address(),
        frontend_addr=f"tcp://127.0.0.1:{get_free_port()}",
        backend_addr=f"tcp://127.0.0.1:{get_free_port()}",
    )


def _identity_evaluate(u):
    return u


def _coerce_service(service_or_callable: Any) -> AbstractNode:
    evaluate = getattr(service_or_callable, "evaluate", None)
    if callable(evaluate):
        return service_or_callable
    if callable(service_or_callable):
        return NodeImpl(service_or_callable)
    raise TypeError(
        "Expected a node service with an evaluate(u) method, or a callable returning a scalar log-likelihood.",
    )


def load_object(import_path: str) -> Any:
    if ":" not in import_path:
        raise ValueError(f"Expected import path in 'module:attribute' form, got {import_path!r}.")
    module_name, attr_name = import_path.split(":", 1)
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr_name)
    except AttributeError as e:
        raise AttributeError(f"Module {module_name!r} has no attribute {attr_name!r}.") from e


def load_service_factory(service_factory: str | Callable[[], Any]) -> Callable[[], AbstractNode]:
    if isinstance(service_factory, str):
        service_factory = load_object(service_factory)
    if not callable(service_factory):
        raise TypeError("service_factory must be a callable or an import string.")

    def create_service() -> AbstractNode:
        return _coerce_service(service_factory())

    return create_service


class NodeActor(RPCActor):

    def __init__(
        self,
        ctl_pub_addr: str,
        ack_rep_addr: str,
        backend_addr: str,
        *,
        service_factory: str | Callable[[], Any] | None = None,
        evaluate_fn: Callable[[Any], Any] | None = None,
        **kwargs,
    ):
        super().__init__(
            ctl_pub_addr=ctl_pub_addr,
            ack_rep_addr=ack_rep_addr,
            backend_addr=backend_addr,
            **kwargs,
        )
        if service_factory is not None and evaluate_fn is not None:
            raise ValueError("Provide only one of service_factory or evaluate_fn.")
        self._service_factory = service_factory
        self._evaluate_fn = evaluate_fn

    @contextlib.contextmanager
    def yield_service(self):
        if self._service_factory is not None:
            service = load_service_factory(self._service_factory)()
        else:
            service = NodeImpl(self._evaluate_fn or _identity_evaluate)
        service = _coerce_service(service)
        try:
            yield service
        finally:
            pass


class NodeClient(ZMQRPCClient, AbstractNode, metaclass=RPCClientMeta):
    ...


class RemoteNodeEvaluator(AbstractNode):
    """
    Thread-aware node evaluator that lazily creates a client per thread.
    """

    def __init__(
        self,
        frontend_addr: str,
        *,
        ident_prefix: str = "node-evaluator",
        assign_timeout_ms: int | None = None,
        retries: int = 0,
    ):
        self.frontend_addr = frontend_addr
        self.ident_prefix = ident_prefix
        self.assign_timeout_ms = assign_timeout_ms
        self.retries = retries
        self._thread_local = threading.local()
        self._lock = threading.Lock()
        self._clients: dict[int, NodeClient] = {}
        self._closed = False

    def _get_client(self) -> NodeClient:
        if self._closed:
            raise RuntimeError("RemoteNodeEvaluator is closed.")
        client = getattr(self._thread_local, "client", None)
        if client is None:
            thread_id = threading.get_ident()
            with self._lock:
                stale_client = self._clients.pop(thread_id, None)
            if stale_client is not None:
                stale_client.__exit__(None, None, None)
            client = NodeClient(
                ident=f"{self.ident_prefix}-{thread_id}",
                frontend_addr=self.frontend_addr,
                assign_timeout_ms=self.assign_timeout_ms,
                retries=self.retries,
            )
            client.__enter__()
            self._thread_local.client = client
            with self._lock:
                self._clients[thread_id] = client
        return client

    def evaluate(self, u):
        return self._get_client().evaluate(u)

    def close(self):
        with self._lock:
            clients = list(self._clients.values())
            self._clients.clear()
        for client in clients:
            client.__exit__(None, None, None)
        self._thread_local.client = None
        self._closed = True

    def __enter__(self):
        self._closed = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def build_scheduler_process_manager(
    addresses: FabricAddresses | Any,
    *,
    profile: bool = False,
    shutdown_timeout: float = 1.0,
    start_method: str = "forkserver",
) -> ProcessManager:
    load_balancer = LoadBalancer(
        ctl_pub_addr=addresses.ctl_pub_addr,
        ack_rep_addr=addresses.ack_rep_addr,
        frontend_addr=addresses.frontend_addr,
        backend_addr=addresses.backend_addr,
    )
    return ProcessManager(
        [load_balancer],
        ctl_pub_addr=addresses.ctl_pub_addr,
        ack_rep_addr=addresses.ack_rep_addr,
        shutdown_timeout=shutdown_timeout,
        profile=profile,
        start_method=start_method,
    )


def build_worker_process_manager(
    addresses: FabricAddresses | Any,
    *,
    service_factory: str | Callable[[], Any] | None = None,
    evaluate_fn: Callable[[Any], Any] | None = None,
    num_workers: int = 1,
    profile: bool = False,
    shutdown_timeout: float = 1.0,
    start_method: str = "forkserver",
    **actor_kwargs,
) -> ProcessManager:
    if num_workers < 1:
        raise ValueError(f"num_workers must be >= 1, got {num_workers}.")
    actors = [
        NodeActor(
            ctl_pub_addr=addresses.ctl_pub_addr,
            ack_rep_addr=addresses.ack_rep_addr,
            backend_addr=addresses.backend_addr,
            service_factory=service_factory,
            evaluate_fn=evaluate_fn,
            **actor_kwargs,
        )
        for _ in range(num_workers)
    ]
    return ProcessManager(
        actors,
        ctl_pub_addr=addresses.ctl_pub_addr,
        ack_rep_addr=addresses.ack_rep_addr,
        shutdown_timeout=shutdown_timeout,
        profile=profile,
        start_method=start_method,
    )


@contextlib.contextmanager
def local_node_evaluator(
    *,
    service_factory: str | Callable[[], Any] | None = None,
    evaluate_fn: Callable[[Any], Any] | None = None,
    num_workers: int = 1,
    ident_prefix: str = "local-node",
    assign_timeout_ms: int | None = None,
    retries: int = 0,
    profile: bool = False,
    start_method: str = "forkserver",
    **actor_kwargs,
):
    addresses = create_local_fabric_addresses()
    scheduler_mgr = build_scheduler_process_manager(
        addresses=addresses,
        profile=profile,
        start_method=start_method,
    )
    worker_mgr = build_worker_process_manager(
        addresses=addresses,
        service_factory=service_factory,
        evaluate_fn=evaluate_fn,
        num_workers=num_workers,
        profile=profile,
        start_method=start_method,
        **actor_kwargs,
    )
    scheduler_mgr.start_all()
    try:
        try:
            worker_mgr.start_all()
            with RemoteNodeEvaluator(
                frontend_addr=addresses.frontend_addr,
                ident_prefix=ident_prefix,
                assign_timeout_ms=assign_timeout_ms,
                retries=retries,
            ) as evaluator:
                yield evaluator
        finally:
            worker_mgr.stop_all()
            worker_mgr.print_tracebacks()
    finally:
        scheduler_mgr.stop_all()
        scheduler_mgr.print_tracebacks()


def make_model_service_factory(model: Model, args: tuple = (), params=None) -> Callable[[], ModelNode]:
    return partial(ModelNode, model=model, args=args, params=params)
