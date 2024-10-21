import warnings
from functools import wraps
from typing import Callable, Tuple, Dict, NamedTuple, Any, Optional, TypeVar

import jax
import jax.numpy as jnp
from jax._src.typing import SupportsDType

__all__ = [
    'get_parameter',
    'get_state',
    'set_state',
    'transform_with_state',
    'transform',
    'convert_external_params',
    'wrap_random',
    'next_rng_key'
]

MutableParams = Dict[str, jax.Array]
ImmutableParams = Dict[str, jax.Array]


class Ctx:
    def __init__(self, rng, stack):
        self.params = {}
        self.states = {}
        self._rng = rng
        self._stack = stack

    def next_rng_key(self):
        self._rng, new_rng = jax.random.split(self._rng)
        return new_rng

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stack.remove(self)
        return


class GlobalContext:
    def __init__(self, rng: Optional[jax.Array] = None):
        self.stack = []

    def new(self, rng):
        new_ctx = Ctx(rng=rng, stack=self.stack)
        self.stack.append(new_ctx)
        return new_ctx

    @property
    def params(self):
        if len(self.stack) == 0:
            raise ValueError("No context available. Must use transform_with_state to create a context.")
        return self.stack[-1].params

    @property
    def states(self):
        if len(self.stack) == 0:
            raise ValueError("No context available. Must use transform_with_state to create a context.")
        return self.stack[-1].states

    def next_rng_key(self):
        if len(self.stack) == 0:
            raise ValueError("No context available. Must use transform_with_state to create a context.")
        return self.stack[-1].next_rng_key()


global_context = GlobalContext()

PT = TypeVar('PT')
InitType = Callable[[Tuple[int, ...], SupportsDType], PT] | Callable[[], PT]


def default_init(shape: Tuple[int, ...], dtype: SupportsDType):
    raise NotImplementedError("No init provided.")


def get_parameter(name: str, shape: Optional[Tuple[int, ...]] = None, dtype: Optional[SupportsDType] = jnp.float32, *,
                  init: InitType = default_init) -> PT:
    """
    Get a parameter variable.

    Args:
        name: the name of the parameter
        shape: the shape of the parameter must be provided if init is not a jax.Array
        dtype: the dtype of the parameter must be provided if init is not a jax.Array
        init: the initializer

    Returns:
        The parameter variable as a jax.Array=
    """
    if name not in global_context.params:
        if callable(init):

            if (shape is None) and (dtype is None):
                global_context.params[name] = init()
            else:
                global_context.params[name] = init(shape, dtype)
        else:
            warnings.warn(
                "Using a constant initializer for state. This is not recommended as it may induce closure issues.")
            global_context.params[name] = init

    return global_context.params[name]


ExtParam = TypeVar('ExtParam')


def convert_external_params(external_params: ExtParam, prefix: str) -> ExtParam:
    """
    Convert external parameters to context parameters. This can be used to convert haiku or flax parameters to
    jaxns parameters for using in models.

    Args:
        external_params: map of name -> value

    Returns:
        The context parameters
    """
    leaf_list, tree_def = jax.tree.flatten(external_params)

    def _unique_name(idx):
        return f"__{prefix}_{idx}"

    ctx_params = [get_parameter(_unique_name(idx), init=leaf) for idx, leaf in enumerate(leaf_list)]
    external_params_ctx = jax.tree_unflatten(tree_def, ctx_params)
    return external_params_ctx


def wrap_random(f):
    """
    Wrap a function to use a random number generator from the context.

    Args:
        f: the function to wrap

    Returns:
        The wrapped function
    """

    @wraps(f)
    def wrapped(*args, **kwargs):
        rng = next_rng_key()
        return f(rng, *args, **kwargs)

    return wrapped


def get_state(name: str, shape: Optional[Tuple[int, ...]] = None, dtype: Optional[SupportsDType] = None, *,
              init: InitType = default_init) -> PT:
    """
    Get a state variable.

    Args:
        name: the name of the state
        shape: the shape of the state must be provided if init is not a jax.Array
        dtype: the dtype of the state must be provided if init is not a jax.Array
        init: the initializer

    Returns:
        The state variable as a jax.Array
    """
    if name not in global_context.states:
        if callable(init):
            if (shape is None) and (dtype is None):
                global_context.states[name] = init()
            else:
                global_context.states[name] = init(shape, dtype)
        else:
            warnings.warn(
                "Using a constant initializer for state. This is not recommended as it may induce closure issues.")
            global_context.states[name] = init
    return global_context.states[name]


def set_state(name: str, value: PT):
    """
    Set a state variable.

    Args:
        name: the name of the state
        value: the value to set

    Returns:
        The state variable as a jax.Array
    """
    if name not in global_context.states:
        raise ValueError(f"State {name} not found.")
    global_context.states[name] = value


class InitWithStateReturn(NamedTuple):
    fn_val: Any
    params: MutableParams
    states: ImmutableParams


class ApplyWithStateReturn(NamedTuple):
    fn_val: Any
    states: ImmutableParams


class TransformedWithStateFn(NamedTuple):
    init: Callable[[jax.Array, ...], InitWithStateReturn]
    apply: Callable[[MutableParams, ImmutableParams, jax.Array, ...], ApplyWithStateReturn]


def transform_with_state(f: Callable) -> TransformedWithStateFn:
    """
    Transform a function to use parameters and states.

    Args:
        f: the function to transform

    Returns:
        A tuple of the init and apply functions
    """

    @wraps(f)
    def init(rng: jax.Array, *args, **kwargs) -> InitWithStateReturn:
        """
        Get initial parameters and states.

        Args:
            rng: the PRNGkey
            *args: args to the function
            **kwargs: kwargs to the function

        Returns:
            The output of the function at the given input, the parameters and the states
        """
        with global_context.new(rng) as ctx:
            return InitWithStateReturn(fn_val=f(*args, **kwargs), params=global_context.params,
                                       states=global_context.states)

    @wraps(f)
    def apply(params: MutableParams, states: ImmutableParams, rng: jax.Array, *args, **kwargs) -> ApplyWithStateReturn:
        """
        Apply the function with given parameters and states.

        Args:
            params: the parameters
            states: the states
            rng: the PRNGkey to apply at
            *args: args to function
            **kwargs: kwargs to function

        Returns:
            The output of the function at the given input and the states
        """
        with global_context.new(rng) as ctx:
            ctx.params = params
            ctx.states = states
            return ApplyWithStateReturn(fn_val=f(*args, **kwargs), states=ctx.states)

    return TransformedWithStateFn(init=init, apply=apply)


class InitReturn(NamedTuple):
    fn_val: Any
    params: MutableParams


class ApplyReturn(NamedTuple):
    fn_val: Any


class TransformedFn(NamedTuple):
    init: Callable[[jax.Array, ...], InitReturn]
    apply: Callable[[MutableParams, jax.Array, ...], ApplyReturn]


def transform(f: Callable) -> TransformedFn:
    """
    Transform a function to use parameters and states.

    Args:
        f: the function to transform

    Returns:
        A tuple of the init and apply functions
    """

    @wraps(f)
    def init(rng: jax.Array, *args, **kwargs) -> InitReturn:
        """
        Get initial parameters and states.

        Args:
            rng: the PRNGkey
            *args: args to the function
            **kwargs: kwargs to the function

        Returns:
            The output of the function at the given input, the parameters and the states
        """
        with global_context.new(rng) as ctx:
            return InitReturn(fn_val=f(*args, **kwargs), params=global_context.params)

    @wraps(f)
    def apply(params: MutableParams, rng: jax.Array, *args, **kwargs) -> ApplyReturn:
        """
        Apply the function with given parameters and states.

        Args:
            params: the parameters
            states: the states
            rng: the PRNGkey to apply at
            *args: args to function
            **kwargs: kwargs to function

        Returns:
            The output of the function at the given input and the states
        """
        with global_context.new(rng) as ctx:
            ctx.params = params
            return ApplyReturn(fn_val=f(*args, **kwargs))

    return TransformedFn(init=init, apply=apply)


def next_rng_key():
    """
    Get the next random number generator

    Returns:
        The next random number generator
    """
    return global_context.next_rng_key()
