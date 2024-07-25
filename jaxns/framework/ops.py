import inspect
import warnings
from typing import Tuple, Callable, Generator

import jax
from jax import numpy as jnp, lax

from jaxns.framework.bases import PriorModelType, BaseAbstractPrior, PriorModelGen
from jaxns.framework.prior import InvalidPriorName, SingularPrior, Prior
from jaxns.internals.maps import pytree_unravel
from jaxns.internals.types import UType, XType, float_type, LikelihoodInputType, FloatArray, LikelihoodType, PRNGKey, \
    isinstance_namedtuple, WType, RandomVariableType

__all__ = [
    'simulate_prior_model'
]


def _get_prior_model_gen(prior_model: PriorModelType) -> PriorModelGen:
    gen = prior_model()
    # Check if gen is a generator
    if not inspect.isgenerator(gen):
        warnings.warn(
            "The provided prior_model is not a generator, this may mean you forget `yield` statements. "
            "This means there are no Bayesian variables."
        )

        def dummy_prior_model(output):
            _ = yield Prior(0.)
            return output

        # Make an empty generator that returns the output.
        gen = dummy_prior_model(gen)
    return gen


def compute_U_ndims(prior_model: PriorModelType) -> int:
    """
    Computes placeholders of model.

    Args:
        prior_model: a callable that produces a prior model generator

    Returns:
        number of U dims
    """
    U_ndims = 0
    gen = _get_prior_model_gen(prior_model=prior_model)

    prior_response = None
    names = set()
    while True:
        try:
            prior: BaseAbstractPrior = gen.send(prior_response)
            d = prior.base_ndims
            U_ndims += d
            u = jnp.zeros(prior.base_shape, float_type)
            prior_response = prior.forward(u)
            if prior.name is not None:
                if prior.name in names:
                    raise InvalidPriorName(name=prior.name)
                names.add(prior.name)
        except StopIteration:
            break
    return U_ndims


def simulate_prior_model(key: PRNGKey, prior_model: PriorModelType) -> Tuple[LikelihoodInputType, XType]:
    """
    Simulate a prior model.

    Args:
        key: PRNGKey
        prior_model: A prior model

    Returns:
        a tuple of the likelihood input variables, and dict of non-hidden (named) prior variables.
    """
    # parse prior
    U_placeholder, _, W_placeholder = parse_prior(prior_model=prior_model)
    _, unravel_fn = pytree_unravel(W_placeholder)
    U = jax.random.uniform(key, shape=U_placeholder.shape, dtype=U_placeholder.dtype)
    W = unravel_fn(U)
    return prepare_input(W=U, prior_model=prior_model), transform(W=W, prior_model=prior_model)


def parse_prior(prior_model: PriorModelType) -> Tuple[UType, XType, WType]:
    """
    Computes placeholders of model.

    Args:
        prior_model: a callable that produces a prior model generator

    Returns:
        U placeholder, X placeholder, W placeholder
    """
    U_ndims = 0
    gen = _get_prior_model_gen(prior_model=prior_model)

    prior_response = None
    names = set()
    X_placeholder: XType = dict()
    W_placeholder: WType = ()
    while True:
        try:
            prior: BaseAbstractPrior = gen.send(prior_response)
            d = prior.base_ndims
            U_ndims += d
            u = jnp.zeros(prior.base_shape, float_type)
            W_placeholder += (u,)
            prior_response = prior.forward(u)
            if prior.name is not None:
                if prior.name in names:
                    raise InvalidPriorName(name=prior.name)
                names.add(prior.name)
                if not isinstance(prior, SingularPrior):
                    X_placeholder[prior.name] = prior_response
        except StopIteration:
            break
    U_placeholder = jnp.zeros((U_ndims,), float_type)
    return U_placeholder, X_placeholder, W_placeholder


def parse_joint(prior_model: PriorModelType, log_likelihood: LikelihoodType) -> Tuple[
    UType, XType, WType, LikelihoodInputType, FloatArray
]:
    """
    Computes placeholders of model.

    Args:
        prior_model: a callable that produces a prior model generator

    Returns:
        U placeholder, X placeholder, W placeholder, likelihood input placeholder, log likelihood placeholder
    """
    U_ndims = 0
    gen = _get_prior_model_gen(prior_model=prior_model)

    prior_response = None
    names = set()
    X_placeholder: XType = dict()
    W_placeholder: WType = ()
    while True:
        try:
            prior: BaseAbstractPrior = gen.send(prior_response)
            d = prior.base_ndims
            U_ndims += d
            u = jnp.zeros(prior.base_shape, float_type)
            W_placeholder += (u,)
            prior_response = prior.forward(u)
            if prior.name is not None:
                if prior.name in names:
                    raise InvalidPriorName(name=prior.name)
                names.add(prior.name)
                if not isinstance(prior, SingularPrior):
                    X_placeholder[prior.name] = prior_response
        except StopIteration as e:
            output = e.value
            if (not isinstance(output, tuple)) or isinstance_namedtuple(output):
                output = (output,)
            break
    likelihood_input_placeholder = output
    log_L_placeholder = jnp.asarray(log_likelihood(*output), float_type)
    U_placeholder = jnp.zeros((U_ndims,), float_type)
    return U_placeholder, X_placeholder, W_placeholder, likelihood_input_placeholder, log_L_placeholder


def transform(W: WType, prior_model: PriorModelType) -> XType:
    """
    Transforms a W sample into the prior variables.

    Args:
        W: tuple of W-space samples
        prior_model: a callable that produces a prior model generator

    Returns:
        the prior variables
    """

    gen = _get_prior_model_gen(prior_model=prior_model)

    prior_response = None
    names = set()
    X_collection = dict()
    idx = 0
    while True:
        try:
            prior: BaseAbstractPrior = gen.send(prior_response)
            u = W[idx]
            idx += 1
            prior_response = prior.forward(u)
            if prior.name is not None:
                if prior.name in names:
                    raise InvalidPriorName(name=prior.name)
                names.add(prior.name)
                if not isinstance(prior, SingularPrior):
                    X_collection[prior.name] = prior_response
        except StopIteration:
            break
    return X_collection


def transform_parametrised(W: WType, prior_model: PriorModelType) -> XType:
    """
    Transforms a W sample into the parametrised prior variables.

    Args:
        W: tuple of W-space samples
        prior_model: a callable that produces a prior model generator

    Returns:
        the parametrised prior variables
    """

    gen = _get_prior_model_gen(prior_model=prior_model)

    prior_response = None
    names = set()
    X_collection = dict()
    idx = 0
    while True:
        try:
            prior: BaseAbstractPrior = gen.send(prior_response)
            u = W[idx]
            idx += 1
            prior_response = prior.forward(u)
            if prior.name is not None:
                if prior.name in names:
                    raise InvalidPriorName(name=prior.name)
                names.add(prior.name)
                if isinstance(prior, SingularPrior):
                    X_collection[prior.name] = prior_response
        except StopIteration:
            break
    return X_collection


def prepare_input(W: WType, prior_model: PriorModelType) -> LikelihoodInputType:
    """
    Transforms a W sample into the likelihood conditional variables.

    Args:
        W: tuple of W-space samples
        prior_model: a callable that produces a prior model generator

    Returns:
        the conditional variables of likelihood model
    """

    gen = _get_prior_model_gen(prior_model=prior_model)

    prior_response = None
    idx = 0
    while True:
        try:
            prior: BaseAbstractPrior = gen.send(prior_response)
            u = W[idx]
            idx += 1
            prior_response = prior.forward(u)
        except StopIteration as e:
            output = e.value
            if (not isinstance(output, tuple)) or isinstance_namedtuple(output):
                output = (output,)
            break
    return output


def compute_log_prob_prior(W: WType, prior_model: PriorModelType) -> FloatArray:
    """
    Computes the prior log-density from a W-space sample.

    Args:
        W: tuple of W-space samples
        prior_model: a callable that produces a prior model generator

    Returns:
        prior log-density
    """

    gen = _get_prior_model_gen(prior_model=prior_model)

    prior_response = None
    log_prob = []
    idx = 0
    while True:
        try:
            prior: BaseAbstractPrior = gen.send(prior_response)
            u = W[idx]
            idx += 1
            prior_response = prior.forward(u)
            log_prob.append(prior.log_prob(prior_response))
        except StopIteration:
            break
    if len(log_prob) == 0:
        return jnp.asarray(0., float_type)
    else:
        return sum(log_prob[1:], log_prob[0])


def compute_log_likelihood(W: WType, prior_model: PriorModelType, log_likelihood: LikelihoodType,
                           allow_nan: bool = False) -> FloatArray:
    """
    Computes the log likelihood from W-space sample.

    Args:
        W: tuple of W-space samples
        prior_model: a callable that produces a prior model generator
        log_likelihood: callable that takes arrays returned by the prior model and returns a scalar float
        allow_nan: whether to allow nans in likelihood

    Returns:
        log-likelihood
    """

    V = prepare_input(W=W, prior_model=prior_model)
    log_L = jnp.asarray(log_likelihood(*V), float_type)
    if log_L.size != 1:
        raise ValueError(f"Log likelihood should be scalar, but got {log_L.shape}.")
    if log_L.shape != ():
        log_L = lax.reshape(log_L, ())
    if not allow_nan:
        is_nan = lax.ne(log_L, log_L)
        log_L = lax.select(is_nan, jnp.asarray(-jnp.inf, log_L.dtype), log_L)
    return log_L


def memoize_prior_model(prior_model: PriorModelType, *args, **kwargs) -> Generator[
    BaseAbstractPrior, RandomVariableType, Callable
]:
    """
    Memoize the prior model into a pure function. This can be used, e.g. to compute jacobians, or gradients inside a
    prior model.

    Args:
        prior_model: a prior model optionally with *args or **kwargs.
        *args: inputs to prior model
        **kwargs: inputs to prior model

    Returns:
        a generator that eventually returns a pure function of prior_model inputs.
    """
    gen = prior_model(*args, **kwargs)
    prior_response = None
    stack = []
    while True:
        try:
            prior: BaseAbstractPrior = gen.send(prior_response)
            prior_response = yield prior
            stack.append(prior_response)
        except StopIteration as e:
            # output = e.value
            break

    def _pure_fn(*args, **kwargs):
        gen = prior_model(*args, **kwargs)
        stack_iter = iter(stack)
        prior_response = None
        while True:
            try:
                _ = gen.send(prior_response)
                # Pass the response that was given at the time of memoization.
                prior_response = next(stack_iter)
            except StopIteration as e:
                output = e.value
                break
        return output

    return _pure_fn
