import string
from typing import List
import logging

import numpy as np

logger = logging.getLogger(__name__)

from jax import random, numpy as jnp

from jaxns.internals.shapes import tuple_prod

import random as py_random

from jaxns.prior_transforms.context import _PRIOR_CHAINS, _PRIOR_CHAIN_NEXT_INDEX, _PRIOR_CHAIN_INDEX_STACK

class PriorBase(object):
    def __init__(self, shape=(0,), dtype=None):
        """
        This defines an independent RV which is transformed to a particular prior. JAXNS depends on the independence of the
        prior domains, and thus it only handles priors which can be represented as functional transforms of an independent
        RV. PriorBase represents these types of RVs.
        """
        self._shape = shape
        self._dtype = dtype

    @property
    def shape(self):
        """
        Shape of prior base (U domain).
        """
        return self._shape

    @property
    def U_ndims(self):
        return tuple_prod(self.shape)

    @property
    def dtype(self):
        """
        Dtype of prior base (U domain).
        """
        return self._dtype

    def _sample_base_RV(self, key, shape, dtype):
        """
        Samples from the prior base distribution.

        Args:
            key: PRNG key
            shape: dimension of RV

        Returns:
            jnp.array of shape [shape] or None
        """
        return None

    def sample_U(self, key):
        """
        Samples a base prior RV, which cannot be dependent on any other RV.

        Args:
            key: PRNG key

        Returns:
            jnp.array of shape [shape]
        """
        U = self._sample_base_RV(key, self.shape, self.dtype)
        if U is not None:
            assert U.dtype == self.dtype
            assert U.shape == self.shape
            return U
        return None


class UniformBase(PriorBase):
    """
    Provides the uniform RV base prior, from which the inverse cdf (quantile function) maps to a particular prior.
    """

    def __init__(self, shape, dtype):
        super(UniformBase, self).__init__(shape, dtype)

    def _sample_base_RV(self, key, shape, dtype):
        return random.uniform(key, shape=shape, minval=0., maxval=1., dtype=dtype)


class Prior(object):
    # TODO: use tfp.distributions and for those without icdf automatically use log_homogeneous_prior
    def __init__(self, name, parents, tracked: bool, prior_base: PriorBase):
        """
        Represents a prior as a transformation from another base prior distribution (typically the U[0,1]).
        A prior has two methods:

            . sample_U - generates a sample from the PriorBase, from the U domain.
            . transform_U - transforms the base RV (in U domain) to a sample from the prior in the codomain.

        The operations +, -, *, /, and ** are defined between Prior's.

        Args:
            name: name of the prior. Must be unique. Highly recommended not to prefix names with '_',
                since we use this prefix to name some parameters unique.
            shape: the shape of samples in U domain from the prior.
            parents: a list of Priors that this one depends on.
            tracked: bool, whether to collect these variables in U space
            prior_base: the base RV for this prior
            dtype: dtype of the prior in U domain. If None then assume the same as prior_base.dtype
        """
        self._name = name
        self._parents = list(parents)
        self._tracked = tracked
        self._prior_base = prior_base
        self._shape = None
        self._dtype = None
        self._built = False

        global _PRIOR_CHAIN_INDEX_STACK, _PRIOR_CHAINS
        if len(_PRIOR_CHAIN_INDEX_STACK) > 0:  # only push on if there is a context to push unto.
            prior_chain = _PRIOR_CHAINS[_PRIOR_CHAIN_INDEX_STACK[-1]]
            prior_chain.push(self)

    @property
    def built(self):
        return self._built

    def build(self, U, *parents, **kwargs) -> jnp.ndarray:
        if self.built:
            logger.warning(f"Trying to build a prior {self}, again.")
        output = self.transform_U(U, *parents, **kwargs)
        self._shape = output.shape
        self._dtype = output.dtype
        self._built = True
        return output

    def rename(self, new_name):
        assert not self._built
        self._name = new_name
        return self

    # operator overloads

    def __add__(self, other):
        return self.add(other)

    def __sub__(self, other):
        return self.subtract(other)

    def __mul__(self, other):
        self.multiply(other)

    def __truediv__(self, other):
        return self.true_divide(other)

    def __neg__(self):
        return self.negative()

    def __pow__(self, other):
        return self.power(other)

    def __matmul__(self, other):
        return self.matmul(other)

    def __getitem__(self, item):
        return self.getitem(item)

    # n-ary ops
    def interp(self, xp, fp, left=None, right=None, period=None, *, name=None, tracked=False):
        """
        Interpolates values are given new positions.

        This prior represents `x` in numpy syntax of interp(x,xp,fp).
        """
        def interp(x, xp, fp):
            return jnp.interp(x, xp, fp, left=left, right=right, period=period)
        return maybe_n_ary_prior_op(interp, [self, xp, fp], name=name, tracked=tracked)

    # binary ops
    def add(self, other, *, name=None, tracked=False):
        return maybe_binary_prior_op(jnp.add, self, other, name=name, tracked=tracked)

    def subtract(self, other, *, name=None, tracked=False):
        return maybe_binary_prior_op(jnp.subtract, self, other, name=name, tracked=tracked)

    def multiply(self, other, *, name=None, tracked=False):
        return maybe_binary_prior_op(jnp.multiply, self, other, name=name, tracked=tracked)

    def true_divide(self, other, *, name=None, tracked=False):
        return maybe_binary_prior_op(jnp.true_divide, self, other, name=name, tracked=tracked)

    def power(self, other, *, name=None, tracked=False):
        return maybe_binary_prior_op(jnp.power, self, other, name=name, tracked=tracked)

    def matmul(self, other, *, name=None, tracked=False):
        return maybe_binary_prior_op(jnp.matmul, self, other, name=name, tracked=tracked)

    # unary ops

    def getitem(self, item, *, name=None, tracked=False):
        def getitem(x):
            return x[item]

        return unnary_prior_op(getitem)(self, name=name, tracked=tracked)

    def negative(self, *, name=None, tracked=False):
        return unnary_prior_op(jnp.negative)(self, name=name, tracked=tracked)

    def sum(self, axis=-1, keepdims=False, *, name=None, tracked=False):
        def sum(x):
            return jnp.sum(x, axis=axis, keepdims=keepdims)

        return unnary_prior_op(sum)(self, name=name, tracked=tracked)

    def mean(self, axis=-1, keepdims=False, *, name=None, tracked=False):
        def mean(x):
            return jnp.mean(x, axis=axis, keepdims=keepdims)

        return unnary_prior_op(mean)(self, name=name, tracked=tracked)

    def sqrt(self, *, name=None, tracked=False):
        return unnary_prior_op(jnp.sqrt)(self, name=name, tracked=tracked)

    def square(self, *, name=None, tracked=False):
        return unnary_prior_op(jnp.square)(self, name=name, tracked=tracked)

    def reciprocal(self, *, name=None, tracked=False):
        return unnary_prior_op(jnp.reciprocal)(self, name=name, tracked=tracked)

    def exp(self, *, name=None, tracked=False):
        return unnary_prior_op(jnp.exp)(self, name=name, tracked=tracked)

    def log(self, *, name=None, tracked=False):
        return unnary_prior_op(jnp.log)(self, name=name, tracked=tracked)

    def transpose(self, axes=None, *, name=None, tracked=False):
        def transpose(x):
            return jnp.transpose(x, axes=axes)

        return unnary_prior_op(transpose)(self, name=name, tracked=tracked)

    def reshape(self, newshape, *, name=None, tracked=False):
        def reshape(x):
            return jnp.reshape(x, newshape=newshape)

        return unnary_prior_op(reshape)(self, name=name, tracked=tracked)

    @property
    def dtype(self):
        """
        Dtype of the prior in U domain. Usually the same as prior base.
        """
        if not self.built:
            raise ValueError("Used outside a PriorChain context.")
        return self._dtype

    @property
    def prior_base(self) -> PriorBase:
        """
        The PriorBase of this prior.
        """
        return self._prior_base

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, ",".join([p.name for p in self.parents]))

    @property
    def parents(self) -> List['Prior']:
        """
        List of parent Priors.
        """
        return self._parents

    @property
    def name(self) -> str:
        """
        Name of prior. Should be unique among priors that will be pushed to the same prior chain.
        """
        return self._name

    @property
    def U_ndims(self) -> int:
        """
        Dimension of prior base RV needed to sample this prior. Should be zero for deterministic transforms of parents.
        Passes through the prior_base U_ndims
        """
        return self.prior_base.U_ndims

    @property
    def tracked(self) -> bool:
        """
        Whether or not to collect this prior.
        """
        return self._tracked

    @property
    def shape(self):
        """
        The shape of the prior in U space.
        """
        if not self.built:
            raise ValueError("Used outside a PriorChain context.")
        return self._shape

    def sample_U(self, key):
        """
        Samples a RV from the PriorBase.
        Args:
            key: PRNG

        Returns:
            A array containing i.i.d. RVs from PriorBase, of shape prior_base.shape
        """
        return self.prior_base.sample_U(key)

    def log_homogeneous_measure(self, X, *parents):
        """
        The log-homogeneous measure, used to reweight the current sample by the appropriate measure.

        Example:
             Let
             U ~ F, and
             Y ~ G, where F and G have the same support.
             Then, any expectation of any function f over F can be written in terms of an expectation over G according to,
             E_F[f] = int f(y) dF/dG

             dF/dG is the homogeneous measure, a Radon–Nikodym derivative.

        This allows one to sample or compute expectations over one RV if they only know how to sample from another.

        Practical example:

            U ~ Gamma distribution (has a difficult quantile function), with density (p_X)
            Y ~ Normal distribution (easy to sample from), with density (p_Y)

            dF/dG = p_X/p_Y, the ratio of the densities.

            log-homogeneous density would be log(p_X) - log(p_Y)

        Args:
            **X: dict of U domain prior samples.

        Returns: scalar, or None if constant homogeneous measure.
        """
        # default is constant homogeneous prior
        return None

    def transform_U(self, U, *parents, **kwargs):
        """
        Transforms a vector of i.i.d. random variables from PriorBase to another random variable.
        The transformation may be dependent on any parent RVs, as well as any kwargs.
        Args:
            U: base RV
            parents: list of RVs early in the chain.
            kwargs: dict of arbitrary extra parameters.
        Returns:
            jnp.ndarray with correct shape in U domain.
        """
        raise NotImplementedError()

    def __call__(self, U, *parents, **kwargs):
        """
        Calls the transform_U method and checks that shapes match.
        """
        assert self.built, "Must build prior first, i.e. feed PriorChain to a NestedSampler."
        transformed_prior = self.transform_U(U, *parents, **kwargs)
        if transformed_prior.shape != self.shape:
            raise ValueError("Expected shape_dict {}, got {}.".format(self.shape, transformed_prior.shape))
        if self.dtype != transformed_prior.dtype:
            raise ValueError("Expected dtype {} got {}.".format(self.dtype, transformed_prior.dtype))
        return transformed_prior


def maybe_binary_prior_op(binary_op, self, other, *, name=None, tracked=False):
    if isinstance(other, Prior):
        return binary_prior_op(binary_op)(self, other, name=name, tracked=tracked)
    elif isinstance(other, (float, int, bool)):
        other = jnp.asarray(other)
        return unnary_prior_op(lambda x: binary_op(x, other))(self, name=name, tracked=tracked)
    elif isinstance(other, jnp.ndarray):
        return unnary_prior_op(lambda x: binary_op(x, other))(self, name=name, tracked=tracked)
    else:
        raise ValueError(f"Invalid type {type(other)}")


def binary_prior_op(binary_op):
    class PriorBinaryOp(Prior):
        def __init__(self, prior_a: Prior, prior_b:Prior, *, name=None, tracked=False):
            """
            Prior for the binary op of two priors.

            Args:
                prior_a, prior_b: priors to apply binary op to
            """
            if name is None:
                name = f"_{binary_op.__name__}_{prior_a.name}_{prior_b.name}_{''.join(py_random.choice(string.ascii_lowercase) for i in range(4))}"
            super(PriorBinaryOp, self).__init__(name, [prior_a, prior_b], tracked, PriorBase())

        def transform_U(self, U, prior_a, prior_b, **kwargs):
            del U
            return binary_op(prior_a, prior_b)

    return PriorBinaryOp


def unnary_prior_op(unary_op):
    class PriorUnaryOp(Prior):
        def __init__(self, prior_a: Prior, *, name=None, tracked=False):
            """
            Prior for the unary op of a prior.

            Args:
                prior_a: prior to apply unary op to
            """
            if name is None:
                name = f"_{unary_op.__name__}_{prior_a.name}_{''.join(py_random.choice(string.ascii_lowercase) for i in range(4))}"
            super(PriorUnaryOp, self).__init__(name, [prior_a], tracked, PriorBase())

        def transform_U(self, U, prior_a, **kwargs):
            del U
            return unary_op(prior_a)

    return PriorUnaryOp

def maybe_n_ary_prior_op(n_ary_op, priors, *, name=None, tracked=False):
    prior_args = []
    static_args = dict()
    num_inputs = len(priors)
    for idx, prior in enumerate(priors):
        if isinstance(prior, Prior):
            prior_args.append(idx)
        elif isinstance(prior, (float, int, bool)):
            prior = jnp.asarray(prior)
            static_args[idx] = prior
        elif isinstance(prior, jnp.ndarray):
            static_args[idx] = prior
        else:
            raise ValueError(f"Invalid type {type(prior)}")

    def curried_n_ary_op(*priors):
        #reconstruct input to op
        args = [None]*num_inputs
        for static_arg, static_val in static_args.items():
            args[static_arg] = static_val
        for idx, prior in zip(prior_args, priors):
            args[idx] = prior
        # call with replaced values
        return n_ary_op(*args)

    # run filtered input on curried function
    filtered_priors = list(filter(lambda x: isinstance(x, Prior), priors))
    return n_ary_prior_op(curried_n_ary_op)(filtered_priors, name=name, tracked=tracked)

def n_ary_prior_op(n_ary_op):
    class PriorNAryOp(Prior):
        def __init__(self, priors: List[Prior], *, name=None, tracked=False):
            """
            Prior for the n-ary op of a set of priors.

            Args:
                *priors: list of priors to apply n-ary op to
            """
            if name is None:
                name = f"_{n_ary_op.__name__}_{'_'.join([prior.name for prior in priors])}_{''.join(py_random.choice(string.ascii_lowercase) for i in range(4))}"
            super(PriorNAryOp, self).__init__(name, priors, tracked, PriorBase())

        def transform_U(self, U, *priors, **kwargs):
            del U
            return n_ary_op(*priors)

    return PriorNAryOp


def prior_docstring(f):
    """
    Puts the Prior docstring below each prior init.
    Args:
        f: callable
    """
    if f.__doc__ is None:
        logger.warning("{} has no docstring".format(f.__name__))
        f.__doc__ = ""
    f.__doc__ = f.__doc__+"\n\nGeneral Prior documentation:\n\n"+Prior.__init__.__doc__
    return f


def get_shape(v):
    """
    Gets shape from a value regardless of what it might be.

    Args:
        v: Prior, array, list, tuple, scalar

    Returns: tuple of shape
    """
    if isinstance(v, Prior):
        return v.shape
    if isinstance(v, (jnp.ndarray, np.ndarray)):
        return v.shape
    if isinstance(v, (list, tuple)):
        return np.asarray(v).shape
    return ()