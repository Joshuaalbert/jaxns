from collections import OrderedDict
from typing import Union, List, Set
from jax import random, numpy as jnp, disable_jit, tree_map
from jax.flatten_util import ravel_pytree
from jaxns.utils import iterative_topological_sort, tuple_prod, broadcast_shapes
import logging
import string
import random as py_random

logger = logging.getLogger(__name__)


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
            shape: the shape of samples in X domain from the prior.
            parents: a list of Priors that this one depends on.
            tracked: bool, whether to collect these variables in X space
            prior_base: the base RV for this prior
            dtype: dtype of the prior in X domain. If None then assume the same as prior_base.dtype
        """
        self._name = name
        self._parents = list(parents)
        self._tracked = tracked
        self._prior_base = prior_base
        self._built = False

        global _PRIOR_CHAIN_INDEX_STACK, _PRIOR_CHAINS
        if len(_PRIOR_CHAIN_INDEX_STACK) > 0:  # only push on if there is a context to push unto.
            prior_chain: PriorChain = _PRIOR_CHAINS[_PRIOR_CHAIN_INDEX_STACK[-1]]
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
    
    def getitem(self, item, *, name=None, tracked=False):
        def getitem(x):
            return x[item]
        return unnary_prior_op(getitem)(self, name=name, tracked=tracked)
    
    def negative(self, *, name=None, tracked=False):
        return unnary_prior_op(jnp.negative)(self, name=name, tracked=tracked)

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

    def prior_chain(self) -> 'PriorChain':
        """
        Return a PriorChain of this prior and its ancestors.

        Returns:
            PriorChain
        """
        return PriorChain(self)

    @property
    def dtype(self):
        """
        Dtype of the prior in X domain. Usually the same as prior base.
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
        The shape of the prior in X space.
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
             X ~ F, and
             Y ~ G, where F and G have the same support.
             Then, any expectation of any function f over F can be written in terms of an expectation over G according to,
             E_F[f] = int f(y) dF/dG

             dF/dG is the homogeneous measure, a Radon–Nikodym derivative.

        This allows one to sample or compute expectations over one RV if they only know how to sample from another.

        Practical example:

            X ~ Gamma distribution (has a difficult quantile function), with density (p_X)
            Y ~ Normal distribution (easy to sample from), with density (p_Y)

            dF/dG = p_X/p_Y, the ratio of the densities.

            log-homogeneous density would be log(p_X) - log(p_Y)

        Args:
            **X: dict of X domain prior samples.

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
            jnp.ndarray with correct shape in X domain.
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
        return binary_prior_op(binary_op)(self, other, name=name)
    elif isinstance(other, jnp.ndarray):
        return unnary_prior_op(lambda x: binary_op(x, other))(self, name=name, tracked=tracked)
    else:
        raise ValueError(f"Invalid type {type(other)}")


def binary_prior_op(binary_op):
    class PriorBinaryOp(Prior):
        def __init__(self, prior_a, prior_b, *, name=None, tracked=False):
            """
            Prior for the binary op of two priors.

            Args:
                prior_a, prior_b: priors to apply binary op to
            """
            if name is None:
                name = f"_{prior_a.name}_{binary_op.__name__}_{prior_b.name}_{''.join(random.choice(string.ascii_lowercase) for i in range(4))}"
            super(PriorBinaryOp, self).__init__(name, [prior_a, prior_b], tracked, PriorBase())

        def transform_U(self, U, prior_a, prior_b, **kwargs):
            del U
            return binary_op(prior_a, prior_b)

    return PriorBinaryOp


def unnary_prior_op(unary_op):
    class PriorUnaryOp(Prior):
        def __init__(self, prior_a, *, name=None, tracked=False):
            """
            Prior for the unary op of a prior.

            Args:
                prior_a: prior to apply unary op to
            """
            if name is None:
                name = f"_{prior_a.name}_{unary_op.__name__}_{''.join(random.choice(string.ascii_lowercase) for i in range(4))}"
            super(PriorUnaryOp, self).__init__(name, [prior_a], tracked, PriorBase())

        def transform_U(self, U, prior_a, **kwargs):
            del U
            return unary_op(prior_a)

    return PriorUnaryOp


_PRIOR_CHAINS = dict()
_PRIOR_CHAIN_NEXT_INDEX = 0  # which prior chain we are on
_PRIOR_CHAIN_INDEX_STACK = []


class PriorChain(object):
    # TODO: use a global variable to collect prior chains in a context so they don't need to be pushed on.
    # can likely make this much more flexible to allow much more complex models.
    def __init__(self, *priors: Union[Prior, 'PriorChain']):
        """
        PriorChain is an object that aggregates all the priors required for a model into a single Prior.
        These priors need to be pushed on, in any particular order. There is topological tracing, such that only leaf
        priors need be pushed on.

        Args:
            priors: list of Prior's or another PriorChain to be pushed onto this one.

        PriorChain represents a hierarchical prior, and thus it can also handle pushing another PriorChain onto it.
        This has the effect of creating a subspace which is sampled separately. For example:

        >>> from jaxns.prior_transforms.common import NormalPrior
        # no need to push onto prior_chain
        >>> with PriorChain() as prior_chain:
        >>>     x = NormalPrior('x', 0., 1.)
        >>>     y  = NormalPrior('y', x, 1.)
        """
        global _PRIOR_CHAIN_NEXT_INDEX, _PRIOR_CHAINS
        self._prior_chain_index = _PRIOR_CHAIN_NEXT_INDEX
        _PRIOR_CHAIN_NEXT_INDEX += 1
        _PRIOR_CHAINS[self._prior_chain_index] = self

        self._built = False
        self._prior_chain = dict()
        self._prior_chain_placeholders = dict()
        self._prior_U_placeholders = dict()

        for prior in priors:
            self.push(prior)

    def __enter__(self):
        global _PRIOR_CHAIN_INDEX_STACK
        _PRIOR_CHAIN_INDEX_STACK.append(self._prior_chain_index)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _PRIOR_CHAIN_INDEX_STACK
        _PRIOR_CHAIN_INDEX_STACK.pop()

    def build(self, **kwargs):
        """
        Run once before using to construct the unravel function.

        Args:
            **kwargs: passed to each prior transform_U function.

        """
        if self.built:
            logger.warning("PriorChain was already built. "
                           "You may be trying to re-use a PriorChain in another NestedSampler. "
                           "Use the same sample NestedSampler.")

        # not strictly necessary in python 3, but we do it for consistency.
        self._prior_U_placeholders = self._order(self._prior_U_placeholders)
        self._U_flat_placeholder, self._unravel_U_flat = ravel_pytree(self._prior_U_placeholders)
        self._built = True
        self._prior_chain_placeholders = self._order(self._prior_chain_placeholders)

    def unravel_U_flat(self, U_flat):
        assert self.built
        return self._unravel_U_flat(U_flat)

    @property
    def U_flat_placeholder(self):
        assert self.built
        return self._U_flat_placeholder

    @property
    def sample_placeholder(self):
        assert self.built
        return self._prior_chain_placeholders

    @property
    def built(self):
        return self._built

    def push(self, prior: Prior):
        """
        Pushs a prior onto the chain, so that the model knows about it. Priors are topologically traced through parents
        such that only leaf priors need be pushed on.
        """
        if self.built:
            raise AssertionError("Can't push on a PriorChain which has already been built, "
                                 "e.g. you've already fed it to a NestedSampler.")
        if not isinstance(prior, Prior):
            raise TypeError("Expected Prior, got {}".format(type(prior)))
        if (prior.name in self._prior_chain.keys()):
            if (prior not in self._prior_chain.values()):
                raise ValueError("Using the same name {} for two different priors.".format(prior.name))
            # TODO: maybe find a better way than filtering
            # logger.debug("Not necessary to push non-leaf prior {}.".format(prior.name))
        else:  # prior not yet in chain.
            self._prior_chain[prior.name] = prior
            for parent in prior.parents:
                self.push(parent)
            # build the prior as we push it on.
            test_U = jnp.zeros(prior.prior_base.shape, prior.prior_base.dtype)
            # these should all be arrays due to the topological order
            parents = [self._prior_chain_placeholders[parent.name] for parent in prior.parents]
            for parent in parents:
                if isinstance(parent, str):
                    raise ValueError("Parent not yet evaluated. Topological ordering must be incorrect.")
            self._prior_chain_placeholders[prior.name] = prior.build(test_U, *parents)
            self._prior_U_placeholders[prior.name] = test_U
        return self

    @property
    def U_ndims(self) -> int:
        """
        The dimension of the prior U domain.
        """
        return sum([p.U_ndims for p in self._prior_chain.values()], 0)

    @property
    def shapes(self):
        """
        Dict of shapes of prior in X domain.
        """
        shapes = {name: p.shape for name, p in self._prior_chain.items()}
        return self._order(shapes)

    @property
    def dtypes(self):
        """
        Dict of dtypes of prior in X domain.
        """
        dtypes = {name: p.dtype for name, p in self._prior_chain.items()}
        return self._order(dtypes)

    def __repr__(self):
        parents_dsk = {p.name: [parent.name for parent in p.parents] for p in self._prior_chain.values()}
        forward_topo_sort = iterative_topological_sort(parents_dsk)[::-1]
        s = []
        for name in forward_topo_sort:
            s.append(
                f"{name} ~ {self._prior_chain[name].dtype}{self._prior_chain[name].shape} : {self._prior_chain[name]}")
        return "\n".join(s)

    def log_homogeneous_measure(self, **X):
        """
        The log-homogeneous measure, used to reweight the current sample by the appropriate measure.

        Example:
             Let
             X ~ F, and
             Y ~ G, where F and G have the same support.
             Then, any expectation of any function f over F can be written in terms of an expectation over G according to,
             E_F[f] = int f(y) dF/dG

             dF/dG is the homogeneous measure, a Radon–Nikodym derivative.

        This allows one to sample or compute expectations over one RV if they only know how to sample from another.

        Practical example:

            X ~ Gamma distribution (has a difficult quantile function), with density (p_X)
            Y ~ Normal distribution (easy to sample from), with density (p_Y)

            dF/dG = p_X/p_Y, the ratio of the densities.

            log-homogeneous density would be log(p_X) - log(p_Y)

        Args:
            **X: dict of X domain prior samples.

        Returns: scalar, or None if constant homogeneous measure.
        """
        parents_dsk = {p.name: [parent.name for parent in p.parents] for p in self._prior_chain.values()}
        forward_topo_sort = iterative_topological_sort(parents_dsk)[::-1]
        log_measure = []
        for i, name in enumerate(forward_topo_sort):
            prior = self._prior_chain[name]
            parents = [X[parent] for parent in parents_dsk[name]]
            _log_measure = prior.log_homogeneous_measure(X[name], *parents)
            if _log_measure is not None:
                log_measure.append(_log_measure)
        if len(log_measure) == 0:
            return None
        return sum(log_measure[1:], log_measure[0])

    def sample_U_flat(self, key):
        """
        Generate a sample from prior bases (U-domain) of prior, and return in flat representation.

        Returns:
            Flat representation of sample from prior bases
        """
        if not self.built:
            raise AssertionError("PriorChain must be built before sampling. "
                                 "Call prior_chain.build() first.")
        return random.uniform(key, (self.U_ndims,), dtype=self.U_flat_placeholder.dtype)

    def _order(self, obj):
        """
        Orders the items of dict to same as prior chain.

        Args:
            obj: dict-like

        Returns:
            OrderedDict with same order as prior_chain
        """
        # TODO: remove as not necessary with python 3 (likely not required at all, as pytrees work for dicts)
        obj_ordered = dict()
        for key in self._prior_chain.keys():
            if key not in obj.keys():
                raise KeyError(f"Key {key} not in U sample.")
            obj_ordered[key] = obj[key]
        return obj_ordered

    def compactify_U(self, U):
        """
        Flatten prior bases samples into a vector representation.

        Args:
            U: dict of prior bases samples in U domain.

        Returns:
            Flat vector representation of U
        """
        U = self._order(U)
        U_flat, _ = ravel_pytree(U)
        return U_flat

    def __call__(self, U_compact, **kwargs):
        """
        Transforms a compact representation of prior bases samples in U domain to X domain.

        Args:
            U_compact: flat vector of prior bases samples
            **kwargs: kwargs to pass to all prior transforms.

        Returns: dict of X domain priors.
        """
        if not self.built:
            raise AssertionError("PriorChain must be built before calling it. "
                                 "Call prior_chain.build() first.")
        U = self.unravel_U_flat(U_compact)  # dict of prior bases in U-domain
        # parents start out as place holder names.
        parents_dsk = {p.name: [parent.name for parent in p.parents] for p in self._prior_chain.values()}
        forward_topo_sort = iterative_topological_sort(parents_dsk)[::-1]
        for name in forward_topo_sort:
            prior = self._prior_chain[name]
            input = U[name]
            # these should all be arrays due to the topological order
            parents = [parents_dsk[parent] for parent in parents_dsk[name]]
            for parent in parents:
                if isinstance(parent, str):
                    raise ValueError("Parent not yet evaluated. Topological ordering must be incorrect.")
            parents_dsk[name] = prior(input, *parents, **kwargs)
        transformed_prior = parents_dsk
        return transformed_prior

    def test_prior(self, key, num_samples, log_likelihood=None, **kwargs):
        """
        Performs a rough sanity check. Loops over samples from the prior, ensuring that there are no nans.
        Optionally, checks that the log-likelihood returns no nans.

        Does not compile with jit.

        Args:
            key: PRNG key
            num_samples: int, number of samples to check
            log_likelihood: optional, callable(**prior, **kwargs)
            **kwargs: optional, dict to pass to log-likelihood and prior generation.
        """
        assert self.built
        with disable_jit():
            keys = random.split(key, num_samples)
            for key in keys:
                U = self.sample_U_flat(key)
                Y = self(U, **kwargs)
                for k, v in Y.items():
                    if jnp.any(jnp.isnan(v)):
                        raise ValueError('nan in prior transform {}'.format(Y))
                log_homogeneous_measure = self.log_homogeneous_measure(**Y)
                if log_homogeneous_measure is not None:
                    if jnp.isnan(log_homogeneous_measure):
                        raise ValueError("Log-homogeneous measure is nan at {}".format(Y))
                    logger.info("Log-homogeneous measure: {}".format(log_homogeneous_measure))
                if log_likelihood is not None:
                    loglik = log_likelihood(**Y, **kwargs)
                    if jnp.isnan(loglik):
                        raise ValueError("Log likelihood is nan at {}".format(Y))
                    logger.info("Log-likelihood: {}".format(loglik))
