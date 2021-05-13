from collections import OrderedDict
from typing import Union, List, Set
from jax import random, numpy as jnp, disable_jit
from jaxns.utils import iterative_topological_sort, tuple_prod, broadcast_shapes
import logging

logger = logging.getLogger(__name__)

class PriorBase(object):
    def __init__(self, shape, dtype):
        """
        This defines an independent RV which is transformed to a particular prior. JAXNS depends on the independence of the
        prior domains, and thus it only handles priors which can be represented as functional transforms of an independent
        RVs. PriorBase represents these types of RVs.

        A prior in JAXNS is always a transform of an independent RV. This excludes some priors that have no inverse cdf
        (quantile function). For these types of priors an trick should be used of multiplying the likelihood by
        the prior density.
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
    def dtype(self):
        """
        Dtype of prior base (U domain).
        """
        return self._dtype

    def _sample_base_RV(self, key, shape, dtype):
        """
        Samples a base prior RV, which cannot be dependent on any other RV.

        Args:
            key: PRNG key
            shape: dimension of RV

        Returns:
            jnp.array of shape [shape]
        """
        return ()

    def sample_U(self, key):
        """
        Samples a base prior RV, which cannot be dependent on any other RV.

        Args:
            key: PRNG key

        Returns:
            jnp.array of shape [shape]
        """
        return self._sample_base_RV(key, self.shape, self.dtype)


class ContinuousBase(PriorBase):
    """
    Handles continuous RVs on a closed interval [lower, upper]. Lower and upper must be finite.
    """

    def __init__(self, lower, upper, shape, dtype):
        self._lower = lower
        self._upper = upper
        super(ContinuousBase, self).__init__(shape, dtype)

    @property
    def lower(self):
        """
        Inclusive lower limit of the continuous RV.
        """
        return self._lower

    @property
    def upper(self):
        """
        Inclusive upper limit of the continuous RV.
        """
        return self._upper


class UniformBase(ContinuousBase):
    """
    Provides the uniform RV base prior, from which the inverse cdf (quantile function) maps to a particular prior.

    Alternatively, a trick can be used to map to priors without a quantile function by multiplying the likelihood by
    the prior density and using a transformed uniform RV to sample within the support of the random variable. An
    approximation of the support can be used, e.g. truncating the support at several standard deviations from the mean.
    """

    def __init__(self, shape, dtype):
        super(UniformBase, self).__init__(0., 1., shape, dtype)

    def _sample_base_RV(self, key, shape, dtype):
        return random.uniform(key, shape=shape, minval=0., maxval=1., dtype=dtype)


class DiscreteBase(PriorBase):
    """
    Handles discrete RVs with a set number of outcomes. They are represented with integers between 0 (inclusive) and
    num_outcomes (exclusive).
    """

    def __init__(self, num_outcomes, shape, dtype):
        self._num_outcomes = num_outcomes
        if not tuple_prod(shape) == 1:
            raise ValueError(f"All discrete variables currently must be of size 1, got shape {shape}")
        super(DiscreteBase, self).__init__(shape, dtype)

    @property
    def num_outcomes(self):
        """
        Number of discrete outcomes.
        """
        return self._num_outcomes

    def _sample_base_RV(self, key, shape, dtype):
        return random.randint(key, shape=shape, minval=0, maxval=self.num_outcomes, dtype=dtype)


class BinaryBase(DiscreteBase):
    """
    Binary discrete RV.
    """

    def __init__(self, shape, dtype):
        super(BinaryBase, self).__init__(2, shape, dtype)


class Prior(object):
    def __init__(self, name, shape, parents, tracked: bool, prior_base: PriorBase, dtype=None):
        """
        Represents a prior as a transformation from an independent RV. A prior has two render_team_chart methods:

            . sample_U - generates a sample from the PriorBase (which we call the U domain).
            . transform_U - transforms the base RV (in U domain) to a sample from the prior
                            (which we call the X domain).

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
        self._shape = shape
        self._parents = list(parents)
        self._tracked = tracked
        self._prior_base = prior_base
        if dtype is None:
            dtype = prior_base.dtype
        self._dtype = dtype
        self._subspace = None

    def __add__(self, other):
        class AddPrior(Prior):
            def __init__(self, prior_a, prior_b):
                """
                Prior for the sum of two priors.

                Args:
                    prior_a, prior_b: priors to add together
                """
                to_shape = broadcast_shapes(prior_a.shape, prior_b.shape)
                name = "_{}_add_{}".format(prior_a.name, prior_b.name)
                super(AddPrior, self).__init__(name, to_shape, [prior_a, prior_b], False, PriorBase((), jnp.float_))

            def transform_U(self, U, prior_a, prior_b, **kwargs):
                del U
                return prior_a + prior_b

        return AddPrior(self, other)

    def __sub__(self, other):
        class SubPrior(Prior):
            def __init__(self, prior_a, prior_b):
                """
                Prior for the difference of two priors.

                Args:
                    prior_a, prior_b: priors to subtract
                """
                to_shape = broadcast_shapes(prior_a.shape, prior_b.shape)
                name = "_{}_sub_{}".format(prior_a.name, prior_b.name)
                super(SubPrior, self).__init__(name, to_shape, [prior_a, prior_b], False, PriorBase((), jnp.float_))

            def transform_U(self, U, prior_a, prior_b, **kwargs):
                del U
                return prior_a - prior_b

        return SubPrior(self, other)

    def __mul__(self, other):
        class MulPrior(Prior):
            def __init__(self, prior_a, prior_b):
                """
                Prior for the product of two priors.

                Args:
                    prior_a, prior_b: priors to multiply together
                """
                to_shape = broadcast_shapes(prior_a.shape, prior_b.shape)
                name = "_{}_mul_{}".format(prior_a.name, prior_b.name)
                super(MulPrior, self).__init__(name, to_shape, [prior_a, prior_b], False, PriorBase((), jnp.float_))

            def transform_U(self, U, prior_a, prior_b, **kwargs):
                del U
                return prior_a * prior_b

        return MulPrior(self, other)

    def __truediv__(self, other):
        class DivPrior(Prior):
            def __init__(self, prior_a, prior_b):
                """
                Prior for the division of two priors.

                Args:
                    prior_a, prior_b: priors to divide
                """
                to_shape = broadcast_shapes(prior_a.shape, prior_b.shape)
                name = "_{}_div_{}".format(prior_a.name, prior_b.name)
                super(DivPrior, self).__init__(name, to_shape, [prior_a, prior_b], False, PriorBase((), jnp.float_))

            def transform_U(self, U, prior_a, prior_b, **kwargs):
                del U
                return prior_a / prior_b

        return DivPrior(self, other)

    def __neg__(self):
        class NegPrior(Prior):
            def __init__(self, prior_a):
                """
                Prior for the negation of a prior.

                Args:
                    prior_a: prior to negate
                """
                to_shape = prior_a.shape
                name = "_neg_{}".format(prior_a.name)
                super(NegPrior, self).__init__(name, to_shape, [prior_a], False, PriorBase((), jnp.float_))

            def transform_U(self, U, prior_a, **kwargs):
                del U
                return -prior_a

        return NegPrior(self)

    def __pow__(self, power, modulo=None):
        class PowPrior(Prior):
            def __init__(self, prior_a, prior_b):
                """
                Prior for prior_a^prior_b.

                Args:
                    prior_a, prior_b: priors to transform
                """
                to_shape = broadcast_shapes(prior_a.shape, prior_b.shape)
                name = "_{}_pow_{}".format(prior_a.name, prior_b.name)
                super(PowPrior, self).__init__(name, to_shape, [prior_a, prior_b], False, PriorBase((), jnp.float_))

            def transform_U(self, U, prior_a, prior_b, **kwargs):
                del U
                return jnp.power(prior_a, prior_b)

        return PowPrior(self, power)

    def prior_chain(self) -> 'PriorChain':
        """
        Return a PriorChain of this prior and its ancestors.

        Returns:
            PriorChain
        """
        return PriorChain(self)

    @property
    def subspace(self) -> Set[str]:
        """
        The computational subspace that this prior is part of.
        """
        return self._subspace

    @subspace.setter
    def subspace(self, value: Set[str]):
        self._subspace = value

    @property
    def dtype(self):
        """
        Dtype of the prior in X domain. Usually the same as prior base.
        """
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
        """
        return tuple_prod(self.prior_base.shape)

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
        return self._shape

    def sample_U(self, key):
        """
        Samples a RV from the PriorBase.
        Args:
            key: PRNG

        Returns:
            A array containing i.i.d. RVs from PriorBase.
        """
        return self.prior_base.sample_U(key)

    def log_homogeneous_measure(self, X, *parents):
        """
        Computes the associated log-measure of the transform.
        That is if,

        X ~ transform_U(U, *parents)
        this computes m(X, *parents) such that this prior density is correct,
        p_X(X, U) = P_U(U | parents) m(X | U, parents)

        How this is used in practice is to modulate a sample from some distribution into the desired.
        e.g. X ~ Uniform, m=Gamma_pdf => X will sample Gamma prior.

        Args:
            X: sample in X domain
            *parents: priors this depends on.
            **kwargs:

        Returns: log-homogeneous prior, or None for constant, i.e. zero.
        """
        # default is flat homogeneous prior
        return None

    def transform_U(self, U, *parents, **kwargs):
        """
        Transforms a vector of i.i.d. random variables from PriorBase to another random variable.
        The transformation may be dependent on any ancestor RVs which are called parents, as well as any kwargs.
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
        transformed_prior = self.transform_U(U, *parents, **kwargs)
        if transformed_prior.shape != self.shape:
            raise ValueError("Expected shape_dict {}, got {}.".format(self.shape, transformed_prior.shape))
        if self.dtype != transformed_prior.dtype:
            raise ValueError("Expected dtype {} got {}.".format(self.dtype, transformed_prior.dtype))
        return transformed_prior


class PriorChain(object):
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
        >>> prior_chain1 = PriorChain(NormalPrior('x', 0., 1.))
        >>> prior_chain2 = PriorChain().push(NormalPrior('y', 0., 1.))
        >>> prior_chain2.push(prior_chain1)

        Subspaces are sampled sequentially, holding the other subspaces constant. This can lead to both better and worse
        behaviour depending on the model. If you are unsure, prefer using a single subspace, i.e. do not push a prior
        chain onto another.
        """
        self.prior_chain = OrderedDict()
        self._subspace = set()
        self._subspaces = []
        for prior in priors:
            self.push(prior)

    def _assert_no_subspace_intersection(self):
        """
        Assert that subspaces do not intersect.
        """
        for i in range(len(self.subspaces)):
            subspace_i = self.subspaces[i]
            for j in range(i + 1, len(self.subspaces)):
                subspace_j = self.subspaces[j]
                if len(subspace_i.intersection(subspace_j)) > 0:
                    raise ValueError(
                        "Subspaces {} and {} intersect\nsubspace 1 : {}\nsubspace 2 : {}\nintersection : {}".format(i,
                                                                                                                    j,
                                                                                                                    subspace_i,
                                                                                                                    subspace_j,
                                                                                                                    subspace_i.intersection(
                                                                                                                        subspace_j)))

    def push(self, prior: Union[Prior, 'PriorChain']):
        """
        Pushs a prior onto the chain, so that the model knows about it. Priors are topologically traced through parents
        such that only leaf priors need be pushed on.

        A prior is attached to the subspaces of the PriorChain that it is pushed onto.
        PriorChain's can also be pushed on, which has the effect of adding a separate subspace representing that
        prior chain. A prior chain is basically a tree of priors or other prior chains.
        """

        if not isinstance(prior, (Prior, PriorChain)):
            raise TypeError("Expected Prior or PriorChain, got {}".format(type(prior)))
        if isinstance(prior, PriorChain):
            # add subspaces to this one's
            self._subspaces += prior.subspaces
            for _name, _prior in prior.prior_chain.items():
                # all the priors in this prior_chain will be already certain to be unique.
                # self.prior_chain[_name] = _prior
                self.push(_prior)
        else:
            if (prior.name in self.prior_chain.keys()):
                if (prior not in self.prior_chain.values()):
                    raise ValueError("Using the same name {} for two different priors.".format(prior.name))
                logger.warning("Not necessary to push non-leaf prior {}.".format(prior.name))
            else:  # prior not yet in chain.
                self.prior_chain[prior.name] = prior
                if prior.subspace is None:
                    if isinstance(prior.prior_base, DiscreteBase):
                        # Priors with DiscreteBase priors get their own subspace
                        self._subspaces.append({prior.name})
                        prior.subspace = self._subspaces[-1]
                    else:
                        # Priors otherwise are appended to the current subspace
                        self._subspace.add(prior.name)
                        prior.subspace = self._subspace
                for parent in prior.parents:
                    self.push(parent)
        self._assert_no_subspace_intersection()
        return self

    def subspace_type(self, subspace):
        """
        Determine if subspace is discrete or continuous.

        Args:
            subspace: set of prior names

        Returns:
            str, 'discrete' if discrete, 'continuous' otherwise.
        """
        for name in subspace:
            if isinstance(self.prior_chain[name].prior_base, DiscreteBase):
                return 'discrete'
        return 'continuous'

    @property
    def subspaces(self):
        """
        Get list of all subspaces.
        """
        return [self.subspace] + self._subspaces

    @property
    def subspace(self) -> Set[str]:
        return self._subspace

    @property
    def U_ndims(self):
        """
        The dimension of the prior U domain.
        """
        return sum([p.U_ndims for p in self.prior_chain.values()])

    @property
    def shapes(self):
        """
        Dict of shapes of prior in X domain.
        """
        shapes = OrderedDict()
        for name, p in self.prior_chain.items():
            shapes[name] = p.shape
        return shapes

    @property
    def dtypes(self):
        """
        Dict of dtypes of prior in X domain.
        """
        dtypes = OrderedDict()
        for name, p in self.prior_chain.items():
            dtypes[name] = p.dtype
        return dtypes

    def __repr__(self):
        parents_dsk = {p.name: [parent.name for parent in p.parents] for p in self.prior_chain.values()}
        forward_topo_sort = iterative_topological_sort(parents_dsk)[::-1]
        s = []
        for name in forward_topo_sort:
            s.append("{} ~ {} : {}".format(name, self.prior_chain[name].shape, self.prior_chain[name]))
        return "\n".join(s)

    def log_homogeneous_measure(self, **X):
        """
        The log homogeneous measure, i.e. supposing that samples from this prior are X, then this measure defines
        the correct amount by which to weight them. Typically, only used to represent non-transformed priors like
        Gamma.

        Args:
            **X: dict of X domain prior samples.

        Returns: scalar, prior log-weight of the sample, None if constant, i.e. zero.
        """
        parents_dsk = {p.name: [parent.name for parent in p.parents] for p in self.prior_chain.values()}
        forward_topo_sort = iterative_topological_sort(parents_dsk)[::-1]
        log_measure = []
        for i, name in enumerate(forward_topo_sort):
            prior = self.prior_chain[name]
            parents = [X[parent] for parent in parents_dsk[name]]
            _log_measure = prior.log_homogeneous_measure(X[name], *parents)
            if _log_measure is not None:
                log_measure.append(_log_measure)
        if len(log_measure) == 0:
            return None
        return sum(log_measure[1:], log_measure[0])

    def sample_U(self, key):
        """
        Generate a sample from U domain of joint prior.

        Returns: dict of subspace U samples
        """
        parents_dsk = {p.name: [parent.name for parent in p.parents] for p in self.prior_chain.values()}
        forward_topo_sort = iterative_topological_sort(parents_dsk)[::-1]
        keys = random.split(key, len(self.prior_chain))
        U_samples = OrderedDict()

        for i, name in enumerate(forward_topo_sort):
            prior = self.prior_chain[name]
            U_samples[name] = prior.sample_U(keys[i])
        return U_samples

    def disperse_U(self, U_compact):
        """
        Turns an efficient organisation of U domain priors into a map to actual U domain priors.
        Undoes compactify_U.

        Args:
            U: tuple of two subspaces, see compactify_U.

        Returns: dict of of priors in U domain.
        """

        def _maybe_reshape(a, shape):
            if a.shape != shape:
                return jnp.reshape(a, shape)
            return a

        parents_dsk = {p.name: [parent.name for parent in p.parents] for p in self.prior_chain.values()}
        forward_topo_sort = iterative_topological_sort(parents_dsk)[::-1]
        subspaces_counters = [0 for _ in self.subspaces]
        U = OrderedDict()
        for i, name in enumerate(forward_topo_sort):
            prior = self.prior_chain[name]
            subspace_idx = self._which_subspace_idx(name)
            counter = subspaces_counters[subspace_idx]
            if isinstance(prior.prior_base, DiscreteBase):
                next_counter = counter + prior.U_ndims
                U[name] = _maybe_reshape(U_compact[subspace_idx][counter:next_counter], prior.prior_base.shape)
                subspaces_counters[subspace_idx] = next_counter
            elif isinstance(prior.prior_base, ContinuousBase):
                next_counter = counter + prior.U_ndims
                U[name] = _maybe_reshape(U_compact[subspace_idx][counter:next_counter], prior.prior_base.shape)
                subspaces_counters[subspace_idx] = next_counter
            elif isinstance(prior.prior_base, PriorBase):
                U[name] = ()
            else:
                raise TypeError("Expected PriorBase {}".format(type(prior.prior_base)))
        return U

    def _which_subspace_idx(self, name):
        """
        Gets the subspace index that name belongs to.

        Args:
            name: prior name

        Returns:
            index of the subspace it belongs to.
        """
        for idx, subspace in enumerate(self.subspaces):
            if name in subspace:
                return idx
        raise ValueError(f"Could not find subspace containing {name}. Open an issue!")

    def compactify_U(self, U):
        """
        Organise RVs in U domain into subspaces for efficient slice sampling.

        Continuous RVs get put into one subspace.
        Discrete RVs get put into another.
        In all cases the RVs are bounded.

        Args:
            U: dict of priors in U domain, which are produced by prior_base.

        Returns:
            tuple of priors in subspaces in U domain concatenated
        """
        parents_dsk = {p.name: [parent.name for parent in p.parents] for p in self.prior_chain.values()}
        forward_topo_sort = iterative_topological_sort(parents_dsk)[::-1]
        subspaces = [[] for _ in self.subspaces]

        for i, name in enumerate(forward_topo_sort):
            prior = self.prior_chain[name]
            subspace_idx = self._which_subspace_idx(name)
            if isinstance(prior.prior_base, DiscreteBase):
                subspaces[subspace_idx].append(U[name].reshape((-1,)))
            elif isinstance(prior.prior_base, ContinuousBase):
                subspaces[subspace_idx].append(U[name].reshape((-1,)))
            elif isinstance(prior.prior_base, PriorBase):
                pass
            else:
                raise TypeError("Expected PriorBase {}".format(type(prior.prior_base)))
        subspaces = tuple([jnp.concatenate(subspace_u) for subspace_u in subspaces])
        return subspaces

    def __call__(self, U_compact, **kwargs):
        """
        Transforms a compact representation of U to X domain.

        Args:
            U_compact: tuple of two subspaces, see compactify_U.
            **kwargs: kwargs to pass to all prior transforms.

        Returns: dict of X domain priors.
        """
        U = self.disperse_U(U_compact)
        # parents start out as place holder names.
        parents_dsk = {p.name: [parent.name for parent in p.parents] for p in self.prior_chain.values()}
        forward_topo_sort = iterative_topological_sort(parents_dsk)[::-1]
        for name in forward_topo_sort:
            prior = self.prior_chain[name]
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
        with disable_jit():
            keys = random.split(key, num_samples)
            for key in keys:
                U = self.compactify_U(self.sample_U(key))
                Y = self(U, **kwargs)
                for k, v in Y.items():
                    if jnp.any(jnp.isnan(v)):
                        raise ValueError('nan in prior transform {}'.format(Y))
                if log_likelihood is not None:
                    loglik = log_likelihood(**Y, **kwargs)
                    if jnp.isnan(loglik):
                        raise ValueError("Log likelihood is nan at {}".format(Y))
                    logger.info("Log-likelihood measured: {}".format(loglik))
