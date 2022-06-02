from typing import Union, Callable
from jax import random, numpy as jnp, disable_jit
from jax.flatten_util import ravel_pytree
from jaxns.internals.maps import prepare_func_args
from jaxns.prior_transforms import Prior
from jaxns.prior_transforms.context import _PRIOR_CHAINS, _PRIOR_CHAIN_NEXT_INDEX, _PRIOR_CHAIN_INDEX_STACK
import logging

logger = logging.getLogger(__name__)




def iterative_topological_sort(graph, start=None):
    """
    Get Depth-first topology.

    :param graph: dependency dict (like a dask)
        {'a':['b','c'],
        'c':['b'],
        'b':[]}
    :param start: str
        the node you want to search from.
        This is equivalent to the node you want to compute.
    :return: list of str
        The order get from `start` to all ancestors in DFS.
    """
    seen = set()
    stack = []  # path variable is gone, stack and order are new
    order = []  # order will be in reverse order at first
    if start is None:
        start = list(graph.keys())
    if not isinstance(start, (list, tuple)):
        start = [start]
    q = start
    while q:
        v = q.pop()
        if not isinstance(v, str):
            raise ValueError("Key {} is not a str".format(v))
        if v not in seen:
            seen.add(v)  # no need to append to path any more
            if v not in graph.keys():
                graph[v] = []
            q.extend(graph[v])

            while stack and v not in graph[stack[-1]]:  # new stuff here!
                order.append(stack.pop())
            stack.append(v)

    return stack + order[::-1]  # new return value!

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
        self._U_flat_placeholder: jnp.ndarray = None
        self._unravel_U_flat: Callable = None

        for prior in priors:
            self.push(prior)

    def __enter__(self):
        global _PRIOR_CHAIN_INDEX_STACK
        _PRIOR_CHAIN_INDEX_STACK.append(self._prior_chain_index)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _PRIOR_CHAIN_INDEX_STACK
        _PRIOR_CHAIN_INDEX_STACK.pop()

    def filter_sample(self, sample):
        """
        Filters a dict's keys to only those where prior variable of same name is tracked.
        Used for removing untracked priors from a dict.

        Args:
            sample: dict

        Returns:
            dict with only keys that correspond to names being tracked.
        """
        assert self.built
        return dict(filter(lambda item: self._prior_chain[item[0]].tracked, sample.items()))

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
        Dict of shapes of prior in U domain.
        """
        shapes = {name: p.shape for name, p in self._prior_chain.items()}
        return self._order(shapes)

    @property
    def dtypes(self):
        """
        Dict of dtypes of prior in U domain.
        """
        dtypes = {name: p.dtype for name, p in self._prior_chain.items()}
        return self._order(dtypes)

    def __repr__(self):
        parents_dsk = {p.name: [parent.name for parent in p.parents] for p in self._prior_chain.values()}
        forward_topo_sort = iterative_topological_sort(parents_dsk)[::-1]
        s = []
        for name in forward_topo_sort:
            s.append(
                f"{'tracked ' if self._prior_chain[name].tracked else '        '}{name} ~ {self._prior_chain[name].dtype}{self._prior_chain[name].shape} : {self._prior_chain[name]}")
        return "\n".join(s)

    def log_homogeneous_measure(self, **X):
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
        Transforms a compact representation of prior bases samples in U domain to U domain.

        Args:
            U_compact: flat vector of prior bases samples
            **kwargs: kwargs to pass to all prior transforms.

        Returns: dict of U domain priors.
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
        log_likelihood = prepare_func_args(log_likelihood)
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
