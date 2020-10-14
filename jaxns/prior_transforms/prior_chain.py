from collections import OrderedDict
from jax import vmap, random, numpy as jnp, disable_jit
from jaxns.utils import iterative_topological_sort


class PriorTransform(object):
    def __init__(self, name, U_ndims, parents, tracked):
        self._name = name
        self._U_ndims = U_ndims
        self._parents = list(parents)
        self._tracked = tracked

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, ",".join([p.name for p in self.parents]))

    @property
    def parents(self):
        return self._parents

    @property
    def name(self):
        return self._name

    @property
    def U_ndims(self):
        return self._U_ndims

    @property
    def tracked(self):
        return self._tracked

    @property
    def to_shape(self):
        """
        tuple providing the shape of the output.
        """
        raise NotImplementedError()

    def forward(self, U, *parents, **kwargs):
        """
        Transforms a vector of i.i.d. Uniform random variables to another random variable.
        The transformation may be dependent on any ancestor RVs.
        Args:
            U: [U_ndims] flat vector drawn from U[0,1]
            **ancestors: arbitrary dict of RVs early in the chain.
        Returns:
            jnp.ndarray with shape_dict given by to_shape
        """
        raise NotImplementedError()

    def __call__(self, U, *parents, **kwargs):
        """
        Calls the forward method and checks that names and shape_dict match to_shapes.
        """
        transformed_prior = self.forward(U, *parents, **kwargs)
        if transformed_prior.shape != self.to_shape:
            raise ValueError("Expected shape_dict {}, got {}.".format(self.to_shape, transformed_prior.shape))
        return transformed_prior


class PriorChain(object):
    def __init__(self):
        self.prior_chain = OrderedDict()

    def push(self, prior: PriorTransform):
        """
        Pushs a prior onto the chain that can be used by all descendents.
        """
        if not isinstance(prior, PriorTransform):
            raise ValueError("Trying to push {} onto chain".format(type(prior)))
        if prior.name in self.prior_chain.keys():
            print("Adding prior {} twice, watch out for duplicate named priors.".format(prior.name))
        self.prior_chain[prior.name] = prior
        for parent in prior.parents:
            self.push(parent)
        return self

    @property
    def U_ndims(self):
        return sum([p.U_ndims for p in self.prior_chain.values()])

    @property
    def to_shapes(self):
        shape = OrderedDict()
        for p in self.prior_chain.values():
            shape[p.name] = p.to_shape
        return shape

    def __repr__(self):
        dsk = {p.name: [parent.name for parent in p.parents] for p in self.prior_chain.values()}
        forward_topo_sort = iterative_topological_sort(dsk)[::-1]
        s = []
        for name in forward_topo_sort:
            s.append("{} ~ {} : {}".format(name, self.prior_chain[name].to_shape, self.prior_chain[name]))
        return "\n".join(s)

    def __call__(self, U, **kwargs):
        dsk = {p.name: [parent.name for parent in p.parents] for p in self.prior_chain.values()}
        forward_topo_sort = iterative_topological_sort(dsk)[::-1]
        idx = 0
        for name in forward_topo_sort:
            p = self.prior_chain[name]
            input = U[idx:idx + p.U_ndims]
            idx += p.U_ndims
            parents = [dsk[parent] for parent in dsk[name]]
            dsk[p.name] = p(input, *parents, **kwargs)
        transformed_prior = dsk
        return transformed_prior

    def test_prior(self,key, num_samples, log_likelihood=None, **kwargs):
        keys = random.split(key, num_samples)
        for key in keys:
            U = random.uniform(key, shape=(self.U_ndims,))
            Y = self(U, **kwargs)
            for k,v in Y.items():
                if jnp.any(jnp.isnan(v)):
                    raise ValueError('nan in prior transform',Y)
            if log_likelihood is not None:
                loglik = log_likelihood(**Y)
                if jnp.isnan(loglik):
                    raise ValueError("Log likelihood is nan", Y)
                if loglik == 0.:
                    print("Log likelihood is zero", loglik, Y)
                print(loglik)




