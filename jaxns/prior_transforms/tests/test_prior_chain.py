from jax import random, numpy as jnp

from jaxns.prior_transforms import PriorChain, DeltaPrior, NormalPrior, UniformPrior, HalfLaplacePrior
from jaxns.prior_transforms.prior_chain import iterative_topological_sort


def test_prior_chain():
    # two subspaces: a -> b, c -> d
    with PriorChain() as chain1:
        a = DeltaPrior('a', 0.)
        b = NormalPrior('b', a, [1.])
        for name in chain1._prior_chain.keys():
            assert name in ['a', 'b', '_b_sigma']
    chain1.build()
    U_compact = chain1.sample_U_flat(random.PRNGKey(42))
    X = chain1(U_compact)
    assert jnp.size(X['b']) == 1
    assert jnp.shape(X['b']) == (1,)


def test_iterative_topological_sort():
    dsk = {'a': [],
           'b': ['a'],
           'c': ['a', 'b']}
    assert iterative_topological_sort(dsk, ['a', 'b', 'c']) == ['c', 'b', 'a']
    assert iterative_topological_sort(dsk) == ['c', 'b', 'a']
    dsk = {'a': [],
           'b': ['a', 'd'],
           'c': ['a', 'b']}
    # print(iterative_topological_sort(dsk, ['a', 'b', 'c']))
    assert iterative_topological_sort(dsk, ['a', 'b', 'c']) == ['c', 'b', 'a', 'd']
    assert iterative_topological_sort(dsk) == ['c', 'b', 'a', 'd']


def test_gh_20():
    with PriorChain() as prior_chain:
        prior_a = UniformPrior('a', -20., 20.)
        prior_b = UniformPrior('b', -20., 20.)
        prior_c = UniformPrior('c', 0, 40)
        prior_d = UniformPrior('d', -20., 20.)
        prior_sigma = HalfLaplacePrior('sigma', 1)

    #shouldn't need to be built
    assert prior_chain.U_ndims == 5 #should be 5

    prior_chain.build()
    assert prior_chain.U_ndims == 5  # should be 5