import jax
import numpy as np
from jax import numpy as jnp

from jaxns.pytree import TreeField, pytree_ravel


def _make_tree_field() -> TreeField[dict[str, jax.Array]]:
    return TreeField({
        'a': jnp.asarray([1.0, 2.0]),
        'b': jnp.asarray([[3.0]])
    })


def test_treefield_helper_methods_return_wrapped_treefield():
    field = _make_tree_field()

    ones = field.ones_like()
    zeros = field.zeros_like()
    normal = field.random_normal_like(jax.random.PRNGKey(0))
    uniform = field.random_uniform_like(jax.random.PRNGKey(1))

    for result in (ones, zeros, normal, uniform):
        assert isinstance(result, TreeField)
        assert not isinstance(result.tree, TreeField)
        assert jax.tree.structure(result) == jax.tree.structure(field)

    np.testing.assert_allclose(ones.tree['a'], jnp.asarray([1.0, 1.0]))
    np.testing.assert_allclose(ones.tree['b'], jnp.asarray([[1.0]]))
    np.testing.assert_allclose(zeros.tree['a'], jnp.asarray([0.0, 0.0]))
    np.testing.assert_allclose(zeros.tree['b'], jnp.asarray([[0.0]]))

    for leaf in jax.tree.leaves(uniform.tree):
        assert jnp.all(leaf >= 0.0)
        assert jnp.all(leaf < 1.0)

    normal_shapes = [leaf.shape for leaf in jax.tree.leaves(normal.tree)]
    field_shapes = [leaf.shape for leaf in jax.tree.leaves(field.tree)]
    assert normal_shapes == field_shapes


def test_treefield_arithmetic_preserves_wrapper_contract():
    field = _make_tree_field()
    other = TreeField({
        'a': jnp.asarray([0.5, 1.5]),
        'b': jnp.asarray([[2.0]])
    })

    results = (
        field + 1.0,
        field - 1.0,
        5.0 - field,
        field * 2.0,
        field / 2.0,
        field + other,
        field - other,
        field * other,
        field / other,
        -field,
    )

    for result in results:
        assert isinstance(result, TreeField)
        assert not isinstance(result.tree, TreeField)
        assert jax.tree.structure(result) == jax.tree.structure(field)

    np.testing.assert_allclose((field + other).tree['a'], jnp.asarray([1.5, 3.5]))
    np.testing.assert_allclose((field - other).tree['b'], jnp.asarray([[1.0]]))
    np.testing.assert_allclose((field * 2.0).tree['a'], jnp.asarray([2.0, 4.0]))
    np.testing.assert_allclose((field / other).tree['b'], jnp.asarray([[1.5]]))
    np.testing.assert_allclose((-field).tree['a'], jnp.asarray([-1.0, -2.0]))


def test_treefield_matmul_matches_dense_matrix_product():
    field = _make_tree_field()
    flat_field, _ = pytree_ravel(field)
    flat_matrix = jnp.asarray([
        [1.0, 2.0, 3.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 1.0],
    ])

    matrix_tree = field.from_flat_matrix(flat_matrix)
    result = matrix_tree @ field

    assert isinstance(matrix_tree, TreeField)
    assert matrix_tree.batch_dim() == flat_matrix.shape[0]
    np.testing.assert_allclose(result, flat_matrix @ flat_field)
