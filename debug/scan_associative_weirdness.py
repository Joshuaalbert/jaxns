import jax
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from jax import numpy as jnp


def explicit_verify_associative(op, elems):
    output_1 = op(op(elems[0], elems[1]), elems[2])
    output_2 = op(elems[0], op(elems[1], elems[2]))
    print(output_1, output_2)
    assert output_1 == output_2


def main():
    elems = jax.random.normal(jax.random.PRNGKey(0), shape=(3,))

    elem_shape = jax.tree.map(lambda x: np.shape(x[0]), elems)  # ()

    def per_elem_op(x) -> jax.Array:
        return jnp.sum(x)

    def associative_op(x, y):
        print(f"x.shape={np.shape(x)}, y.shape={np.shape(y)}")
        assert np.shape(x) == elem_shape
        assert np.shape(y) == elem_shape
        return per_elem_op(x) + per_elem_op(y)

    explicit_verify_associative(associative_op, elems)

    _ = tfp.math.scan_associative(associative_op, elems)


if __name__ == '__main__':
    main()
