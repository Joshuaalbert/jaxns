import jax.numpy as jnp
from jax import jit

def boxes_intersect_volume(a1, a2, b1, b2):
    return jnp.prod(jnp.maximum(jnp.minimum(a2,b2) - jnp.maximum(a1, b1), 0.))

def boxes_intersect(a1, a2, b1, b2):
    return jnp.all(jnp.minimum(a2, b2) - jnp.maximum(a1, b1) > 0.)

def cubes_intersect_volume(x,y,l):
    return jnp.prod(jnp.maximum(jnp.minimum(x, y) - jnp.maximum(x, y) + 2.*l , 0.))

def cubes_intersect(x,y,l):
    return jnp.all(jnp.minimum(x, y) - jnp.maximum(x, y) + 2.*l > 0.)

def test_cubes_intersect_volume():
    x = jnp.array([0., 0.])
    y = jnp.array([1., 1.])
    l = 1.
    assert cubes_intersect_volume(x,y,l) == boxes_intersect_volume(x-l,x+l,y-l, y+l)

    x = jnp.array([0., 0.])
    y = jnp.array([1., 1.])
    l = 1.5
    assert cubes_intersect_volume(x, y, l) == boxes_intersect_volume(x - l, x + l, y - l, y + l)


def test_boxes_intersect_volume():
    a1 = jnp.array([0., 0.])
    a2 = a1 + 1.
    b1 = jnp.array([1., 1.])
    b2= b1 + 1.
    assert boxes_intersect_volume(a1, a2, b1, b2) == 0.
    assert boxes_intersect_volume(a1, a2, b1, b2) == boxes_intersect_volume(b1, b2, a1, a2)

    a1 = jnp.array([0., 0.])
    a2 = a1 + 2.
    b1 = jnp.array([1., 1.])
    b2 = b1 + 1.
    assert boxes_intersect_volume(a1, a2, b1, b2) == 1.
    assert boxes_intersect_volume(a1, a2, b1, b2) == boxes_intersect_volume(b1, b2, a1, a2)

    a1 = jnp.array([0., 0.])
    a2 = a1 + 1.
    b1 = jnp.array([0.5, 0.5])
    b2 = b1 + 1.
    assert boxes_intersect_volume(a1, a2, b1, b2) == 0.25
    assert boxes_intersect_volume(a1, a2, b1, b2) == boxes_intersect_volume(b1, b2, a1, a2)

    a1 = jnp.array([0., 0.])
    a2 = a1 + 1.
    b1 = jnp.array([0.5, -0.5])
    b2 = b1 + 1.
    assert boxes_intersect_volume(a1, a2, b1, b2) == 0.25
    assert boxes_intersect_volume(a1, a2, b1, b2) == boxes_intersect_volume(b1, b2, a1, a2)

def point_in_cube(x, y, l):
    return (x < y + l) & (x > y - l)

