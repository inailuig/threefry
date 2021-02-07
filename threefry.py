import jax
from jax import lax
import jax.numpy as jnp
from jaxlib import xla_client
from jax.interpreters import xla
import jax.random
from jax.random import threefry2x32_p
import numpy as np

# TODO
import sys
sys.path.append("./build")
import threefry_avx

for _name, _value in threefry_avx.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="cpu")


def threefry2x32(c, keys, data):
    assert len(keys) == 2, keys
    assert len(data) == 2, data

    dims = c.get_shape(data[0]).dimensions()
    dimsk = c.get_shape(keys[0]).dimensions()
    dtype = np.dtype(np.uint32)

    ndims = len(dims)
    size = np.prod(dims).astype(np.int64)
    layout = tuple(range(ndims - 1, -1, -1))
    layoutk = tuple(range(len(dimsk) - 1, -1, -1))

    shapekey = xla_client.Shape.array_shape(dtype, dimsk, layoutk)
    shapedata = xla_client.Shape.array_shape(dtype, dims, layout)

    shape_n = xla_client.Shape.array_shape(np.dtype(np.int64), (), ())
    n = xla_client.ops.ConstantLiteral(c, size)

    return xla_client.ops.CustomCallWithLayout(
      c,
      b"threefry2x32",
      operands=(n, keys[0], keys[1], data[0], data[1]),
      operand_shapes_with_layout=(shape_n,)+(shapekey,) * 2+(shapedata,)*2,
      shape_with_layout=xla_client.Shape.tuple_shape((shapedata,)*2)
    )


def _threefry2x32_translation_rule(c, k1, k2, x1, x2):
    shape = lax.broadcast_shapes(c.get_shape(k1).dimensions(), c.get_shape(k2).dimensions(), c.get_shape(x1).dimensions(), c.get_shape(x2).dimensions())
    rank = len(shape)

    def _broadcast(x):
        ndims = c.get_shape(x).rank()
        return xla_client.ops.BroadcastInDim(x, shape, tuple(range(rank - ndims, rank)))

    return threefry2x32(c, (k1, k2), (_broadcast(x1), _broadcast(x2)))


xla.backend_specific_translations['cpu'][threefry2x32_p] = _threefry2x32_translation_rule
