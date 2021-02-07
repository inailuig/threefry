#!/usr/bin/env ipython3

import jax
import jax.numpy as jnp
import numpy as np
from jax.random import threefry2x32_p

num = 2**26
k = jax.random.PRNGKey(123)

key1, key2 = k
x0 = jax.lax.iota(np.uint32, num).block_until_ready()
x1 = (x0+num).block_until_ready()

def bench():
    f = jax.jit(lambda k: jax.random.split(k, 2**26)) # force re-jit; needed cause we change the impl
    _ = f(k) # compile
    get_ipython().magic("timeit f(k).block_until_ready()")

def bench2():
    f = jax.jit(threefry2x32_p.bind)
    _ = f(key1, key2, x0, x1) # compile
    get_ipython().magic("timeit f(key1, key2, x0, x1)[0].block_until_ready()")


print("jax:")
bench()
bench2()


import threefry
print("\nthreefry:")
bench()
bench2()
