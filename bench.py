#!/usr/bin/env ipython3

import jax
import jax.numpy as jnp
import numpy as np
from jax.random import threefry2x32_p

num = 128*8192*6
k = jax.random.PRNGKey(123)

key1, key2 = k
x0 = jax.lax.iota(np.uint32, num).block_until_ready()
x1 = (x0+num).block_until_ready()

def bench():
    f = jax.jit(lambda k: jax.random.split(k, num)) # force re-jit; needed cause we change the impl
    _ = f(k) # compile
    get_ipython().magic("timeit f(k).block_until_ready()")

def bench2():
    def _test(*args):
        a = threefry2x32_p.bind(*args)
        return a[0][-1]+a[1][-1]
    f = jax.jit(_test)
    _ = f(key1, key2, x0, x1) # compile
    get_ipython().magic("timeit f(key1, key2, x0, x1).block_until_ready()")

def bench3():
    # shape 2*num since we want to generate the same number of bits as the other benchs
    f = jax.jit(lambda k: jax.random.uniform(k, shape=(2*num,))) # force re-jit; needed cause we change the impl
    _ = f(k) # compile
    get_ipython().magic("timeit f(k).block_until_ready()")

def bench4():
    f = jax.jit(lambda k: jax.random.randint(k, shape=(2*num,), minval=0, maxval=256)) # force re-jit; needed cause we change the impl
    _ = f(k) # compile
    get_ipython().magic("timeit f(k).block_until_ready()")


print("jax:")
bench()
bench2()
bench3()
bench4()

import threefry
print("\nthreefry:")
bench()
bench2()
bench3()
bench4()
