# AVX threefry kernels for jax (WIP)
- iirc roughly 1.7 times faster than the autovectorised ones in in the tf part of jaxlib (which already works amazingly well)
- achieved by upping ILP through manual unrolling
