# flow-equations
Wegner's flow equation approach for interacting fermionic systems in one-dimension. Written in python; parallelized through `jax`.

# Installation
To use the `flowequations` package, it is required to install the following dependencies:

### Dependencies
- [numpy](https://pypi.org/project/numpy/)
- [dill](https://pypi.org/project/dill/)
- [jax](https://jax.readthedocs.io/en/latest/installation.html) ([homepage](https://github.com/google/jax))
- [diffrax](https://docs.kidger.site/diffrax/) ([homepage](https://docs.kidger.site/diffrax/))

### Notes on installation
Due to `jax` utilizing CUDA drivers, there might be some complications during installation. If at first you don't succeed with `jax`, please consider installing the CUDA drivers locally on your machine. This is what I did, since the pip installation never worked.

# TODO:
- Include references to thesis (first complete the thesis).

