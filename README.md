# PEREGRINE: Accessible, Performant, Portable Multiphysics CFD

<p align="center">
    <picture>
      <source media="(prefers-color-scheme: dark)" width="800" srcset="docs/images/pgSplashD.jpg">
      <source media="(prefers-color-scheme: light)" width="800" srcset="docs/images/pgSplashL.jpg">
      <img alt="peregrine logo" width="800" src="docs/images/pgSplashL.jpg">
    </picture>
</p>

## About

PEREGRINE is a multiblock, structured-grid, finite volume, 3D
multiphysics CFD solver: the Euler and Navier-Stokes equations of a
multi-species gas, with thermally perfect and real-gas equations of state,
kinetic-theory transport, and finite-rate chemistry from a Cantera
mechanism. The case is described and driven in
[Python](https://www.python.org) and runs on the host's CPUs with OpenMP,
on an NVIDIA GPU, or on an AMD GPU, so one source runs a case on a laptop
and, without a line changed, on a GPU supercomputer. PEREGRINE is parallel
across nodes by MPI.

## Installation

A C++ compiler with OpenMP (on macOS `brew install libomp`), then
`pip install -e .` from the checkout; an NVIDIA GPU adds the CUDA
toolkit, an AMD GPU ROCm. Every kernel a case needs is compiled at first
use into `~/.cache/peregrine`. See [installing](docs/install.md) for the
dependencies, parallel HDF5 and profiling.

## Documentation

Start at the [documentation index](docs/README.md): how to
[run a case](docs/running.md), the [config file](docs/config.md), the
[mixture](docs/mixture.md), [boundary conditions](docs/boundaryConditions.md),
[plugins](docs/plugins.md), and the [files](docs/files.md) PEREGRINE reads
and writes.

## Attribution

Please use the following BibTex to cite PEREGRINE in scientific writing:

```
@misc{PEREGRINE,
   author = {Kyle A. Schau},
   year = {2021},
   note = {https://github.com/kaschau/PEREGRINE},
   title = {PEREGRINE: Accessible, Performant, Portable Multiphysics CFD}
}
```

## License

PEREGRINE is released under the New BSD License (see the LICENSE file for details).
Documentation is made available under a Creative Commons Attribution 4.0
license (see <http://creativecommons.org/licenses/by/4.0/>).
