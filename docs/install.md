# Installing

PEREGRINE has no build of its own. Every kernel a case needs is compiled
at first use, by the toolchain of the backend the case is run on, and
cached. So installing for the host is: a C++ compiler with OpenMP, a
Python with the dependencies, and `pip install -e .`; an NVIDIA GPU adds
the CUDA toolkit, an AMD GPU ROCm.

## The host compiler

The `openmp` backend compiles every kernel with the host's C++ compiler,
the one the config's `backend-openmp.cxx` names, `c++` unless it says
otherwise, as C++20 with OpenMP; on macOS that is Apple clang with
Homebrew's libomp, so there `brew install libomp`.

## CUDA

The `cuda` backend compiles every kernel with NVRTC, `libnvrtc`, as C++20
for the compute capability of the device this rank takes, and runs it
through the driver, `libcuda`; both are loaded by name, from
`$PEREGRINE_LIBRARY_PATH`'s directories or the system's search, or from
`$PEREGRINE_NVRTC_LIBRARY_PATH` and `$PEREGRINE_CUDA_LIBRARY_PATH` when
set. The ranks sharing a node take its devices in turn by their rank among
themselves. A unit NVRTC refuses goes to `nvcc` when one is on the path.

## HIP

The `hip` backend compiles every kernel with HIPRTC, `libhiprtc`, as C++20
for the architecture of the device this rank takes, and runs it through
the HIP runtime, `libamdhip64`; both are loaded by name as the CUDA
backend's are, `$PEREGRINE_HIPRTC_LIBRARY_PATH` and
`$PEREGRINE_AMDHIP64_LIBRARY_PATH` overriding. A unit HIPRTC refuses goes
to `hipcc` when one is on the path.

## Python

Python 3.10 or later. From the checkout, an editable install brings the
dependencies (`numpy`, `mpi4py`, `h5py`, `pyyaml`, `lxml`, `Mako`) and puts
the `peregrine` command on the path:

```
pip install -e .
```

An editable install and a plain one run the same; editable is the form for
a checkout that is edited. `mpi4py` builds against the MPI on the path, so
on a cluster load the MPI module first. Extras: `pip install -e ".[tools]"`
adds what the grid and interpolation tools need, scipy, matplotlib and
pymetis; `".[test]"` pytest, Cantera and CoolProp; `".[examples]"`
matplotlib and Cantera. Cantera is not needed to read a mechanism; ParaView
is imported only by a case that asks for the Catalyst plugin. Without
network access pip cannot fetch the build backend, so use the setuptools
already there: `pip install -e . --no-build-isolation`.

## The kernel cache

Compiled libraries go to `~/.cache/peregrine`, or `$PEREGRINE_CACHE`. The
cache is safe to delete; the next run rebuilds what it needs. A unit that
did not compile leaves its source in the cache, where the compiler's
errors point.

A cold cache compiles every kernel of a case over again, so on a cluster
keep it between jobs. With `PEREGRINE_JIT_DEBUG` set, every library is
built with debug information and the kernels' asserts in place; a debug
library sits beside the plain one.

## MPI and parallel HDF5

Every rank's h5py, mpi4py and HDF5 have to be built against the same MPI.
Grids and results are written collectively into one file, by the grid
tools and by a run alike, so a serial h5py cannot write one even on a
single rank, and two MPIs in one process will crash it; the readers open
a file plainly, so a serial h5py can still read one. A parallel h5py is
built from source against a parallel HDF5:

```
export CC=mpicc
export HDF5_MPI="ON"
export HDF5_DIR="/path/to/parallel/hdf5"   # if it is not found by default
pip install h5py --no-binary=h5py
```

To check which MPI each is using:

```
python -c "import h5py; print(h5py.get_config().mpi)"          # must be True
python -c "from mpi4py import MPI; print(MPI.get_vendor())"
ldd $(python -c "import h5py,os;print(os.path.dirname(h5py.__file__))")/*.so | grep libmpi
```

`libmpi.12` is the MPICH ABI soname and `libmpi.40` is Open MPI's; if h5py
and mpi4py name different ones, rebuild h5py against the MPI mpi4py uses.
On macOS read the libraries with `otool -L` instead of `ldd`.

A halo exchange straight from device memory (`halo-exchange.kind: device`)
needs an MPI that is GPU-aware, which is the installation's to know; the
default, `hoststaged`, stages messages through pinned host memory and
works with any MPI.

## Profiling

On the `cuda` and `hip` backends every kernel's entry point is named
for the kernel, `pg_<name>`, so Nsight Systems and Compute, and rocprof,
see each by name as they are:

```
nsys profile mpiexec -n 1 peregrine run -b cuda peregrine.yaml g.h5
rocprofv3 --kernel-trace -- peregrine run -b hip peregrine.yaml g.h5
```

Every kernel is compiled for the case that runs it, so a profile is of
that case's species count, halo depth and models, not a generic build.
