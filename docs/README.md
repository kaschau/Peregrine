# PEREGRINE documentation

- [Installing](install.md): the host compiler, the CUDA toolkit or ROCm
  for a GPU, the Python dependencies, the kernel cache, MPI and parallel
  HDF5, profiling.
- [Running a case](running.md): a case in a script, executable mode with
  the `peregrine` command, naming the backend, starting from a state you
  make, the tools, the examples.
- [The config file](config.md): every section and key, what it does, and
  what is refused.
- [The mixture](mixture.md): where species come from, the equation of
  state, the transport, diffusion and mixing-rule models, the fits,
  chemistry, reading a mixture from a script.
- [Boundary conditions](boundaryConditions.md): how an interface gets its
  name, what each kind reads, expressions of position and time, the synthetic turbulence
  inlet.
- [The synthetic eddy method](sem.md): the turbulent inlet the `sub-in-sem`
  kind puts on an interface: how it works, the flows and their database,
  determinism across ranks and restarts.
- [Plugins](plugins.md): how often a plugin acts, reporting, writing
  results, checking for NaNs, tracing points, the viscous sponge, Catalyst.
- [Files](files.md): the grid, partitions, results, the config file.
- [Glossary](glossary.md): the words the code uses and what each means.
