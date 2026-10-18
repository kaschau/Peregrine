# Running a case

There are two ways to run PEREGRINE. A script builds the case in Python and
steps it, which suits small studies and anything that wants to look at the
state as it goes; [examples/](../examples) is a set of them. Executable mode
runs a case from a config file and a grid file, or from a result, with
the `peregrine` command, which is how production cases run.

## In a script

```python
from mpi4py import MPI  # first, so MPI is up before the backend
import peregrine as pg

# the case
config = pg.Config()
config["simulation"]["simulator"] = "navier-stokes"
config["simulation"]["dt"] = 1e-6
config["mixture"]["species"] = ["O2", "N2"]
config["mixture"]["eos"] = "tpg"
config["mixture"]["trans"] = "kinetic-theory"
config["mixture"]["T-range"] = [200.0, 1000.0]
config["bcvals"]["still"] = {"kind": "no-slp-adia-wall"}
config["bcvals"]["moving"] = {"kind": "mov-adia-wall", "u": 5.0, "v": 0.0, "w": 0.0}
config.validate()

# the mesh: a box of one block, its j interfaces named, periodic in i and k
mesh = pg.mesher.CubeMesher(
    mbdims=[1, 1, 1],
    blkdims=[2, 50, 2],
    lengths=[0.001, 0.025, 0.001],
    periodic=[True, False, True],
    bcnames={3: "still", 4: "moving"},
)

backend = pg.get_backend("openmp", config)
mb = pg.get_solver(config, mesh, backend)
print(mb)
for _ in range(1000):
    mb.step(config["simulation"]["dt"])
data = mb.sim.iovars.data(mb.blocks[0], ["u", "T"])
backend.stop()
```

What the pieces are:

- **The config** says everything about the case but its shape. Every key is
  in the [config reference](config.md). `validate()` casts the values
  as reading a file would.
- **The mesh** is a mesher, or a grid file's reader. `CubeMesher` cuts a
  box into a lattice of `mbdims` blocks of `blkdims` nodes each;
  `AnnulusMesher` does the same for a wedge of an annulus. `bcnames`
  names the outside interfaces by number, 1 and 2 the low and high i, 3 and 4
  j, 5 and 6 k, and `defaultbc` the ones not listed, for
  `bcvals` to say what each name is; a periodic axis has no outside on
  it, and a boundary left unnamed is refused.
- **The backend** is where the case runs, named here and never in the
  config: `pg.get_backend("openmp", config)`, or `cuda` or `hip`. It is
  made per case, from the config's precision and its `backend-<name>`
  section, and started on construction; `backend.stop()` at the end stops
  what it started. The `cuda` backend takes this rank's device, the ranks
  sharing a node taking its devices in turn; `hip` does the same. A script
  that only reads or writes files needs none.
- **The solver**, `pg.get_solver(config, mesh, backend, state=None)`, is the
  case ready to step. With no `state` it starts uniform at the config's
  initial conditions; with a `RestartReader` it starts from that result, at
  its step and time.

The solver is the one handle a script holds:

| | |
|---|---|
| `mb.step(dt)` | One step of `dt`. `mb.nrt` and `mb.tme` move on. `mb.tme` is the solver's clock and `mb.dt` the step being taken, read as `mb.tme.get()`. |
| `mb.run()` | `simulation.niter` steps sized by the controller, with every plugin acting as often as it says. |
| `mb.blocks`, `mb.get_blk(n)` | This rank's blocks; a block's arrays are attributes, `blk.Q.get()` a host copy of the solution over every cell, halos included. |
| `mb.sim.iovars.data(blk, names)` | Host arrays of the named variables of a block, as the physics derives them: `rho`, `p`, `u`, `v`, `w`, `T` and every species. |
| `mb.set_prims(primitives)` | Sets the state from one host array per block of the primitive vector, `p, u, v, w, T` and every species' mass fraction but the last's, over every cell, halos included. |
| `mb.rhs(Q="Q1", dQ="Q2")` | The right-hand side of one register of the solution into another, the registers named by `mb.regs`, `("Q", "Q1", ...)`, `Q` the solution between steps. Set the step with `mb.dt.set(dt)` first: what begins `dQ`, a chemistry's source, is sized by it. |
| `print(mb)` | The case as the report plugin prints it. |

The script runs under MPI like any other: `mpiexec -n 4 python -m mpi4py
case.py`. A mesher gives every rank the whole lattice, so a multi-rank
script wants a partitioned grid file instead, each rank taking its share:
`pg.readers.GridReader("g.h5", ranks=(size, ranks_per_node))`, with `size`
from `pg.misc.get_comm_rank_size()` and `ranks_per_node` from
`pg.misc.get_ranks_per_node()`. See [files](files.md#partitions).

## Executable mode

The `peregrine` command, which the install puts on the path, runs a case
from files; `python -m peregrine` is the same thing:

```
peregrine run -b openmp peregrine.yaml g.h5
peregrine run -b openmp -r q.00000100.h5
mpiexec -n 64 peregrine run -b cuda peregrine.yaml g.h5
```

`-b` (`--backend`) names where the case runs, `openmp`, `cuda` or `hip`,
and is required: the config does not say, and a result restarts wherever
it is told to. The first form takes a config and a grid, and starts from
the config's initial conditions. The second restarts from a result, which
carries the config it was written with and the grid it sits on; a config
or a grid given beside `-r` wins over the result's. The run takes
`simulation.niter` steps and prints nothing but what its plugins print;
the report plugin is its progress. A failure on any rank aborts the whole
run.

A grid run on more than one rank has to carry a partition for that rank
count, made by `peregrine partition`; the run says so, and how to make one,
if it does not. See [files](files.md#partitions).

Nothing depends on where the files sit: the config names the results
directory, and a result names its grid relative to itself.

## Starting from a state you make

To start a production case from a flow field built in Python, build the
case in a script, set the state, and write a result to restart from:

```python
backend = pg.get_backend("openmp", config)
mb = pg.get_solver(config, pg.readers.GridReader("g.h5"), backend)
# one array per block, shaped blk.Q.shape[:3] + (mb.nprims,): every cell, halos included
q = [cases_own_field(blk) for blk in mb.blocks]
mb.set_prims(q)
pg.writers.RestartWriter(mb, mb.sim.iovars, "q.{n:08d}.h5", "g.h5", precision="double", cfg=config).write(mb, mb.nrt, mb.tme.get())
```

Then `peregrine run -b openmp -r q.00000000.h5`. The result carries the config the
writer is given, which is what the restart runs with.

## The tools

The rest of the `peregrine` command works on a case's files. Every input
and output is a file named on the command line, and `peregrine <tool> -h`
says what each takes.

| | |
|---|---|
| `peregrine config peregrine.yaml` | Writes a config of every default to start a case from. |
| `peregrine partition g.h5 -ranks 64 -ranks_per_node 4` | Adds a load-balanced partition to a grid; see [files](files.md#partitions). |
| `peregrine verify g.h5` | Checks a grid's connectivity and that joined interfaces' coordinates match. |
| `peregrine analyze g.h5` | Reports the block sizes and the partitions a grid carries. |
| `peregrine interpolate q.h5 other.h5 out.h5` | Interpolates a result onto another grid. |
| `peregrine gridpro2pg blk.tmp blk.tmp.conn blk.tmp.pty g.h5` | Translates a GridPro grid. |
| `peregrine icem2pg info.topo g.h5` | Translates an ICEM Multi-Block Info grid. |
| `peregrine channel channel.yaml g.h5` | Makes a channel or boundary layer grid spaced in wall units; `-template` writes the yaml. |
| `peregrine rotate sector.h5 g.h5 -segments 8 -angle 45 -axis 1,0,0` | Copies a sector grid about its axis into more sectors. |

What the tools need beyond the solver, scipy, matplotlib and pymetis, is
the `tools` extra of the install.

## Examples

Every script in [examples/](../examples) runs from a checkout and takes
the backend's name first, `python examples/oneDAdvection.py openmp`:
shockCapture.py takes its exact Riemann solver from tests/riemann.py,
autoIgnition.py and detonationProfile.py point Cantera at the package's
mechanism database, and coProcess.py wants two ranks. They cover
one-dimensional advection and diffusion, a two-dimensional
Euler vortex, Taylor-Green on a straight and a skewed box, a normal shock,
shock capturing, Couette flow against its analytical solution, auto-ignition
and a detonation profile against Cantera, real-gas properties, Catalyst
co-processing, and an MPI scaling test.
