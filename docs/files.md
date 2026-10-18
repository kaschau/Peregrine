# Files

What PEREGRINE reads and writes: the grid, results, the config, and the
small input files some boundaries take. Grids and results are HDF5 with an
XDMF sidecar, so ParaView opens the `.xmf` directly. Both are written
collectively into one file, which needs a parallel h5py on one rank as much
as on many; see [installing](install.md#mpi-and-parallel-hdf5).

## The grid

`g.h5` carries the grid and everything about it that does not change with
the case: the coordinates of every block, how the blocks connect, and any
partitions the grid has been balanced into.

```
g.h5
  nblocks                                        attribute
  peregrine, commit, host, ranks, command, written   where it came from
  coordinates_000000/{x,y,z}                         one group per block, (nk, nj, ni)
  connectivity/{nbr,orient,bcname}                   (nblocks, 6)
  connectivity/{rot,xlate}   how a periodic moves
  partitions/1x1/rank                                the base grid
  partitions/64x4/rank                               which rank owns each block
  partitions/64x4/{cuts,connectivity}                a partition of pieces: which slab of which block, how they join
```

Coordinates are stored `(nk, nj, ni)`.
A block's six interfaces are numbered 1 to 6: the low and high ends of i,
then of j, then of k. The connectivity uses GridPro's orientation notation.
A periodic interface stores the transform that reaches its partner, a
rotation and a translation, because that is the shape of the grid.

What kind of boundary an interface is does not live here. The grid gives an
interface a name and the case says what the name means, in `bcvals`; see
[boundary conditions](boundaryConditions.md). An interface with a neighbor
is interior, one with a transform periodic; every other interface is a
boundary, and a boundary the grid did not name, or named for no entry of
`bcvals`, is refused when a solver is built.

`pg.writers.GridWriter(mb, fname).write(mb)` writes one from a multiblock.
`peregrine gridpro2pg` and `peregrine icem2pg` write grids from GridPro and
ICEM output, and condition them on the way: which interfaces are periodic is
read off the points, every interface that can be merged away is, every block
is re-indexed so its longest extent is i, and the interfaces left are matched
exactly.

### Partitions

A grid carries as many partitions side by side as it has been balanced for,
named `<ranks>x<ranks_per_node>`, so one grid runs on 64 ranks of 4 per node
or of 8 without being rebalanced. The two place blocks differently, because
what crosses a node costs more than what stays on one. A run picks the
partition for its rank count, and the one for its ranks per node if the
grid has it, else another layout of the same count. A grid with no partition
for the rank count refuses to run and says how to add one:

```
peregrine partition g.h5 -ranks 64 -ranks_per_node 4
```

Blocks are placed whole unless `-granularity` says to cut them down first,
to pieces no bigger than an even share of the cells divided by it: 1 cuts
only what cannot fit a rank, 2 aims for two pieces per rank. Placed whole,
the largest block sets a ceiling no placement can beat, and the tool reports
when that, rather than the grouping, is what binds. A partition of pieces
stores, beside the rank of each piece, the table of which slab of which
block each piece is and the pieces' own connectivity, and a run on it holds
the pieces as its blocks. `peregrine analyze` reports a grid's cell count,
its largest and smallest block, and the partitions it carries. The
partitioner is `auto` unless `-method` says `greedy` or `metis`.

What it balances is work, not cells: a block weighs its interior cells,
plus every halo cell behind its six interfaces at `-halocost` each,
wall or trade alike, plus every halo cell behind a traded interface at
`-tradecost` more for its pack, unpack and fill. The halo is `-ng` deep,
one unless said, two for a case with a switch or a MUSCL reconstruction;
the grid does not know the case's. The defaults come from per-rank device
times on the burner grid; a case whose ranks pace the step by block count
rather than by cells has its halos underweighted, and the two costs are
what a refit against its per-rank times moves.

## Results

A result is one pair of files, `<name>.h5` and `<name>.xmf`, named from the
step or the time it was written at, `q.<nrt>` unless the writer plugin says
otherwise. There is no distinction between a restart and a frame of an
animation: a run writes as many results as it is asked for, and any one of
them can be restarted from or animated through.

```
q.00000042.h5
  nrt, tme                                           when this is from
  privars, variables                                what a case starts from, what each block group holds
  grid                                               the grid file, relative to this one
  config                                             the case, as its yaml
  peregrine, commit, host, ranks, command, written   where it came from
  results_000000/<variable>                          one group per block, one dataset per export variable
```

The export variables are what the physics says a result holds: for a
fluid, `rho`, `p`, `u`, `v`, `w`, `T` and every species' mass fraction,
the last species included, with the velocity as a vector in the `.xmf`.
A result is read back primitive by name: a missing one takes the value
the physics starts it from, zero for a species, and a missing one with no
such value, a pressure, is refused by name. A tool reading a result takes the
config the result carries, or one it is given, and writes the vector alone,
since a result carries nothing a tool could derive. The primitive vector is
what a case starts from, so a result is a complete restart. Each variable is stored over
the cells of a block of the grid, not of the partition that wrote it, so
any partition, of blocks or of pieces, can read the result back. A result
carries no grid of its own; its `.xmf` points at the grid it was run on.

A result is the state a run starts from, and
`peregrine run -b openmp -r q.00000042.h5` takes the grid and the config
from it too.

## The config

In executable mode the config is a yaml file with the sections of the
[config reference](config.md), read over the defaults by
`pg.read_cfg(path)`. A result carries the config that wrote it, so a
restart needs no file. `pg.write_cfg(config, fname)` writes one out, and
`peregrine config peregrine.yaml` writes one of every default to start
from.
