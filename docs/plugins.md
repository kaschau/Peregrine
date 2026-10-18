# Plugins

A plugin is something a run does alongside the stepping: writing results,
printing, checking the state, sampling points, or adding to a physics'
right-hand side. The `plugins` section of the [config](config.md#plugins)
names the ones a run has, each with its own options, a solver's with how
often it acts. A run with no plugins steps in silence and writes nothing.
The solver's plugins act after a step; a physics' acts with the physics,
and only for the physics it is valid for.

```yaml
plugins:
  report:
    niter-out: 10
  writer:
    dt-out: 1.0e-4
    dir: Results
    precision: double
  nancheck: {}
```

An entry of the `plugins` section that no part of the case takes is
refused by name, and a plugin named for a physics it is not valid for, a
sponge on an Euler case, is refused saying which it is for.

## How often

Every solver plugin takes one of two keys, and neither given means every
step. A trace may name its own, so one section can trace at several
cadences. A physics' plugin takes neither: it acts every time the physics
does.

| key | what it does |
|---|---|
| `niter-out` | Act after every so many steps. |
| `dt-out` | Act at every multiple of this many seconds. The cfl controller shortens a step to land on the multiple; with a fixed step the plugin acts on the first step past it. |

## report

The run's progress as it steps; nothing else prints it.

At the start: the banner and the case, as `print(mb)` would show it. After
every step it is due on: the step number, the time, the step size, and the
largest CFL numbers on any rank, acoustic, convective and combined, each
as a wave oblique to the grid sees a cell (see
[`cfl`](config.md#simulation)). At the end: the wall time, the steps
this run took, which a restart counts from where it began rather than from
the case's first step, and the seconds per step per cell over every rank's
cells. No options.

## writer

Results into a directory, as numbered pairs of `.h5` and `.xmf` files. See
[files](files.md#results) for what a result holds.

| key | default | what it does |
|---|---|---|
| `dir` | `.` | The directory, made if need be. |
| `precision` | `single` | What the file holds: `single` or `double`, whatever the case computes in. |
| `basename` | `q.{n:08d}` | The file name, formatted from the step `n` and the time `t`. |

The `.xmf` points at the grid file the case came from. A case meshed in a
script has no grid file, so one is written into the directory first. The
solution is all an integrator carries between steps, so a result is a
complete restart.

## nancheck

Stops the run on a non-finite value of the solution. Every rank writes
where its non-finite values are to `nans_<block>.log`, the state is
written if there is a writer, and the run raises. No options.

## trace

The primitive variables at the cells nearest chosen points, appended to one
csv per trace each time it acts. A trace is a point, a line of `n` points
from `p0` to `p1`, or a plane of `n01` by `n02` points spanned from `p0` by
`p1` and `p2`, and names the file it writes. Each row is the time, the cell
center, and the primitive variables there; the header names them.

```yaml
plugins:
  trace:
    niter-out: 10
    traces:
      probe:
        type: point
        p0: [0.5, 0.0, 0.0]
        file: probe.csv
        niter-out: 1           # its own cadence; the others act on the section's
      centerline:
        type: line
        p0: [0.0, 0.0, 0.0]
        p1: [1.0, 0.0, 0.0]
        n: 50
        file: centerline.csv
      inlet:
        type: plane
        p0: [0.0, 0.0, 0.0]
        p1: [0.0, 1.0, 0.0]
        p2: [0.0, 0.0, 1.0]
        n01: 20
        n02: 20
        file: inlet.csv
```

A point is traced at the cell whose center is nearest it, on whichever rank
holds that cell, so a point outside the grid lands on the nearest cell there
is. A file that exists is appended to, so a restart continues it.

## viscsponge

Raises the viscosity along a line, from `origin` to `ending`, to
`multiplier` times itself, before every right-hand side, once the
transport properties are made. A viscous case only.

| key | what it does |
|---|---|
| `origin` | `[x, y, z]` where the ramp starts. |
| `ending` | `[x, y, z]` where it reaches the full multiplier. |
| `multiplier` | What the viscosity is multiplied by at the end. |

## catalyst

ParaView Catalyst in situ processing, driven by a pipeline script. ParaView
has to be importable in the case's Python; only a case that asks for this
plugin imports it.

| key | what it does |
|---|---|
| `script` | The Catalyst pipeline script. |
