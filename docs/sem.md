# The synthetic eddy method

`sem` is the ensemble synthetic eddy method: the turbulent signal the
[`sub-in-sem`](boundaryConditions.md#synthetic-turbulence-inlet) kind
imposes, generated as the run advances.

Cite it as Schau, Johnson, Muller and Oefelein, "An ensemble Synthetic Eddy
Method for accurate treatment of inhomogeneous turbulence", Computers &
Fluids 248 (2022) 105671; the method it builds on is Jarrin's synthetic
eddy method.

## How it works

Eddies convect through the inlet cross-section along `x`, the time-like
direction: an eddy at `x` reaches the plane at `x / U`. Rather than a small
box of eddies convected past the inlet, the points of the inlet are carried
down a long box of static eddies, so a point's whole signal is the sum, at
each time, of the eddies whose footprint covers its line. Eddies are drawn
in slabs indexed by the time their centres cross, each slab from its own
generator seeded by `(seed, slab)`, so the field is a function of the seed
and the time alone, whatever points are asked for. An eddy's footprint is a
cosine blob: in each direction one cosine over the largest of its length
scales, the components with smaller scales riding on it with their own
cosine, and nothing beyond the largest scale.

The signal comes in consecutive ensembles, **chunks**, of `ensemble-time`
each, sampled at `nframes` frames. Over each chunk every point's raw sums
are whitened over the ensemble and given the Cholesky factor of the flow's
Reynolds stresses, so the target is met exactly, the mean profile is added,
and the frames are held with the cubic spline through them. Chunks
partition time at shared knots: each blends its first `nblend` frames
into the previous chunk's normalization, so the signal is continuous across
the boundary with only a small slope change. The next chunk is generated
ahead while the current one is in use, unless `prefetch` is off.

## The `sem` section

The whole of the signal comes from the `sem` section of the boundary's
`bcvals` entry.

- `flow` names a canonical flow by `type` and gives its shape
  and the case's scalars: `Uo`, `viscosity` and `utau`, the last two needed
  only by a flow with a wall, and `scale`, a multiplier on every
  length scale, which for the isotropic flow is the one scale itself. A
  flow is the cross-section and the data for it in one: it answers, for
  any point, how far the wall is, the coordinate its models are tabulated
  against, and which way the wall normal points, the frame the length
  scales and stresses live in; and it holds the mean profile, the eddy
  length scales and the Reynolds stresses built from its data file for the
  scalars. `delta`, the scale the tables are made dimensional with, is
  fixed by the shape except for a boundary layer, which takes it.

## The flows

| type | shape | delta | plane origin | data |
|---|---|---|---|---|
| `channel(y-dim, z-dim, periodic-z)` | walls at `y = 0` and `y-dim`, `z` in `[0, z-dim]` | half the height | the corner | the wall law, Jarrin's scales, Moser's stresses |
| `boundary-layer(y-dim, z-dim, periodic-z, delta)` | one wall at `y = 0`, eddies up to `delta`, the free stream above | given | the corner | the wall law, scales linear to `delta`, Spalart's stresses |
| `isotropic(y-dim, z-dim, periodic-y, periodic-z)` | no wall, periodic both ways unless said otherwise | none | the corner | `Uo`, one scale, the identity |
| `pipe(R)` | a circle about the origin, the normal inward | `R` | the axis | the channel's |
| `annulus(Ri, Ro)` | a ring about the origin, the normal outward | half the gap | the axis | the channel's |

- The rest of the section are the inlet's own options: `ensemble-time`,
  `nframes`, `seed`, `t0`, `nblend`, `prefetch`, `c-eddy`,
  `population`, `convect`, `interp`, `ufloor`, `tslab`,
  `max-eddies`; a keyword it does not take is refused. `nframes` wants
  the smallest eddy to span four frames. `population`
  is `pdf`, eddies placed by the local eddy size, or `random`, uniformly;
  `convect` is `uniform`, every eddy at `Uo`, or `local`, each at the mean
  velocity of its height.

## The flows database

A canonical flow with data of its own has one file in the database,
`<type>.yaml`, with the three families, `profile`, `sigmas` and `stats`:
each is either a law with its constants, the wall law of a channel or
boundary layer profile say, or a table against the wall coordinate, with
the source it came from in a comment beside it as the species library does.
Tables run from the wall outward; a flow with two walls, the channel and
the annulus, mirrors them about delta, a column odd in v changing sign, and
a flow with one does not. `coordinate` is a table's y: `ydelta` scales with
delta; `yplus` scales with nu/u_tau near the wall and blends to delta
scaling in the core, so a table at one Re_tau serves another. Columns are
normalized by u_tau, u_tau^2, delta or delta/u_tau as the family fixes. A
pipe or annulus names the channel's file as its `data`; a flow with data of
its own reads the file of its name.

## Determinism, ranks, restarts

Every rank builds the signal from identical `flow`, `ensemble-time`,
`nframes`, `seed`, `t0` and options, and its own points; the values at
shared points are bit-identical and no communication is needed. A restart
constructs with the configured `t0` and asks for the chunk at the restart
time; nothing is persisted, and the values match the uninterrupted run.
