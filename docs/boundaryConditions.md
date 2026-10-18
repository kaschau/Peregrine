# Boundary conditions

A boundary is split between the grid and the case, by what each one owns.
The grid gives an interface a **name**, and for a periodic interface the
transform that reaches its partner: the shape of the grid, which does not
change from run to run. The case says what each name **is** and what it
**reads**, in the `bcvals` section of the [config](config.md#bcvals).
So one grid runs as a wall on one case and an inlet on the next without
being rewritten.

```yaml
bcvals:
  inlet:
    kind: sup-in-fa
    p: 101325.0
    u: 650.0
    v: 0.0
    w: 0.0
    T: 415.0
    CH4: 0.055        # an inlet carries the composition, by species name
  walls:
    kind: slp-adia-wall
```

Every name the grid carries needs an entry, and a name the config does not
know is refused when the solver is built, naming what it does know. A
kind that reads no values needs only its `kind`; one that reads values
is refused an entry that does not give every one of them, and a key the
kind does not read is left alone.

An entry's values are compiled in, so changing a value rebuilds that
entry's kernels, cached after.

## Where a name comes from

A grid file carries the name of every named interface in its
connectivity, put there by the mesh translator or by hand. A mesher names
the outside interfaces of its lattice by interface number,
`bcnames={3: "still", 4: "moving"}` for the low and high j
interfaces, with 1 and 2 the i and 5 and 6 the k, and
`defaultbc="wall"` names every outside interface not listed.

An interface with a neighbor is interior and one with a transform is
periodic; neither is a boundary. Every other interface is a boundary,
and has to be named for `bcvals` to say what it is: the grid carries no
kind of its own, and a solver refuses a boundary the grid left unnamed.

## The kinds

A kind is what a physics imposes behind the interfaces of one boundary
name, and a `kind` the physics does not declare is refused. An inlet also
reads a mass fraction for any carried species named in its entry, every
species but the last; a carried species not named is zero, and the last
species takes the remainder, a value given for it left alone.

Euler's kinds:

| kind | reads |
|---|---|
| `slp-adia-wall` | nothing |
| `slp-isot-wall` | `T`, held at the face, which a halo cell takes too |
| `sup-in-fa` | `p`, `u`, `v`, `w`, `T`, species: the entry's state whole |
| `sub-in-fvt` | `u`, `v`, `w`, `T`, species; the pressure extrapolated |
| `sub-in-ftptt` | `pt`, `Tt`, species: the halo's speed along the inward normal from the interior's outgoing Riemann invariant solved at the entry's total enthalpy, the eos at `Tt` over the entry's composition, and its static state by the isentropic relations; an interior whose invariant cannot reach that enthalpy, in a transient, takes the root the invariant allows |
| `sub-in-fmt` | `mflux`, `T`, species: the mass flux per unit area into the block, carried by the face's averages of density and of velocity; the pressure extrapolated |
| `sub-in-sem` | `T`, species, `frame`, `sem`: the velocity is sem's signal; the pressure extrapolated |
| `sub-out-fp` | `p`; the temperature and composition the first cell's, the velocity extrapolated |
| `sup-out-fn` | nothing: everything extrapolated, the pressure no higher than the interior's, p and T no lower than a hundredth of it |

Navier-Stokes keeps every one of these and adds the walls a viscous flow
has:

| kind | reads |
|---|---|
| `no-slp-adia-wall` | nothing: the flow sticks to the wall |
| `mov-adia-wall` | `u`, `v`, `w`, the wall's velocity, which the flow sticks to |
| `no-slp-isot-wall` | `T`, held at the face, the flow stuck |
| `mov-isot-wall` | `u`, `v`, `w`, `T` |

The double-flux physics have the same kinds: `euler-df` takes Euler's as
they are, and `navier-stokes-df` Navier-Stokes', its walls writing their
faces' energy row in both planes.

A viscous inlet or exit is Euler's kind of the same name, and sets the
halo's gradients layer for layer from its image's, every row copied. A
wall the flow slips along is Euler's slip wall, which mirrors its image's
momentum and copies its state; one the flow sticks to
reflects its image's whole velocity about the wall's, still or moving,
keeping the image's internal energy, or, holding a temperature, rebuilt
by the eos at the image's pressure and the wall's temperature, so the
first cell's gradient and a reconstruction at the wall see the wall's
velocity and temperature, and the halo is a state the eos has whatever
the flow against the wall.

A kind whose data fix the flux through its faces writes that flux itself,
in place of what the flux function made of the halo, so nothing crosses a
wall under any flux function or reconstruction. Every wall writes the
wall pressure on the momentum rows and nothing else advected: the inward
cell's pressure brought to rest against the wall, the acoustic estimate
p + rho c u_n with u_n the cell's speed toward the wall, the cell's own
pressure where the flow runs along the wall, more where it runs into it,
nothing where it runs away faster than p over rho c, a sound speed over
gamma in an ideal gas. A viscous wall adds the viscous rows from the
first cell inward: the stress of the velocity's normal derivative at the
wall, the cell's velocity brought to the wall's over its distance to the
face, every component where the flow sticks and the normal one alone
where it slips; the stress's work at the wall's velocity; and the
conduction of the two-point gradient from the cell's temperature to the
wall's where it holds one. The coefficients are the cell's at an
adiabatic wall and, at an isothermal one, the transport model's at the
wall's temperature with the cell's pressure and composition.

Velocities are in m/s, temperatures in K, pressures in Pa. Where a kind
extrapolates, halo layer l is the reflection about the first interior
cell of interior cell l + 1, linear through the first two cells. A block
thinner than a kind reaches gives its deepest cell instead, so a one-cell
slab between walls reflects its one cell into every layer. Where an
exit's first cell flows back in, its velocity is reflected about the face
layer for layer instead, like a slip wall's, so reverse flow cannot drag
the halo inside.

## Expressions

A value in an entry is a number or a string, an expression of the face
center's `x`, `y`, `z` and the clock `t`: `u: "10.0 + 5.0 * y"`,
`mflux: "0.116 * pow(max(1.0 - sqrt(y**2 + z**2) / 0.004, 0.0), 1.0 / 7.0)"`,
`T: "300.0 * min(t / 1e-3, 1.0)"`. A species' mass fraction may be one
too. The expression is parsed as Python and refused by name unless it is
numbers, those four names, `+ - * / **`, a unary sign and the functions
`sin cos tan exp log sqrt tanh abs min max pow`; it is evaluated at every
face, every step. So a profile over an inlet is written in the entry as
the function it is, and a ramp is an expression in `t`. A changed string
recompiles.

## Synthetic turbulence inlet

`sub-in-sem` is the kind that puts a turbulent signal from the
[synthetic eddy method](sem.md) on its interfaces, generated as the run goes. The
entry carries `T` and species as every inlet does, no velocity, a
`frame`, and a `sem` section handed to sem whole; what goes in it, the
flow and the signal's options, is
[sem's to say](sem.md#the-sem-section).

```yaml
bcvals:
  inlet:
    kind: sub-in-sem
    T: 162.7
    CH4: 1.0
    frame:                      # sem's (y, z) plane in grid coordinates
      origin: [0.0, 0.0, 0.0]
      ey: [0.0, 1.0, 0.0]
      ez: [0.0, 0.0, 1.0]       # eX = eY x eZ points into the domain
    sem:                        # handed to sem whole, see sem.md
      flow:
        type: channel
        y-dim: 0.1
        z-dim: 0.157
        periodic-z: true
        Uo: 4.52
        viscosity: 1.81e-5
        utau: 0.2126
      ensemble-time: 0.28
      nframes: 600
      seed: 10101
```

`frame` places sem's plane in grid coordinates: its `origin` and the grid
directions of its `y` and `z`, which have to be orthogonal; `eX = eY x eZ`
has to point into the domain, which is checked against the face normals.
The frame describes the whole inlet, not a block's piece of it, so it is
the same on every rank and is given explicitly, its origin where the
flow's shape puts it, a rectangle's corner or a pipe's axis.

The interfaces of one name on a rank share one signal over the centres of
the faces they own; that every rank's signal agrees with every other's,
and a restart's with the uninterrupted run's, is sem's
[determinism](sem.md#determinism-ranks-restarts).
