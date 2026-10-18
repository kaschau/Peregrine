# The config file

A case is described by one config: in a script, a `pg.Config()` whose
entries you set; at the command line, a yaml file with the same sections,
read over the defaults by `pg.read_cfg`
(`peregrine run -b openmp case.yaml grid`), or the config a result
carries (`peregrine run -b openmp -r result`). Every section and key
below exists in the defaults, so a misspelled key is refused when it is
set, and a missing one takes its default; `bcvals` and `plugins` alone
take the case's own names. Where the case runs, its backend, is named
when it is run, `-b openmp`, and not in the config.
`peregrine config out.yaml` writes the defaults as a file to start from.

A complete file, with every default:

```yaml
simulation:
  simulator: navier-stokes   # euler, navier-stokes, euler-df or navier-stokes-df
  precision: double         # double or single
  niter: 1                  # steps a run takes
  controller: fixed         # fixed or cfl
  dt: 1.0e-3                # s: the fixed controller's step
  dt-max: 1.0e-3             # s: the cfl controller's ceiling
  cfl: 0.1               # the cfl controller's target

mixture:
  species: null             # a mechanism file, a list of library names, or a dict
  eos: cpg                  # cpg, tpg, real-gas or real-gas-df
  trans: null               # constant, kinetic-theory, chung or entropy-scaling
  diffusion: lewis          # lewis, binary or takahashi
  mixing-rule: wilke         # wilke or herning
  chemistry: null            # null or finite-rate: the rates of the mechanism's reactions
  T-range: null              # [Tlow, Thigh] every fit is made over
  refit-tol: 1.0e-3          # relative error a refit polynomial meets
  refit-max-deg: 6         # and the degree it may go to

chem-integration:
  kind: null              # null, coupled or sub-stepped
  max-substeps: 200
  bulk-substeps: 1
  stiff-bins: 8
  equilibrium-tolerance: 1.0e-7

ics:
  p: 101325.0
  u: 0.0
  v: 0.0
  w: 0.0
  T: 300.0
  Y: {}                     # mass fraction by species name

time-integration:
  integrator: tvd-rk3       # rk1, rk2, tvd-rk3, rk34 or rk4

adv-flux:
  formulation: single       # single, blended or correction
  primary: kepaec           # the flux function of the two sides of a face
  recon: constant           # how a side is made of the cells
  limiter: null             # muscl's
  secondary: null           # a second flux function blended in by the switch
  correction: null          # a dissipation added to the primary, weighed by the switch
  switch: null              # the face's two sensors
  values: {}                # the flux formulas' numbers, over their defaults

diff-flux:
  formula: alpha-damping    # the diffusive flux through a face, for a viscous case
  values: {}                # the formula's numbers, over its own defaults

backend-openmp:  {cxx: c++, tile-cells: 128, tile-elements: 1024, constform: literal}
backend-cuda:    {tile-cells: 128, tile-elements: 1024, launch-threads: 256, launch-waves: 1, constform: literal}
backend-hip:     {tile-cells: 128, tile-elements: 1024, launch-threads: 256, launch-waves: 2, constform: literal}

halo-exchange:
  kind: hoststaged          # hoststaged or device

bcvals: {}                # by boundary name; see boundaryConditions.md
plugins: {}                 # by plugin name; see plugins.md
```

## simulation

What is simulated, in what precision, for how long, and how each step is
sized.

| key | default | what it does |
|---|---|---|
| `simulator` | `navier-stokes` | The equations solved. `euler` is inviscid: no transport model, no gradients, no diffusive flux. `navier-stokes` adds the transport properties, the gradients of velocity, temperature and mass fractions, and the diffusive flux, and needs a transport model in `mixture.trans`. `euler-df` and `navier-stokes-df` are the same equations with the double flux of Ma, Lv and Ihme for a contact between fluids, written for the real gas: once a step each cell freezes the tangent plane of its internal energy in pressure, density and the partial densities of its species at its state, and its energy advances in that plane; the energy a face carries is written once in each cell's plane, and the pressure between freezes is the plane's. A contact of two fluids, or of one fluid at two temperatures, then keeps its pressure through the update, which a conservative energy flux cannot do for a real gas; the frozen system keeps the eos's speed of sound and its pressure response to heating, mixing and reaction. The price is that energy is not conserved: the two cells at a face take different energy fluxes, each in its own plane, and what the plane leaves off the eos at a step's end, second order in the state's change, is dropped; the error is confined to where the state changes fastest. A flux function without a double-flux form is refused. A diffusive flux adds its energy in both planes. A state just set is made consistent by the eos once, then frozen. The plane is `real-gas-df`'s; any other `eos` is refused. |
| `precision` | `double` | What every array and kernel value is: `double` or `single`. Results are written in the writer plugin's own precision, single unless it says otherwise. |
| `niter` | `1` | How many steps `run()` takes. A script stepping by hand ignores it. |
| `controller` | `fixed` | How each step is sized. `fixed` takes `dt` every step. `cfl` takes the largest step `cfl` allows, from the largest acoustic plus convective speed on any rank, a cell's being what a wave oblique to the grid sees: each axis's normal velocity over its cell length, summed, plus the speed of sound over the length 1 / sqrt(sum of 1 / length squared), an isotropic cell's over sqrt(3); capped at `dt-max`, and shortened to land on the next time a plugin acts at by its `dt-out`. |
| `dt` | `1e-3` | The step, in seconds, for the fixed controller. Cast to a float when the config is validated, so `1e-7` in yaml is fine. |
| `dt-max` | `1e-3` | The cfl controller's ceiling, in seconds. |
| `cfl` | `0.1` | The cfl controller's target CFL number: the combined acoustic and convective speed a wave oblique to the grid sees in a cell, times the step, a number of the state and the grid alone that means the same in one dimension as in three. What value holds is the flux's and the scheme's: `tvd-rk3` holds the two-point central flux to about 1.7, and a dissipative flux to less. |

## mixture

The gas: its species, and the models that give their properties. The
[mixture page](mixture.md) says what each choice needs of a species and
which choices go together.

| key | default | what it does |
|---|---|---|
| `species` | `None` | Where the species come from. A mechanism file, which also brings the reactions: Cantera yaml, as a path or the name of one shipped, or a reactions-only CHEMKIN file, its species completed from the library; a CHEMKIN file carrying THERMO or TRANSPORT data is refused, to be converted once with Cantera's `ck2yaml`. A list of names from the species library. A dict of name to data, which the library completes. The order is the species order everywhere, and the last species takes what the others leave. |
| `eos` | `cpg` | The equation of state: `cpg` calorically perfect, every species a constant cp with its enthalpy referenced as the others' are, `h = cp (T - 298.15 K) + hf`, so a species' `dHf298` counts; `tpg` thermally perfect; `real-gas` cubic with the thermally perfect gas as its ideal reference; `real-gas-df` the cubic with the tangent plane of its internal energy in pressure, density and the partial densities of its species, which the double-flux simulators freeze once a step. |
| `trans` | `None` | How momentum and heat diffuse: each species' viscosity and conductivity by `constant`, `kinetic-theory` (Chapman-Enskog from Lennard-Jones parameters), `chung` (kinetic theory times Chung's density enhancement) or `entropy-scaling` (kinetic theory plus the residual entropy scaling excess on the cubic's residual entropy). A viscous case picks one; an Euler case leaves it `None`. |
| `diffusion` | `lewis` | How species diffuse, only in a viscous case: `lewis` gives each species a diffusivity from the conductivity and a Lewis number, one unless the species states its own; `binary` fits every pair's coefficient from the collision integrals and needs `kinetic-theory` or a model built on it; `takahashi` is `binary` with Takahashi's high-pressure correction on every pair, and needs `real-gas` for the critical points. |
| `bath` | `None` | The species the mixture-averaged diffusion rule sums a coefficient's partners over, by name: every species when `None`, which costs a pair per species pair; a few bath species cost a pair per species per bath species, the partners the bath leaves out taken to diffuse as the bath does. Read by `binary` and `takahashi`. |
| `mixing-rule` | `wilke` | How the species' viscosities combine into the mixture's: `wilke` or `herning`. A model like the others, present only with a transport model. The conductivity is not the rule's: whatever the rule, it is the mean of the mole-weighted series and parallel sums of the species'. |
| `chemistry` | `None` | The kinetics of the reactions the mechanism carries: `finite-rate`, mass-action rates in the Arrhenius, three-body, Lindemann and Troe forms the mechanism states, the reverse rates from the species' entropy fits, so it needs `tpg` or a model built on it; `None` leaves the reactions unread. What a step does with the rates is the chemistry section's `kind`. |
| `T-range` | `None` | `[Tlow, Thigh]`, the temperature range every temperature-dependent property is fit over. Required whenever a property is fit: `tpg`, `real-gas`, `kinetic-theory`, `binary`, `finite-rate` and the models built on them. The source data is extended past its ends where the range goes further. |
| `refit-tol` | `1e-3` | The relative error each refit polynomial (in ln T) has to meet. |
| `refit-max-deg` | `6` | The degree a refit may go to: the lowest degree within the tolerance is taken, or the cap's. Seven terms is what the source data has and what a polynomial in ln T stays well conditioned at. |

## chem-integration

How the rates of the mixture's `chemistry` model enter a step. A reacting
kind needs that model, which brings the reactions the mechanism carries; a
species list or dict has none and cannot react. Rates are evaluated in log
space with floors and clamps that keep an absent species and a hot cell
finite in any precision.

| key | default | what it does |
|---|---|---|
| `kind` | `None` | `None` for no chemistry. `coupled` begins the right-hand side with the net production rates of the state. `sub-stepped` integrates the rates over the step in substeps sized on the spot, each taking the state as far as the fastest species allows before it leaves [0, 1], the temperature following at constant volume, and begins the right-hand side with the mean rate over the step; compacted: a bulk pass over every cell and a dense pass over the cells it left stiff. |
| `max-substeps` | `200` | `sub-stepped`: the most substeps one step may take; a cell that runs out carries the mean rate over the part of the step its substeps covered. |
| `bulk-substeps` | `1` | `sub-stepped`: the substeps the bulk pass gives every cell before listing the cells it leaves unfinished for the dense pass. On a device a warp of cells runs as long as its stiffest cell, so a few stiff cells bog down every warp they sit in; the bulk pass bounds what a warp pays, and the dense pass runs the stiff cells packed. One to `max-substeps`; equal to it, the one-pass form, every cell going to the cap in place. |
| `stiff-bins` | `8` | `sub-stepped`: the bins the listed cells are kept in, log-spaced between the bulk count and the cap by the substeps a cell is expected to need, the count it took in the last right-hand side where it was stiff then, the top bin for one just turned stiff, so a warp of the dense pass holds cells of about one cost. One to sixteen; one is one list. |
| `equilibrium-tolerance` | `1e-7` | `sub-stepped`: a cell at whose first substep no species would move past this mass fraction along its rates over what remains of the step, nor the temperature past a tenth of a kelvin, finishes in one move with every species relaxed at its own destruction frequency: a slow species moves along its rate, a fast one lands on its quasi-steady value, a fixed point stays put. Zero for no exit. A cell whose fast radicals the explicit substeps keep throwing off their quasi-steady values is not settled at a step's start and substeps as if there were no exit. |

## ics

The uniform state a case starts from when it does not restart. A script may
overwrite it afterwards with `set_prims`.

| key | default | what it does |
|---|---|---|
| `p` | `101325.0` | Pressure, Pa. |
| `u`, `v`, `w` | `0.0` | Velocity, m/s. |
| `T` | `300.0` | Temperature, K. |
| `Y` | `{}` | Mass fraction by species name. A species not named is zero, and the last species of the mixture takes the remainder. |

## time-integration

The integrator, and the settings that are its own.

| key | default | what it does |
|---|---|---|
| `integrator` | `tvd-rk3` | `rk1`, `rk2`, `tvd-rk3`, `rk34` and `rk4` are explicit Runge-Kutta schemes of that order. |

## adv-flux

The advective flux at every face, as one unit in the formulation the
section names: `single`, one flux function of the reconstructed sides;
`blended`, two, weighed by a switch; `correction`, one with a correction
added to it, weighed by a switch. A reconstruction makes a state either
side of the face from the cells it reads, the flux function makes the flux
of the two, and with a switch the face is weighed and a second flux
function blended in or a correction added.

| key | default | what it does |
|---|---|---|
| `formulation` | `single` | How the flux is put together: `single`, `blended` or `correction`. |
| `primary` | `kepaec` | The flux function of the two sides. `kepaec`, the two-point central scheme of Shima et al., kinetic energy and pressure equilibrium preserving and entropy consistent, on the two cells as they are; it carries the thermal part of the internal energy, `cv T`, as the harmonic mean of the two sides' and the rest of `e` as the arithmetic mean, so it is regular wherever `e` changes sign and keeps its two properties for a calorically perfect gas. `kepaec4`, kepaec to fourth order in Pirozzoli's form, 4/3 of the two-point flux of the near pair less 1/6 of each pair a cell apart, which keeps the two-point flux's properties; it reads two cells either side and takes the `constant` reconstruction only. Riemann solvers, which take any reconstruction: `rusanov`, the average of the two sides' fluxes less the largest wave speed times the jump; `ausmup`, Liou's, with its low-Mach cutoff; `hllc`, Toro's, the flux of whichever of the four states the face sits in by Davis's outer wave speeds and the contact's between them, each star state keeping its side's tangential velocity, so a contact at rest passes nothing and keeps its pressure. |
| `recon` | `constant` | How a side is made of the cells: `constant`, each side its cell as it is, first order; `muscl`, two cells either side, the pressure, temperature, velocity and mass fractions each limited in Sweby's form and the rest of the side from the equation of state at the face, so a side is a state of the eos whatever the limiters did. |
| `limiter` | `None` | `muscl`'s, each phi of the slope ratio in Sweby's form: `koren`, van Leer's kappa scheme at a third, third order for cell averages on a uniform line and second on a general grid, upwind-biased; and the symmetric ones, which treat the two slopes alike: `minmod`, the smaller slope or none where they disagree, the most dissipative; `mc`, the monotonized central, the generalized minmod at theta 2; `superbee`, the most compressive that stays TVD, which steepens a smooth profile toward a step; `vanleer`, the harmonic mean of the two slopes, smooth in the ratio. |
| `secondary` | `None` | Shock capturing by blending: a second flux function, typically `rusanov`, weighed into the primary, (1 - w) primary + w secondary, w the larger of the switch's sensors. Under `blended`, with a `switch`. |
| `correction` | `None` | Shock capturing by addition: a dissipation added to the whole primary after it is written, each of its rows weighed by the switch's sensor for it, so the primary is untouched where the switch is off. `matrix-dissipation`, the Roe-type matrix dissipation of Swanson and Turkel: each characteristic field of the jump across the face, taken in p, u, T and Y between the two sides, damped at its own wave speed -- the pressure, the normal velocity and the isentropic part of the temperature at the speed of sound, the rest of the temperature, the transverse velocity and the species at the flow speed -- with the matrix at the face's mean state and the eos's density derivatives and species enthalpies there; the pressure and the velocity are weighed by the shock sensor alone, the temperature and the species by the larger of the two sensors, so a composition gradient does not dissipate the shear that rides on it. It damps a central scheme's ringing where the switch fires and does not bound the temperature at a contact, the dispersion it damps being `kepaec`'s own transport of the entropy and species waves; a contact whose temperature must stay between its two fluids wants a reconstructed Riemann solver. Under the double flux the energy row is written once per side in that cell's plane, the other rows one flux the planes' derivatives make. Under `correction`, with a `switch`; it reads no `values` of its own. |
| `switch` | `None` | The face's two sensors, shock and contact, each clipped to one: `jameson`, the shock sensor the second difference of the pressure along the face normal, and the contact the larger of it and the jump across the face of gamma as rho c^2 over p, which changes across a contact between fluids or a steep temperature change in a real gas, and is what a conservative energy flux cannot carry through without a pressure error; the pressure sensor alone is `jameson` at `gamma-gain` 0. `ducros`, one sensor for both: Hendrickson, Kartha and Candler's form of Ducros's, the dilatation squared over the dilatation plus the vorticity squared, times 4/3 and clipped to one, so a plane compression reads one and a shear none, confined to shocks by a high-pass on the dilatation at `nu` times the flow speed over the cell's length, the shortest of its half-spans, and floored; at the cell either side of the face, the larger. It reads the velocity gradients, so it is the viscous flow's and is refused under `euler`. |
| `values` | `{}` | The flux formulas' numbers by name, over their defaults: `ausmup` a `cutoff-mach`, the Mach number its low-Mach dissipation is sized at wherever the flow is slower, the square root of a tenth; `jameson` a `p-gain` and a `gamma-gain`, one each, zero turning that sensor off; `ducros` a `nu`, the high-pass's cutoff as a fraction of the flow speed over the cell's length, a tenth, and a `floor`, the sensor's least value, none. A name no formula reads is ignored. |

## diff-flux

The diffusive flux at every face, for a viscous case; `euler` does not
read it.

| key | default | what it does |
|---|---|---|
| `formula` | `alpha-damping` | The diffusive flux through a face, added to the advective flux there. `alpha-damping`, Nishikawa's: each face gradient is the two sides' average, damped toward the difference of the values each side extrapolates to the face along the unit normal. |
| `values` | `{}` | The formula's numbers by name, over the defaults it holds: `alpha-damping` an `alpha`, the damping coefficient, one. A name no formula reads is ignored. |

## backend-openmp, backend-cuda, backend-hip

How the kernels are built and launched, one section per backend. A run
reads the section of the backend it is given, `peregrine run -b openmp`
or `pg.get_backend("openmp", config)`, and ignores the rest.

| key | default | what it does |
|---|---|---|
| `cxx` | `c++` (openmp) | The host's C++ compiler, which every kernel of the `openmp` backend is compiled with, a command on the path or a path to one; no OpenMP flag needs setting. The `cuda` backend compiles with NVRTC, `hip` with HIPRTC. |
| `tile-cells` | `128` | The cells of one block a tile is, for a launch over cells or faces: one thread's work on the host, one thread block's on a device. |
| `tile-elements` | `1024` | The elements a tile is for a launch over the elements of an allocation rather than its cells, a copy for one. |
| `launch-threads` | `256` (device) | The launch bound the device kernels are compiled with: threads per block. At least `tile-cells`. |
| `launch-waves` | `1` cuda, `2` hip | The floor of resident waves per multiprocessor the compiler is told to fit: the register budget it sizes a kernel for. `2` measured best on an MI100, where a floor of `4` spilled the viscous flux and transport kernels; `1` keeps nvcc's own budget. |
| `constform` | `literal` | Where the case's constants (species fits, reaction coefficients) sit in a kernel: `literal` writes them into the code, `memory` keeps them in the backend's constant memory, `__constant__` on a device, and the kernel reads them from there. `literal` measured 23 % faster per step on an H100: the values become immediates in the instructions, where `__constant__` reads miss the small constant cache. The `hip` backend compiles with `-mllvm -disable-machine-licm`, since LLVM otherwise hoists the register builds of a kernel's literals out of its loops and spills them. A device keeps 64 KB of `__constant__` per compiled unit, enough for every shipped mechanism. |

Every value but `cxx` and `constform` is a positive integer.

## halo-exchange

| key | default | what it does |
|---|---|---|
| `kind` | `hoststaged` | The kind of message a halo travels between ranks by. `hoststaged` copies through pinned host memory. `device` sends straight from device memory, which needs a GPU-aware MPI; the installation knows whether it has one, the solver cannot tell. |

## bcvals

One entry per boundary, by the name the grid gives its interfaces -- every
boundary interface has a name and every name an entry, or the solver refuses
it: what the boundary is, its `kind`, and every value that kind reads, each
a number or an expression of the face center's `x`, `y`, `z` and the clock
`t`. A key the kind does not read is left alone. See
[boundary conditions](boundaryConditions.md#expressions).

## plugins

What runs alongside the stepping, by plugin name, each with its own options
and how often it acts. A run prints nothing but what its plugins print, and
`report` is its progress. See [plugins](plugins.md).

## What is refused

Reading a config casts `dt` to a float. Everything else is refused when
the backend or the solver is made:

- a key no section has, `bcvals` and `plugins` excepted;
- a `simulator`, `controller`, `integrator`, `eos`, `trans`, `diffusion`,
  `mixing-rule`, `formulation` or `halo-exchange.kind` that is not one of the
  names above, and models that do not go together, as `binary` without
  `kinetic-theory` or `takahashi` without `real-gas`;
- a `navier-stokes` case with no transport model;
- a flux key that is none of the names above, or left null where the
  formulation reads it;
- `euler-df` or `navier-stokes-df` with an `eos` other than
  `real-gas-df`, or with a flux function that has no double-flux form;
- a `mixture.chemistry` that is not one of the names above, whose species
  bring no reactions, or with the calorically perfect gas;
- a `chem-integration.kind` that is not one of the names above, or set without
  `mixture.chemistry`; a sub-stepped count or tolerance outside its range;
- an initial `Y` naming a species the mixture does not have, or summing past
  one;
- a backend value other than `cxx` that is not a positive integer, or on
  a device a `tile-cells` above `launch-threads`; a `cxx` that does not
  compile OpenMP;
- a boundary interface the grid did not name, or named for no entry of
  `bcvals`; an entry whose `kind` is not one of the physics' kinds, or
  that does not give a value its kind reads; an expression naming
  anything a kernel cannot evaluate;
- a `plugins` entry named for no plugin, or for one no part of this case
  takes.
