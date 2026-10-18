# The mixture

The gas a case solves: its species, and the models that give their
properties. The `mixture` section of the [config](config.md#mixture) names
them; this page says what each choice needs and which go together.

## Where the species come from

`mixture.species` is one of three things.

**A mechanism file.** Cantera yaml is the form to use: it carries the
species' thermodynamic and transport data beside the reactions, Cantera
itself validates it, and it is the form of the shipped mechanisms in
`mixture/database/mechanisms`.
Give a path, or the name of a shipped one. A CHEMKIN file, by
its suffix, `.inp`, `.dat`, `.ck` or `.mech`, is read for its reactions only: it is for a skeletal
mechanism distributed as reactions alone, whose species the library knows,
so it runs without conversion. A CHEMKIN mechanism that comes with its
THERMO or TRANSPORT data is refused: it belongs in Cantera yaml,
converted once with Cantera's `ck2yaml`, which checks it. Either
form is read without Cantera installed. The file brings the species in its
own order, whatever data it carries, and the reactions, which is the only
way a case gets chemistry; a species the file states no data for takes them
from the library. The CHEMKIN reader takes the reaction kinds PEREGRINE
has, Arrhenius, third-body and Lindemann and Troe falloff with DUPLICATE and
efficiencies, and refuses the rest by name: SRI, PLOG, explicit reverse
rates and the like.

```yaml
mixture:
  species: CH4_O2_FFCMY.yaml
  eos: tpg
  T-range: [300.0, 3500.0]
```

**A list of library names.** The species library, `species.yaml`,
carries curated, cited data for the species PEREGRINE knows, what is known
of each: thermodynamic data, a NIST-JANAF table or a NASA7 fit;
Lennard-Jones transport parameters; a critical point; and the residual
entropy scaling coefficients `resmu` and `reskappa`, which every entry
has. Name the ones the case has, in the order wanted.

Every entry says where each datum came from. The scaling coefficients have
one of three sources, named in the entry: the published fit on
Soave-Redlich-Kwong (Yang, Bell et al., ACS Omega 2025); a fit of the same
form made here against a fluid's reference transport surface in CoolProp,
where the paper has none; or, for a species no one has measured dense, the
paper's prediction, its group's parameters with the scaling factor at one.
[tools/entscale.py](../tools/entscale.py) makes every species' coefficients
from the paper's tables and CoolProp.

```yaml
mixture:
  species: [O2, N2]
  eos: tpg
  trans: kinetic-theory
  T-range: [200.0, 1000.0]
```

**A dict of name to data.** A gas stated in full, for one the library does
not carry: a single-component air, say. The data are whatever the models
need.

```python
config["mixture"]["species"] = {
    "Air": {"MW": 28.97, "cp0": 1002.8, "mu0": 1.859e-05, "kappa0": 0.02625}
}
```

Data are taken in this order of precedence: what the species come with,
the case's dict or the mechanism file, then the library. Nothing overrides
what came before it. A species that can answer for its composition
(`comp`, a dict of element to count) needs no molecular weight.

Everywhere in PEREGRINE the species are in this order, and the last one is
what the others leave: it is not a solved variable, and a result writes it
as one minus the rest.

## The models

Each model declares what it needs from every species, and the mixture
refuses a species that cannot answer, by name and quantity. Units are SI
with kmol: molecular weights in kg/kmol, energies in J/kmol.

### Equation of state, `eos`

| name | what it is | needs per species |
|---|---|---|
| `cpg` | Calorically perfect: one cp per species, so enthalpy and entropy are closed form. | `MW` or `comp`; `cp0` in J/kg/K; `dHf298` in J/kmol, zero when absent, which counts where species of different formation enthalpy mix (the case states it beside `cp0`, or a Cantera `constant-cp` entry's `h0` gives it). A chemistry kind refuses `cpg`, and the entropy `s298` feeds only the reverse reaction rates, so it is never used. |
| `tpg` | Thermally perfect: cp against temperature refit from the species' data. | `MW` or `comp`; thermodynamic data in one of the known formats: a JANAF table (`janaf`) or a NASA7 polynomial pair (`NASA7`). |
| `real-gas` | A cubic equation of state, Soave-Redlich-Kwong, with the thermally perfect gas as its ideal reference; a species may carry a constant volume translation, which shifts the volume the cubic gives and with it the density and the enthalpy, `cp` untouched. | Everything `tpg` needs, and a critical point: `Tcrit`, `pcrit`, `Vcrit`, `acentric`, or `well` and `diam` to derive one from; `vshift` in m³/kmol where a species states one, zero otherwise. |

### Transport, `trans`

Momentum and heat: each species' viscosity and conductivity. A viscous case
picks one; an Euler case has none.

| name | what it is | needs per species | goes with |
|---|---|---|---|
| `constant` | One viscosity and one conductivity per species, stated by the case. | `mu0` in Pa s, `kappa0` in W/m/K. | any eos |
| `kinetic-theory` | Chapman-Enskog from Lennard-Jones parameters, refit against temperature. | `well`, `diam`, `dipole`, `polarize`, `zrot`, `geometry` | `tpg` or `real-gas` |
| `chung` | Kinetic theory's dilute transport, mixed as `kinetic-theory` mixes it, times Chung's density enhancement: the ratio of Chung's dense to dilute value for the mixture as one nonpolar fluid, its size, well depth, weight and acentric factor made from the species' by Chung's mixing rules. Polar species are handled where kinetic theory handles them, in the collision integrals; Chung's own dipole terms are not applied, since under the one-fluid rule a few mole percent of water would swing the dense-fluid coefficients by tens of percent. | what `kinetic-theory` needs, and the critical point | `real-gas` |
| `entropy-scaling` | Kinetic theory's dilute transport, mixed as `kinetic-theory` mixes it, plus the residual of residual entropy scaling (Bell 2019; Yang, Bell et al. 2022-2025): made dimensionless by the number density and the thermal velocity, the residual viscosity and conductivity are functions of the cubic's residual entropy alone, shaped by each species' coefficients, which mix in mole fraction. The residual entropy is the case's own cubic's at the cell's state, so transport and thermodynamics are one model; no binary parameters. The coefficients are fitted on Soave-Redlich-Kwong. | what `kinetic-theory` needs, the critical point, and `resmu` (three coefficients) and `reskappa` (four) | `real-gas` |

### Species diffusion, `diffusion`

A separate choice from how momentum and heat diffuse, and only made in a
viscous case.

| name | what it is | needs per species | goes with |
|---|---|---|---|
| `lewis` | D = kappa / (rho cp Le), with a Lewis number of one unless the species states `lewis`. | nothing more | any transport model |
| `binary` | Every pair's diffusion coefficient from the collision integrals, refit against temperature, combined into each species' mixture coefficient by the mixture-averaged rule over the bath, the species the case names under `bath` or every species; a species alone in a cell, within rounding of the whole, or with none of its bath partners present, takes its self-diffusion coefficient. | `well`, `diam`, `dipole`, `polarize` | `kinetic-theory`, or a dense model built on it |
| `takahashi` | `binary` with Takahashi's (1975) high-pressure correction on every pair: (D p) over its dilute value as a function of the pair's reduced temperature and pressure, its critical point the two species' weighted by their mole fractions, as Cantera's high-pressure transport applies it. Tabulated to a reduced pressure of 5 and held at the table's edge beyond. | what `binary` needs, and the critical point | `real-gas`, with `kinetic-theory` or a dense model |

### Mixing rule, `mixing-rule`

How the species' viscosities combine into the mixture's: `wilke` or
`herning`. A model like the others, present only with a transport model; it
needs nothing of a species beyond its weight. The conductivity is not the
rule's: whatever the rule, it is the mean of the mole-weighted series and
parallel sums of the species'.

### Chemistry, `chemistry`

The rates of the reactions the mechanism carries. A model like the others,
given the reactions beside the species; `None` leaves the reactions unread.

| name | what it is | needs per species | goes with |
|---|---|---|---|
| `finite-rate` | Mass-action kinetics in the forms the mechanism states, elementary, three-body, Lindemann and Troe falloff: Arrhenius rates with third-body efficiencies and falloff blends, the equilibrium constants from the Gibbs energies, every reaction's rate folded into each species' net production rate. | nothing beyond the thermodynamic data `tpg` reads | `tpg` or `real-gas` |

## The fits

Every temperature-dependent property the kernels read is a polynomial in
ln T over `T-range`, refit from the species' data. Each refit takes the lowest
degree whose relative error is within `refit-tol`, or the best it can do at
`refit-max-deg`. The range is required whenever a fit is made, which is for
`tpg`, `real-gas`, `kinetic-theory` with the dense models built on it, and
`binary`; a calorically perfect gas with constant transport properties has
no fit and needs none. Where the range
reaches past the source data, the data is extended.

A kernel evaluates a fit at whatever temperature the cell has; nothing
clips to the range.

## Chemistry

Reactions come from the mechanism; the mixture's `chemistry` model gives
their rates, and the [chemistry section](config.md#chemistry) of the config
chooses how the source enters the right-hand side. A mixture given as a list
or a dict has no reactions, and a chemistry model on it is refused.

## Reading the mixture from a script

The mixture a case built is on the solver, and can be made on its own from a
config for a look at what it holds:

```python
mb = pg.get_solver(config, mesh, pg.get_backend("openmp", config))
m = mb.sim.mixture
m.spnames, m.ns, m.eos.name, m.species["O2"]["MW"]

from peregrine.simulator.mixture import get_mixture
m = get_mixture(config)
```
