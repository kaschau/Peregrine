# To do

The running list for the `makoTransition` branch, by kind: what is wrong,
what is slow, what is less exact than it should be, what is missing, and
what is untidy. One line per item; an item done or wrong is deleted, not
kept. Nothing here is reference: the docs under `docs/` describe what is.

## Bug Fixes

What is wrong today, in a run or a tool.

- [ ] `GridWriter._write_conn` builds the connectivity tables from the blocks this rank holds and `_write_table` has rank 0 supply the whole table, so a grid written from ranks that hold different blocks carries only rank 0's rows, the names sized to rank 0's longest (`astype("S")` per rank), the rest `""` and -1; a one-rank write or ranks holding the whole grid are right. The write per block, as the coordinates go, or an allgather of the rows before the collective, fixes it; `examples/coProcess.py` meshes per rank and so cannot write the grid file `docs/running.md` prescribes for a multi-rank script.

## Performance

Work done for nothing, or more than once.

- [ ] `in` views are reloaded after every store to an `out` view: the compiler cannot know `F` does not alias `S` or `Q`, so the flux kernels read `S` again for each momentum row and `Q[0]` again for the species rows, a third of kepaec.mako's loads, all L1 hits. `in`/`out` make a view const and pin its pointer per team; const is not non-aliasing. `__restrict__` on the view members changes nothing under clang, which honors it on function parameters only. Two ways that work: the body reads before it writes, `S` and the densities into locals as `muscl.mako` already does; or the render emits the body as an inlined function whose view pointers are `__restrict__` parameters, which LLVM turns into alias scopes at the inlining, so nvcc and hipcc get it too.
- [ ] Single precision takes the library's log where double takes the layer's table log: `backend/base/kernels/mathfns.mako`'s `tlog` branches on `finfo.bits` because its body is double's bit layout (52-bit mantissa, 11-bit exponent, `long long`, double tables), so no single-precision table log has been written or timed. A float body is its own: 23-bit mantissa, 8-bit exponent and bias, bins from the top seven mantissa bits, float tables, a three-term series since `u` below 1/256 reaches float precision there. Whether it pays is open, `logf` being far shorter than `log` on the GPU; measure it on the probe harness the double form went through, and against the chemistry benchmark in single. `tests/chemistry/test_rates.py::test_single_follows_double` holds the fallback right.

- [ ] The rate evaluation by a vector of lanes a cell (`tests/chemistry/kernels/lanes.mako` on `chemistry/kernels/lanerates.mako`, the `lanes` policy, team scratch and its prologue, integer tables) runs 1.8 times slower than the thread form at 8 lanes on Zhukov and worse at 16 and 32, with the tables through L1 or in team scratch alike: the data-driven form makes every operand a load, and under Kokkos each is a generic-pointer load with 64-bit address arithmetic, the loops carry the vector-range bookkeeping and the fold is padded, 770 warp instructions a cell against 234. Its case returns as a native kernel once Kokkos is out, `__shared__` with 32-bit indices, species by shuffle, records packed into wide loads, the fold in registers, which should shed most of the tax; and for mechanisms the size of GRI-Mech 3.0, where the thread form's 255 registers spill and the lanes' trips fill. Measure both there before judging the layout again. CHANGES.md of the bo-prz chemistry benchmark holds the numbers and profiles.
## Accuracy

Where the numbers are less exact than they should be, and the data and design calls behind them.

- [ ] The inlets deliver their flux at the `postflux` hook as the walls do: each kind's kernel over the faces on its interfaces builds the face state from the entry and the first cell inward and writes that state's own flux, the mass-flux inlet's mass flux exact (`tests/bcs/test_duct.py` holds that pair to 5e-3 today, the halo cell's average), the stagnation inlet's static state from the Riemann invariant, the velocity, supersonic and sem inlets' the entry's; their `postgrads` kernels go with it, and the double flux's inlet files under `navstokesdf/kernels/bcs/` write the energy row in the first cell's plane by `eside`. Exits keep the halo cell: their one datum is a pressure.
- [ ] Single precision: `eos/kernels/species.mako`'s `hspecies` and the tpg `ideal_cph` evaluate the enthalpy as a Horner polynomial in ln T in the case's precision, which in single costs the energy row three decades: 9.4e-5 relative right after `set_prims` against 2e-7 for every other row and for cpg's energy, the momentum rows following it to 7.8e-4 after 12 steps; `tests/simulator/test_precision.py` locks the measured values. A centred ln T basis, or per-species dimensionless Horner then the Y-weighted sum, would recover develop's 2.4e-5; the same fits give the reverse rates' g / (Ru T) in single to 3e-2 absolute at 1500 K, 5e-2 of the largest rate at a fresh 1500 K cell.
- [ ] The gradient of a linear field does not converge on a grid that is not a smooth mapping of index space: `_Jinv` inverts the cell's mean-of-four-edges jacobian while `dqdxyz.mako` differences the neighbouring cell centres, and the two agree only where the mapping is smooth (orders 1.5 and 1.9 on a smooth mapping, none on jostled nodes; `tests/navstokes/test_gradients.py` pins the numbers). The centre-span metric, the inverse of the 3x3 whose rows are half the centre differences, makes the kernel exact for linear fields on any grid; it also changes the wall-adjacent cells, whose spans reach the extrapolated halo centres.
- [ ] `mixture/database/species.yaml`: the JANAF formation enthalpies of eleven radicals differ from GRI-Mech 3.0's by 3 to 89 MJ/kmol (C2H -89, NCO +28, NH +20, HNCO +16, HO2 -10.5, CH2O -7, ...), which shifts the equilibrium constants and so the reverse rates of a CHEMKIN reactions-only mechanism completed from the library; a Cantera mechanism carries its own NASA7 and is unaffected. `tests/mixture/test_library.py` pins each gap as known; whether to update the entries is a data decision.

## Features

What the tree does not do yet.

- [ ] A kernel's declarations say what a name is, not only its intent and offset: a plain array with its components (`Q="in [6] at mirror"`), an array of structs by the struct's name (`state="in EosState at mirror"`), a scalar as such (`tme="in scalar"`), a broadcast and a plane by their kinds, so `ArraySpec.check` and `ScalarSpec.check` in `launch/arrays.py` refuse a kernel written for another struct or shape at declaration, with the kernel and the array named, where today a struct array is declared bare and a wrong field is the C++ compiler's error at first use. `Declaration.parse` in `codegen/grammar.py` reads the form; every template's tags change with it.
- [ ] A launch over what an exchange lands covers one slab of cells or one plane of faces, so a kernel reading deeper beside a landing, a flux under MUSCL alongside the solution's flight, is not covered and nothing refuses it; a landing as deep as the deepest reader, tiled at the wait, is the extension.
- [ ] A coupled system needs a second physics, and fluid is the only one (the systems are `BaseSimulator`, the advected and diffused bases, then the equations; the gas is the mixture they compose). Its parts: the system name per block in the grid file, `connectivity/system` beside `connectivity/nbr`; on `BaseInterface` in `multiblock/interface/base.py`, `system` and `nbrsystem`, set where `readers/grid.py` sets `nbr`; `intkind` answering `coupled`, with `offrank` false for it; `Solver._check_bcs` refusing one in a one-system case; then the coupled kinds declared per physics in its `bcs.py` and keyed in `bcvals` by the system across, the coupling the run makes from the exchange's parts, and a run above `get_solver` in `run.py` holding the clock the solver holds today (`nrt`, `tme`, `dt`, `step`, `run`), the scheme and the systems.

## Sanitation

Names, duplication, dead code, style and docstrings. A smell is a name
declared in one place and read by literal somewhere else, a base class
depending on what only one subclass declares, or a physics word in a layer
that is not the physics': each layer says only what it owns.


## Testing

The suite, its references and its recipes.


## Documentation

Written reference and docstrings that lag the tree.

