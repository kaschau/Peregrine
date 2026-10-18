import numpy as np
import pytest

from peregrine.config import ConfigError

from tests.chemint.cases import duct, interior, listed
from tests.gases import same

# the hot half of the duct runs to the cap over this step; over 1e-6 s one substep covers it
STIFF = 1e-5


def rhs(mb):
    return interior(mb, mb.regs[-1])


@pytest.mark.parametrize("bulk", [1, 2])
def test_the_compaction_is_the_one_pass_form_to_the_bit(backend, bulk):
    # a bulk count equal to the cap is the one-pass form
    plain = duct(backend, dt=STIFF, bulk_substeps=200)
    compact = duct(backend, dt=STIFF, bulk_substeps=bulk)
    assert listed(compact)[0] > 0 and listed(plain)[0] == 0
    assert same(compact, rhs(compact), rhs(plain))
    # and through a step, the list emptied and refilled each stage
    for mb in (plain, compact):
        mb.step(STIFF)
    assert same(compact, interior(compact, "Q"), interior(plain, "Q"))
    assert np.isfinite(interior(compact, "Q")).all()


def test_the_list_holds_the_stiff_cells_alone(backend):
    # one bin: the header is the total and the bin's count, the codes follow
    nx = 8
    mb = duct(backend, nx=nx, dt=STIFF, bulk_substeps=1, stiff_bins=1)
    total, counts, cells = listed(mb)
    # the hot half, every cell of it, and none of the cold half
    hot = (nx // 2) * 2 * 2
    assert total == hot == counts[0]
    (cells,) = cells
    assert (cells[:, 0] < nx // 2).all()
    assert len({tuple(c) for c in cells}) == total
    # the cold half took one substep, the hot half the cap
    substeps = interior(mb, "substeps")
    assert (substeps[nx // 2 :] == 1).all() and (substeps[: nx // 2] == 200).all()
    # a cold cell's chemistry is nothing over the step, a hot cell's is not
    source = rhs(mb)[..., 5:]
    assert np.abs(source[nx // 2 :]).max() < 1e-6 * np.abs(source[: nx // 2]).max()
    # the count is reset before each right-hand side, not carried on
    mb.rhs(Q=mb.regs[0], dQ=mb.regs[-1])
    assert listed(mb)[0] == hot


def test_the_viscous_fluid_compacts_alike(backend):
    # the composition is uniform, so the species diffuse nothing and the viscous source is Euler's
    euler = duct(backend, dt=STIFF, bulk_substeps=1)
    viscous = duct(backend, dt=STIFF, bulk_substeps=1, simulator="navier-stokes")
    species = rhs(euler)[..., 5:]
    assert np.allclose(rhs(viscous)[..., 5:], species, rtol=0.0, atol=1e-14 * np.abs(species).max())
    onepass = duct(backend, dt=STIFF, bulk_substeps=200, simulator="navier-stokes")
    assert same(viscous, rhs(viscous), rhs(onepass))
    viscous.step(STIFF)
    assert np.isfinite(interior(viscous, "Q")).all()


def test_nothing_is_listed_where_one_substep_covers_the_step(backend):
    mb = duct(backend, dt=1e-6, bulk_substeps=1, stiff_bins=1)
    assert listed(mb)[0] == 0
    assert (interior(mb, "substeps") == 1).all()


def test_bulk_substeps_is_a_count_up_to_the_cap(backend):
    for bad in (-1, 0, 201, 2.5):
        with pytest.raises(ConfigError):
            duct(backend, bulk_substeps=bad)


def binof(count, bulk, cap, bins):
    if bins == 1:
        return 0
    b = int(np.log(max(count, bulk) / bulk) * bins / np.log(cap / bulk))
    return min(max(b, 0), bins - 1)


def test_the_bins_hold_the_cells_of_their_count(backend):
    # thirty short steps leave the hot cells part way into ignition at two to a few dozen substeps each; a second right-hand side of the same state bins each by the count it took the time before, so its bin is exact
    nx, bulk, cap, bins, dt = 8, 1, 200, 8, 2e-8
    hot = np.linspace(1200.0, 2800.0, nx // 2)
    mb = duct(backend, nx=nx, dt=dt, hot=hot, bulk_substeps=bulk, maxsubsteps=cap, stiff_bins=bins)
    for _ in range(30):
        mb.step(dt)
    mb.rhs(Q=mb.regs[0], dQ=mb.regs[-1])
    mb.rhs(Q=mb.regs[0], dQ=mb.regs[-1])
    total, counts, cells = listed(mb)
    assert total == counts.sum() > 0
    assert (counts > 0).sum() > 1
    # every cell's count is kept; the listed ones are those past the bulk count
    substeps = interior(mb, "substeps")
    assert (substeps > bulk).sum() == total
    seen = []
    for b, members in enumerate(cells):
        for i, j, k in members:
            assert binof(substeps[i, j, k], bulk, cap, bins) == b
        seen += [tuple(c) for c in members]
    assert len(set(seen)) == total
    # and the binned pass is the one-pass form to the bit
    plain = duct(backend, nx=nx, dt=dt, hot=hot, bulk_substeps=cap, maxsubsteps=cap)
    for _ in range(30):
        plain.step(dt)
    plain.rhs(Q=plain.regs[0], dQ=plain.regs[-1])
    assert same(mb, rhs(mb), rhs(plain))


def test_one_bin_is_one_list(backend):
    # a bin count outside 1..16 is refused
    for bad in (0, 17, 2.0):
        with pytest.raises(ConfigError):
            duct(backend, bulk_substeps=1, stiff_bins=bad)
