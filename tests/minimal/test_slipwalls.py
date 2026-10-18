from functools import partial

import numpy as np
import pytest

import peregrine as pg

from tests.minimal.cases import advflux, box


# nearest the wall first, the order interface.halo gives its layers in
def images(interface, array):
    ng = interface.ng
    planes = np.moveaxis(array, interface.axis, 0)
    if interface.is_low:
        return planes[ng : 2 * ng]
    return planes[len(planes) - 2 * ng : len(planes) - ng][::-1]


def mirrored(image, axis):
    out = image.copy()
    out[..., 1 + axis] *= -1.0
    return out


def walled(backend, entry):
    config, mesh = box(dims=(5, 6, 7))
    config["bcvals"]["wall"] = entry
    advflux(config, recon="muscl", limiter="koren")
    mb = pg.get_solver(config, mesh, backend(config))
    blk = mb.blocks[0]
    # two halo layers, so a halo filled from the first cell alone would not pass
    assert blk.ng == 2
    # a state varying along every axis, halos included, over the uniform one
    privars = mb.sim.iovars.privars
    prims = np.empty((*blk.Q.get().shape[:3], len(privars)))
    prims[...] = mb.sim.ics()
    i, j, k = np.indices(prims.shape[:3])
    prims[..., privars.index("T")] = 300.0 + 5.0 * i + 7.0 * j + 9.0 * k
    prims[..., privars.index("u")] = 10.0 + i
    prims[..., privars.index("v")] = 20.0 + j
    prims[..., privars.index("w")] = 30.0 + k
    mb.set_prims([prims])
    return mb


# the isothermal walls: one hotter than the state and one far colder
WALLS = [
    ({"kind": "slp-adia-wall"}, lambda Tin: Tin),
    ({"kind": "slp-isot-wall", "T": 400.0}, lambda Tin: np.full_like(Tin, 400.0)),
    ({"kind": "slp-isot-wall", "T": 50.0}, lambda Tin: np.full_like(Tin, 50.0)),
]


@pytest.mark.parametrize("entry, Thalo", WALLS)
def test_the_halo_is_the_images_reflection(backend, entry, Thalo):
    mb = walled(backend, entry)
    blk = mb.blocks[0]
    mb.apply_bcs("preadvflux")
    Q, state = blk.Q.get(), blk.state.get()
    fields = list(mb.sim.statefields)
    T, p, rho, e = (fields.index(f) for f in ("T", "p", "rho", "e"))
    ng = blk.ng
    # the interface proper: the edges and corners of a halo are never written
    proper = np.s_[ng:-ng, ng:-ng]
    adiabatic = entry["kind"] == "slp-adia-wall"
    # the adiabatic wall copies and reflects, bitwise; the isothermal wall rebuilds the halo through the eos, measured within 3.2e-16
    close = np.array_equal if adiabatic else partial(np.allclose, rtol=1e-14, atol=0.0)
    for interface in blk.intfaces:
        axis = interface.axis
        tangential = [m for m in range(3) if m != axis]
        layers = zip(interface.halo(Q), images(interface, Q), interface.halo(state), images(interface, state))
        for halo, image, haloS, imageS in layers:
            halo, image, haloS, imageS = (a[proper] for a in (halo, image, haloS, imageS))
            u_h = halo[..., 1:4] / halo[..., :1]
            u_i = image[..., 1:4] / image[..., :1]
            # the normal velocity flipped, the tangential kept
            assert close(u_h[..., axis], -u_i[..., axis])
            assert close(u_h[..., tangential], u_i[..., tangential])
            # the composition the image's
            assert close(halo[..., 5:] / halo[..., :1], image[..., 5:] / image[..., :1])
            # the temperature as the wall says, the pressure the image's
            assert close(haloS[..., T], Thalo(imageS[..., T]))
            assert close(haloS[..., p], imageS[..., p])
            # the solution is the state's: its density, and its energy at the halo's own speed
            assert close(halo[..., 0], haloS[..., rho])
            energy = haloS[..., rho] * (haloS[..., e] + 0.5 * np.sum(u_h**2, axis=-1))
            assert np.allclose(halo[..., 4], energy, rtol=1e-14, atol=0.0)
            if adiabatic:
                assert np.array_equal(halo, mirrored(image, axis))
