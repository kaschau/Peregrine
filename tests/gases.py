# constants only, nothing from the library
air = {
    "Air": {
        "MW": 28.96,
        "cp0": 1005.0,
        "mu0": 1.8591191080521142e-05,
        "kappa0": 0.02625394405190068,
    }
}

gases = {"air": air, "CH4_O2": "CH4_O2_FFCMY.yaml"}


def configure(config, gas, simulator="navier-stokes"):
    config["simulation"]["simulator"] = simulator
    mixture = config["mixture"]
    mixture["species"] = gases[gas]
    if gas == "air":
        mixture["eos"] = "cpg"
        mixture["trans"] = "constant"
    else:
        mixture["eos"] = "tpg"
        mixture["trans"] = "kinetic-theory"
        mixture["T-range"] = (300.0, 3500.0)


def primitives(mb, blk):
    import numpy as np

    data = mb.sim.iovars.data(blk, mb.sim.iovars.privars)
    return np.stack([data[name] for name in mb.sim.iovars.privars], axis=-1)


def hcoeff(cpcoeff):
    """The enthalpy's coefficients from cp's over the last axis, as the kernels form them: hcoeff[m] = cpcoeff[m] - (m + 1) hcoeff[m + 1] down from the top."""
    import numpy as np

    h = np.zeros_like(cpcoeff)
    h[..., -1] = cpcoeff[..., -1]
    for m in reversed(range(cpcoeff.shape[-1] - 1)):
        h[..., m] = cpcoeff[..., m] - (m + 1) * h[..., m + 1]
    return h


def ondevice(mb):
    return mb.backend.name != "openmp"


# a device compiler contracts a multiply-add differently from one kernel to the next, so only the host compares to the bit
def same(mb, a, b):
    import numpy as np

    if not ondevice(mb):
        return np.array_equal(a, b)
    return np.allclose(a, b, rtol=1e-6, atol=1e-6 * np.abs(b).max())
