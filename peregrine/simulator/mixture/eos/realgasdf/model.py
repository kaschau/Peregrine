from peregrine.simulator.mixture.eos.realgas.model import RealGasModel


class DoubleFluxRealGasModel(RealGasModel):
    name = "real-gas-df"

    # the tangent plane of rho e, frozen once a step by the double flux: rho e = theta p + eps rho + sum zeta_n rho Y_n + kappa
    def planefields(self, ns):
        return ("theta", "eps", "kappa", *(f"zeta{n}" for n in range(ns - 1)))
