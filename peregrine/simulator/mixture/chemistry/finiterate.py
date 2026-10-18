import math
from types import SimpleNamespace

import numpy as np
from numpy.polynomial import chebyshev

from peregrine.codegen import Struct
from peregrine.misc import subclass_where
from peregrine.simulator.mixture.caloric.tdepcp.model import TempDepCpModel
from peregrine.simulator.mixture.chemistry.base import BaseChemistryModel
from peregrine.simulator.mixture.constants import Ru
from peregrine.simulator.mixture.eos.base import BaseEosModel
from peregrine.simulator.mixture.reaction import BaseReaction

LOG10E = 0.4342944819032518
LN10 = 2.302585092994046
# the table exp's arguments stay inside this over the case's temperatures for a reaction to take the product form
EXPBOUND = 680.0


class FiniteRateModel(BaseChemistryModel):
    name = "finite-rate"

    def __init__(self, cfgsect, reactions):
        super().__init__(cfgsect, reactions)
        # the reverse rates are made from the entropy fits of the thermally perfect caloric model
        self.deps = (TempDepCpModel,)

    def constants(self, species):
        records, scalars = self._reactions(species)
        column = self._columns(records)
        sets = self._thirdbodies(records)
        # by reaction, row k the kernel's fold index; the Troe fits by Troe reaction and the efficiencies by third-body set, as the reactions are numbered
        own = dict(
            rxnlogA=column["logA"],
            rxnb=column["b"],
            rxnEaR=column["EaR"],
            rxnlogprA=column["logprA"],
            rxnprb=column["prb"],
            rxnprEaR=column["prEaR"],
            rxnnusum=column["nusum"],
            rxnlfcpoly=self.padded([r.lfcpoly for r in records if r.troe]),
            rxnfwdorder=column["fwdorder"],
            rxnnetnu=column["netnu"],
            rxnnetnuMW=column["netnuMW"],
            rxnreactnu=column["reactnu"],
            tbdefeff=np.array([t.defeff for t in sets]),
            tbeffdev=self.padded([[dev for _, dev in t.eff] for t in sets]),
        )
        return dict(super().constants(species), **scalars, **own)

    # the third-body sets the reactions share, each reaction numbered by its set
    @staticmethod
    def _thirdbodies(records):
        sets, keys = [], []
        for r in records:
            r.tb = None
            if r.thirdbody:
                key = (r.defeff, tuple(r.eff))
                if key not in keys:
                    keys.append(key)
                    sets.append(SimpleNamespace(defeff=r.defeff, eff=list(r.eff)))
                r.tb = keys.index(key)
        return sets

    # the rate state a cell's fold carries: logc and logtb serve the log form, which single precision always takes, cG and ctb the product form
    def structs(self, species):
        rxns = self.terms(species)
        ns, ntb = len(species), len(rxns.thirdbodies)
        fields = ("T", "logT", "Tinv", "logprefRuT", "prefRuT", "prefRuTinv", "ctot")
        arrays = {"logc": ns, "gRT": ns, **({"cG": ns} if rxns.product else {}), **({"logtb": ntb} if ntb else {}), **({"ctb": ntb} if rxns.plainbodies else {})}
        return (Struct("RateState", fields, arrays),)

    def terms(self, species):
        records, scalars = self._reactions(species)
        thirdbodies = self._thirdbodies(records)
        bodies = bool(thirdbodies)
        troes = any(r.troe for r in records)
        # the bodies read as a concentration, by the product-form three-body reactions, and as a log, by the falloff and log-form ones
        plainbodies = sorted({r.tb for r in records if r.product and r.thirdbody and not r.falloff})
        logbodies = sorted({r.tb for r in records if r.thirdbody and (r.falloff or not r.product)})
        product, logform = any(r.product for r in records), not all(r.product for r in records)
        return SimpleNamespace(reactions=records, thirdbodies=thirdbodies, bodies=bodies, troes=troes, maxgain=float(scalars["maxgain"]), product=product, logform=logform, plainbodies=plainbodies, logbodies=logbodies)

    # a reaction's one-way rates as products of concentrations and per-species exponentials: its orders are its reactant stoichiometry, whole, and every exponential's argument stays inside the table exp's range over the case's temperatures, K's with Troe's blend counted
    @classmethod
    def _productform(cls, r, T, logT, Tinv, gRT, ns):
        whole = all(float(v).is_integer() and v > 0 for _, v in r.react + r.prod)
        if not whole or dict(r.fwd) != dict(r.react):
            return False
        species = [sp for sp, _ in r.react + r.prod]
        if np.abs(gRT[species]).max() >= EXPBOUND:
            return False
        xk = r.logA + r.b * logT - r.EaR * Tinv - sum(nu * gRT[sp] for sp, nu in r.react)
        low, high = xk.min(), xk.max()
        if r.troe:
            lfc = cls.lfc(r.fcent, T)
            low, high = low + LN10 * min(lfc.min(), 0.0), high + LN10 * max(lfc.max(), 0.0)
        if r.falloff:
            xpr = r.logprA + r.prb * logT - r.prEaR * Tinv
            # the body's concentration enters Pr as a factor, up to e^8 of headroom
            if np.abs(xpr).max() >= EXPBOUND - 8.0:
                return False
        if not (low > -EXPBOUND and high < EXPBOUND):
            return False
        # the chains of factors stay finite in the kernel's order: a concentration at most e^12 kmol / m3, a third body's at most that times its efficiencies summed
        logc = 12.0
        chain = xk + (logc + np.log(r.defeff * ns + sum(max(dev, 0.0) for _, dev in r.eff)) if r.thirdbody and not r.falloff else 0.0)
        for sp, nu in r.react:
            for _ in range(int(nu)):
                chain = chain + gRT[sp] + logc
                if chain.max() >= EXPBOUND:
                    return False
        chain = xk + abs(r.nusum) * 6.0
        for sp, nu in r.prod:
            for _ in range(int(nu)):
                chain = chain + gRT[sp] + logc
                if chain.max() >= EXPBOUND:
                    return False
        return True

    def rxnconsts(self, species):
        records, scalars = self._reactions(species)
        return {**self._columns(records), **scalars}

    @classmethod
    def _columns(cls, records):
        padded = cls.padded
        column = {
            "nr": np.array(len(records)),
            "type": np.array([r.name for r in records]),
            "reversible": np.array([int(r.reversible) for r in records], np.int32),
            "nfcent": np.array([len(r.fcent) for r in records], np.int32),
        }
        for name in ("defeff", "logA", "b", "EaR", "logprA", "prb", "prEaR", "nusum"):
            column[name] = np.array([getattr(r, name) for r in records])
        for k, name in enumerate(("fcentsign", "fcentc0", "fcentcT", "fcentcTinv")):
            column[name] = padded([[t[k] for t in r.fcent] for r in records])
        column["lfcpoly"] = padded([r.lfcpoly for r in records])
        column["fwdsp"] = padded([[sp for sp, _ in r.fwd] for r in records], np.int32)
        column["fwdorder"] = padded([[v for _, v in r.fwd] for r in records])
        column["netsp"] = padded([[sp for sp, _, _ in r.net] for r in records], np.int32)
        column["netnu"] = padded([[nu for _, nu, _ in r.net] for r in records])
        column["netnuMW"] = padded([[nuMW for _, _, nuMW in r.net] for r in records])
        # reverse orders d ln(reverse) / d ln c: the forward orders plus the net stoichiometry
        rev = []
        for r in records:
            order = {}
            for sp, v in (r.fwd + [(sp, nu) for sp, nu, _ in r.net]) if r.reversible else []:
                order[sp] = order.get(sp, 0.0) + v
            rev.append([(sp, v) for sp, v in order.items() if v != 0])
        column["revsp"] = padded([[sp for sp, _ in row] for row in rev], np.int32)
        column["revorder"] = padded([[v for _, v in row] for row in rev])
        # the product form's sides, and which reactions the kernel renders in it
        column["product"] = np.array([int(r.product) for r in records], np.int32)
        column["reactsp"] = padded([[sp for sp, _ in r.react] for r in records], np.int32)
        column["reactnu"] = padded([[v for _, v in r.react] for r in records])
        column["prodsp"] = padded([[sp for sp, _ in r.prod] for r in records], np.int32)
        column["prodnu"] = padded([[v for _, v in r.prod] for r in records])
        column["effsp"] = padded([[sp for sp, _ in r.eff] for r in records], np.int32)
        column["effdev"] = padded([[v for _, v in r.eff] for r in records])
        return column

    def _reactions(self, species):
        names = list(species)
        MW = np.array([species[n]["MW"] for n in names])
        index = {n: k for k, n in enumerate(names)}
        reactions = [r for r in self.reactions if r["rate"][0] != 0.0]
        for r in reactions:
            if r["rate"][0] < 0 or r.get("lowrate", (1.0,))[0] <= 0:
                why = "a rate is evaluated in log space, A is positive"
                raise ValueError(f"{r['equation']}: {why}")
        # Troe's centre is a polynomial in t = (ln T - uc) / uh over the case's temperatures
        uc, uh = 0.5 * (np.log(self.trange[1]) + np.log(self.trange[0])), 0.5 * (np.log(self.trange[1]) - np.log(self.trange[0]))
        records, nets, fwds = [], [], []
        for r in reactions:
            A, b, EaR = r["rate"]
            # a falloff's pressure ratio k0 / kinf is its own Arrhenius rate
            A0, b0, EaR0 = r.get("lowrate", (1.0, 0.0, 0.0))
            logA = np.log(A)
            nu = {}
            for sp, v in r["reactants"].items():
                nu[sp] = nu.get(sp, 0.0) - v
            for sp, v in r["products"].items():
                nu[sp] = nu.get(sp, 0.0) + v
            net = [(index[sp], v) for sp, v in nu.items() if v != 0]
            fwd = [(index[sp], v) for sp, v in r["fwd"].items() if v != 0]
            react = [(index[sp], v) for sp, v in r["reactants"].items() if v != 0]
            prod = [(index[sp], v) for sp, v in r["products"].items() if v != 0]
            eff = [(index[sp], e - r["defeff"]) for sp, e in r["efficiencies"].items() if e != r["defeff"]]
            kind = subclass_where(BaseReaction, name=r["type"])
            records.append(
                kind(
                    reversible=bool(r["reversible"]),
                    # the Troe reactions numbered in turn: the row of the kernel's table of their fits
                    troeidx=sum(1 for x in records if x.troe) if kind.troe else None,
                    logA=float(logA),
                    b=float(b),
                    EaR=float(EaR),
                    logprA=float(np.log(A0) - logA),
                    prb=float(b0 - b),
                    prEaR=float(EaR0 - EaR),
                    fcent=self._fcent(r.get("troe")),
                    lfcpoly=self._lfcpoly(self._fcent(r.get("troe")), uc, uh),
                    fwd=fwd,
                    react=react,
                    prod=prod,
                    net=[(sp, v, float(v * MW[sp])) for sp, v in net],
                    eff=eff,
                    nusum=float(sum(v for _, v in net)),
                    defeff=float(r["defeff"]),
                )
            )
            nets.append(net)
            fwds.append(fwd)
        grid = np.geomspace(self.trange[0], self.trange[1], 2048)
        logT, Tinv, gRT = np.log(grid), 1.0 / grid, BaseEosModel.gibbs(species, grid)
        for r in records:
            r.product = self._productform(r, grid, logT, Tinv, gRT, len(species))
        # the worst amplification from one rate of progress to a production rate
        gain = np.zeros(len(species))
        for row in nets:
            for sp, v in row:
                gain[sp] += abs(v)
        maxgain = max(1.0, (gain * MW).max()) if records else 1.0
        logcfloor = self._logcfloor(species, reactions, records, nets, fwds, index)
        scalars = {
            "logprefRu": np.array(np.log(101325.0 / Ru)),
            "prefRu": np.array(101325.0 / Ru),
            "Rupref": np.array(Ru / 101325.0),
            "logcfloor": np.array(logcfloor),
            "cfloor": np.array(math.exp(logcfloor)),
            "maxgain": np.array(maxgain),
            "troeuc": np.array(uc),
            "troeuhinv": np.array(1.0 / uh),
        }
        return records, scalars

    # a signed log-sum-exp about the largest term; a centre with no terms takes Cantera's floor
    @staticmethod
    def lfc(fcent, T):
        T = np.asarray(T, dtype=np.float64)
        if not fcent:
            return np.full(T.shape, np.log(np.finfo(np.float64).tiny) * LOG10E)
        e = np.array([c0 + cT * T + cTinv / T for _, c0, cT, cTinv in fcent])
        ref = e.max(axis=0)
        total = sum(sign * np.exp(em - ref) for (sign, *_), em in zip(fcent, e))
        return (ref + np.log(np.maximum(np.finfo(np.float64).tiny, total))) * LOG10E

    # a Chebyshev least-squares fit in the power basis, coefficients rising: degree 12 is within 1e-3 of the centre, the precision Troe parameters are given to
    @classmethod
    def _lfcpoly(cls, fcent, uc, uh, degree=12):
        if not fcent:
            return []
        k = np.arange(8 * degree + 64)
        t = np.cos(np.pi * (k + 0.5) / k.size)
        coef = chebyshev.chebfit(t, cls.lfc(fcent, np.exp(uc + uh * t)), degree)
        return [float(c) for c in chebyshev.cheb2poly(coef)]

    # a mechanism disables a Troe term by a sentinel T; a term whose exponent underflows at every gas temperature is dropped, a coefficient below rounding is zeroed
    @staticmethod
    def _fcent(troe):
        if troe is None:
            return []
        alpha, T3, T1, T2 = troe
        tmin, tmax = 100.0, 20000.0
        candidates = []
        if alpha != 1.0 and T3 != 0:
            sign, c0 = np.sign(1.0 - alpha), np.log(abs(1.0 - alpha))
            candidates.append((sign, c0, -1.0 / T3, 0.0))
        if alpha != 0.0 and T1 != 0:
            sign, c0 = np.sign(alpha), np.log(abs(alpha))
            candidates.append((sign, c0, -1.0 / T1, 0.0))
        if T2:
            candidates.append((1.0, 0.0, 0.0, -T2))
        terms = []
        for sign, c0, cT, cTinv in candidates:
            if cT * tmin < -745 or cTinv / tmax < -745:
                continue
            cT = 0.0 if abs(cT) * tmax < 1e-14 else cT
            cTinv = 0.0 if abs(cTinv) / tmin < 1e-14 else cTinv
            terms.append((sign, c0, cT, cTinv))
        return terms

    # deep enough that every direction of every reaction an absent species joins is below e^-45 at any state over the T-range, with ln c <= 12 for the other factors
    def _logcfloor(self, species, reactions, records, net, fwd, index):
        logcmax, margin = 12.0, 45.0
        Tlow, Thigh = self.trange
        T = np.geomspace(max(Tlow, 100.0), Thigh, 256)
        logT, gRT = np.log(T), BaseEosModel.gibbs(species, T)
        # no shallower than the smallest normal float32, ln(1.18e-38) = -87.3
        required = 90.0
        for r, record, netrow, fwdrow in zip(reactions, records, net, fwd):
            A, b, EaR = r["rate"]
            logkf = np.log(A) + b * logT - EaR / T
            aux = record.boost(logcmax)
            top = logkf.max() + aux
            for k, vk in fwdrow:
                if vk <= 0:
                    continue
                other = sum(v * logcmax for n, v in fwdrow if n != k and v > 0)
                required = max(required, (top + other + margin) / vk)
            if r["reversible"]:
                affinity = sum(v * gRT[n] for n, v in netrow)
                nusum = sum(v for _, v in netrow)
                logkr = logkf + affinity + nusum * np.log(Ru * T / 101325.0)
                products = r["products"].items()
                rev = [(index[sp], float(v)) for sp, v in products if v > 0]
                top = logkr.max() + aux
                for k, vk in rev:
                    other = sum(v * logcmax for n, v in rev if n != k)
                    required = max(required, (top + other + margin) / vk)
        return -float(required)
