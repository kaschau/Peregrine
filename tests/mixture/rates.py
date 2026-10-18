# the rates kernel's own log-space forms, kept here so these twins check what the kernel evaluates with no kernel built
import cantera as ct
import numpy as np

from peregrine.backend.base.mathfns import EXPBINS, EXPBITS, EXPK, EXPTAB, LN2, LN2BINS_HI, LN2BINS_LO, LOGBINS, LOGBITS, LOGTAB, RCPTAB
from peregrine.misc import subclass_where
from peregrine.simulator.mixture import Ru
from peregrine.simulator.mixture.reaction import BaseReaction

log10e, ln10 = 0.4342944819032518, 2.302585092994046
tiny, huge = np.finfo(np.float64).tiny, np.finfo(np.float64).max
logtab, rcptab, exptab = np.array(LOGTAB), np.array(RCPTAB), np.array(EXPTAB)


# the host compiler fuses each multiply-add, so a kernel agrees with this to the ulp, not to the bit; the tests' bounds allow for it
def tlog(x):
    x = np.asarray(x, dtype=np.float64)
    bits = x.view(np.int64)
    e = ((bits >> 52) & 0x7FF) - 1022
    k = (bits >> (52 - LOGBITS)) & (LOGBINS - 1)
    m = ((bits & 0x000FFFFFFFFFFFFF) | 0x3FE0000000000000).view(np.float64)
    u = m * rcptab[k] - 1.0
    s = np.full_like(u, 0.2)
    s = s * u - 0.25
    s = s * u + 1.0 / 3.0
    s = s * u - 0.5
    s = s * u + 1.0
    return e * LN2 + logtab[k] + u * s


def texp(x):
    x = np.asarray(x, dtype=np.float64)
    t = x * EXPK + 6755399441055744.0
    k = (t.view(np.int64) & 0xFFFFFFFF).astype(np.int32).astype(np.int64)
    kd = t - 6755399441055744.0
    r = (x - kd * LN2BINS_HI) - kd * LN2BINS_LO
    s = np.full_like(r, 1.0 / 120.0)
    s = s * r + 1.0 / 24.0
    s = s * r + 1.0 / 6.0
    s = s * r + 0.5
    s = s * r + 1.0
    s = s * r + 1.0
    m = exptab[k & (EXPBINS - 1)] * s
    n = np.maximum(k >> EXPBITS, -1022)
    return (m.view(np.int64) + (n << 52)).view(np.float64)


# logT given when ln T was taken another way than the library's
def troe(r, row, T, logT=None):
    if r["nfcent"][row] == 0:
        return np.log(tiny) * log10e
    t = ((np.log(T) if logT is None else logT) - r["troeuc"]) * r["troeuhinv"]
    poly = r["lfcpoly"][row]
    lfc = poly[-1]
    for cm in poly[-2::-1]:
        lfc = lfc * t + cm
    return lfc


# the signed log-sum-exp the fit stands in for
def troe_exact(r, row, T):
    terms = r["nfcent"][row]
    if terms == 0:
        return np.log(tiny) * log10e
    e = r["fcentc0"][row, :terms] + r["fcentcT"][row, :terms] * T + r["fcentcTinv"][row, :terms] / T
    ref = e.max()
    total = np.sum(r["fcentsign"][row, :terms] * np.exp(e - ref))
    return (ref + np.log(max(tiny, total))) * log10e


# the worst relative error the fitted centres put into a rate over the mixture's temperatures
def troefiterr(mixture):
    r = mixture.chemistry.rxnconsts(mixture.species)
    T = np.geomspace(mixture.chemistry.trange[0], mixture.chemistry.trange[1], 1000)
    worst = 0.0
    for row in range(int(r["nr"])):
        if subclass_where(BaseReaction, name=r["type"][row]).troe and r["nfcent"][row] > 0:
            worst = max(worst, max(abs(troe(r, row, Tk) - troe_exact(r, row, Tk)) for Tk in T))
    return ln10 * worst


# the kernel's product form in its order of operations: K = texp(ln k - the reactants' g / Ru T) times each side's c e^(g / Ru T)
def kform(r, row, T, c, gRT, cG):
    logT, Tinv = float(tlog(T)), 1.0 / T
    clamp = np.log(huge) - np.log(2.0 * r["maxgain"])
    lk = r["logA"][row]
    if r["b"][row] != 0.0:
        lk += r["b"][row] * logT
    if r["EaR"][row] != 0.0:
        lk -= r["EaR"][row] * Tinv
    kind = subclass_where(BaseReaction, name=r["type"][row])
    if kind.thirdbody:
        ctbc = r["defeff"][row] * c.sum() + np.sum(r["effdev"][row] * c[r["effsp"][row]])
    if kind.falloff:
        logpr = float(tlog(max(tiny, ctbc))) + r["logprA"][row]
        if r["prb"][row] != 0.0:
            logpr += r["prb"][row] * logT
        if r["prEaR"][row] != 0.0:
            logpr -= r["prEaR"][row] * Tinv
        pr = float(texp(min(logpr, clamp)))
        pfac = pr / (1.0 + pr)
    if kind.troe:
        lfc = troe(r, row, T, logT)
        C, N = -0.4 - 0.67 * lfc, 0.75 - 1.27 * lfc
        A = logpr / ln10 + C
        B = N - 0.14 * A
        B2 = B * B
        lk += lfc * B2 / (B2 + A * A) * ln10
    react = [(sp, nu) for sp, nu in zip(r["reactsp"][row], r["reactnu"][row]) if nu != 0]
    prod = [(sp, nu) for sp, nu in zip(r["prodsp"][row], r["prodnu"][row]) if nu != 0]
    xk = lk
    for sp, nu in react:
        xk -= nu * gRT[sp]
    K = float(texp(min(xk, clamp)))
    if kind.thirdbody and not kind.falloff:
        K *= ctbc
    if kind.falloff:
        K *= pfac
    fwd = K
    for sp, nu in react:
        for _ in range(int(nu)):
            fwd *= cG[sp]
    if not r["reversible"][row]:
        return fwd
    rev = K
    nusum = r["nusum"][row]
    for _ in range(int(abs(nusum))):
        rev *= r["Rupref"] * T if nusum > 0 else r["prefRu"] * Tinv
    for sp, nu in prod:
        for _ in range(int(nu)):
            rev *= cG[sp]
    return fwd - rev


# exact: the library's logs and exps, Troe's centre by its log-sum-exp, the net rate by expm1; cG given, the product form for the reactions rendered in it
def progress(r, row, T, c, logc, gRT, exact=False, cG=None):
    if cG is not None and not exact and r["product"][row]:
        return kform(r, row, T, c, gRT, cG)
    flog, fexp = (np.log, np.exp) if exact else (tlog, texp)
    logT, Tinv = float(flog(T)), 1.0 / T
    lk = r["logA"][row] + r["b"][row] * logT - r["EaR"][row] * Tinv
    kind = subclass_where(BaseReaction, name=r["type"][row])
    if kind.thirdbody:
        ctbc = r["defeff"][row] * c.sum() + np.sum(r["effdev"][row] * c[r["effsp"][row]])
    if kind.thirdbody and not kind.falloff:
        lk += float(flog(max(tiny, ctbc)))
    if kind.falloff:
        logpr = float(flog(max(tiny, ctbc))) + r["logprA"][row] + r["prb"][row] * logT - r["prEaR"][row] * Tinv
        lk += min(logpr, 0.0) - np.log1p(fexp(-abs(logpr)))
    if kind.troe:
        lfc = troe_exact(r, row, T) if exact else troe(r, row, T, logT)
        C, N = -0.4 - 0.67 * lfc, 0.75 - 1.27 * lfc
        A = logpr / ln10 + C
        B = N - 0.14 * A
        B2 = B * B
        lk += lfc * B2 / (B2 + A * A) * ln10
    lrp = lk + np.sum(r["fwdorder"][row] * logc[r["fwdsp"][row]])
    clamp = np.log(huge) - np.log(2.0 * r["maxgain"])
    if not r["reversible"][row]:
        return fexp(min(lrp, clamp))
    diff = r["nusum"][row] * (r["logprefRu"] - logT)
    diff -= np.sum(r["netnu"][row] * (gRT[r["netsp"][row]] + logc[r["netsp"][row]]))
    if exact:
        return np.copysign(np.exp(min(lrp - min(diff, 0.0), clamp)), diff) * (-np.expm1(-abs(diff)))
    return fexp(min(lrp, clamp)) - fexp(min(lrp - diff, clamp))


# form: the kernel's choice per reaction, or every reaction in the log or the product form
def production(mixture, rho, Y, T, exact=False, form=None):
    r = mixture.chemistry.rxnconsts(mixture.species)
    if form is not None:
        r = {**r, "product": np.full_like(r["product"], form == "product")}
    MW = np.array([mixture.species[n]["MW"] for n in mixture.spnames])
    c = rho * Y / MW
    flog = np.log if exact else tlog
    with np.errstate(divide="ignore"):
        logc = np.maximum(r["logcfloor"], flog(c))
    gRT = mixture.eos.gibbs(mixture.species, [T], u=None if exact else [tlog(T)])[:, 0]
    cG = None if exact else np.maximum(c, 0.0) * texp(gRT)
    omega = np.zeros(mixture.ns)
    for row in range(int(r["nr"])):
        rp = progress(r, row, T, c, logc, gRT, exact, cG)
        # the padding's (0, 0) entries repeat species 0, which a fancy += would take once
        np.add.at(omega, r["netsp"][row], r["netnuMW"][row] * rp)
    return omega


def density(mixture, Y, T, p):
    MW = np.array([mixture.species[n]["MW"] for n in mixture.spnames])
    MWmix = 1.0 / np.sum(Y / MW)
    return p * MWmix / (Ru * T)


# written under the mechanism's own name, so a case root pointed at where resolves that name to the one-way twin
def oneway_twin(gas, name, where):
    reactions = []
    for rxn in gas.reactions():
        rxn.reversible = False
        reactions.append(rxn)
    twin = ct.Solution(thermo="ideal-gas", kinetics="gas", species=gas.species(), reactions=reactions)
    twin.write_yaml(str(where / name))
    return twin
