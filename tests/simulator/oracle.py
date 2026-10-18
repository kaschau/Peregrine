import numpy as np

Ru = 8314.46261815324
Tref = 298.15
fields = ("p", "T", "rhoinv", "rho", "e", "cp", "gamma", "c")


def dot(a, b):
    return (a * b).sum(axis=-1)


def velocity(side):
    return np.stack([side["u"], side["v"], side["w"]], axis=-1)


###############################################################################
# Sides
###############################################################################
def cpg(gas, p, T, Y):
    MW = np.array([s["MW"] for s in gas.values()])
    cp0 = np.array([s["cp0"] for s in gas.values()])
    Yall = np.concatenate([Y, 1.0 - Y.sum(axis=-1, keepdims=True)], axis=-1)
    R, cp = Yall @ (Ru / MW), Yall @ cp0
    cv = cp - R
    # h = cp (T - Tref) with no hf, so e = cv T - cp Tref
    side = dict(p=p, T=T, rho=p / (R * T), e=cv * T - cp * Tref, cp=cp, gamma=cp / cv)
    side["c"] = np.sqrt(side["gamma"] * R * T)
    return side


def cell(Q, state):
    rho = Q[..., 0]
    side = {f: state[..., n] for n, f in enumerate(fields)}
    side["u"], side["v"], side["w"] = (Q[..., 1 + d] / rho for d in range(3))
    side["Y"] = Q[..., 5:] / rho[..., None]
    return side


def koren(r):
    return np.maximum(0.0, np.minimum(np.minimum(2.0 * r, (2.0 + r) * 0.3333333333333333), 2.0))


def minmod(r):
    return np.maximum(0.0, np.minimum(1.0, r))


def mc(r):
    return np.maximum(0.0, np.minimum(np.minimum(2.0 * r, (1.0 + r) / 2.0), 2.0))


def superbee(r):
    return np.maximum(0.0, np.maximum(np.minimum(2.0 * r, 1.0), np.minimum(r, 2.0)))


def vanleer(r):
    return (r + np.abs(r)) / (1.0 + np.abs(r))


# every limiter under recon/, by its config name: phi of the slope ratio in Sweby's form
limiters = dict(koren=koren, minmod=minmod, mc=mc, superbee=superbee, vanleer=vanleer)


def limited(dm, dp, limiter=koren):
    with np.errstate(divide="ignore", invalid="ignore"):
        phi = limiter(dm / dp)
    return np.where(dp == 0.0, 0.0, phi)


def left(vLL, vL, vR, limiter=koren):
    return vL + 0.5 * limited(vL - vLL, vR - vL, limiter) * (vR - vL)


def right(vL, vR, vRR, limiter=koren):
    return vR - 0.5 * limited(vRR - vR, vR - vL, limiter) * (vR - vL)


def muscl(gas, Q, state, limiter=koren):
    cells = {o: cell(Q[o], state[o]) for o in (-2, -1, 0, 1)}
    names = ("p", "T", "u", "v", "w", "Y")
    L = {k: left(cells[-2][k], cells[-1][k], cells[0][k], limiter) for k in names}
    R = {k: right(cells[-1][k], cells[0][k], cells[1][k], limiter) for k in names}
    sides = []
    for side in (L, R):
        closed = cpg(gas, side["p"], side["T"], side["Y"])
        sides.append({**closed, **{k: side[k] for k in ("u", "v", "w", "Y")}})
    return sides


###############################################################################
# Flux functions: F of shape (faces, 5 + ns - 1)
###############################################################################
def kepaec(L, R, S):
    uL, uR = velocity(L), velocity(R)
    uf = 0.5 * (uL + uR)
    U = dot(S, uf)
    pf, rho = 0.5 * (L["p"] + R["p"]), 0.5 * (L["rho"] + R["rho"])
    C = rho * U
    Kj = C * 0.5 * dot(uL, uR)
    Pj = 0.5 * (L["p"] * dot(uR, S) + R["p"] * dot(uL, S))
    # Shima et al. (2021): the thermal part cv T of e is the harmonic mean, the rest the arithmetic
    thL, thR = L["cp"] / L["gamma"] * L["T"], R["cp"] / R["gamma"] * R["T"]
    ef = 2.0 * thL * thR / (thL + thR) + 0.5 * ((L["e"] - thL) + (R["e"] - thR))
    rows = [C, *(C * uf[..., d] + pf * S[..., d] for d in range(3)), ef * C + Kj + Pj]
    species = 0.5 * (L["Y"] + R["Y"]) * C[..., None]
    return np.concatenate([np.stack(rows, axis=-1), species], axis=-1)


# Pirozzoli (2010)
def kepaec4(LL, L, R, RR, S):
    return 4.0 / 3.0 * kepaec(L, R, S) - 1.0 / 6.0 * (kepaec(LL, R, S) + kepaec(L, RR, S))


# Toro ch. 10.5, with Davis's bound on the fastest wave
def rusanov(L, R, S):
    area = np.sqrt(dot(S, S))
    uL, uR = velocity(L), velocity(R)
    unL, unR = dot(uL, S) / area, dot(uR, S) / area
    lam = np.maximum(np.abs(unL) + L["c"], np.abs(unR) + R["c"]) * area
    UL, UR = unL * area, unR * area
    EL = L["rho"] * (L["e"] + 0.5 * dot(uL, uL))
    ER = R["rho"] * (R["e"] + 0.5 * dot(uR, uR))
    rows = [0.5 * (UR * R["rho"] + UL * L["rho"] - lam * (R["rho"] - L["rho"]))]
    for d in range(3):
        mL, mR = L["rho"] * uL[..., d], R["rho"] * uR[..., d]
        rows.append(0.5 * (UR * mR + R["p"] * S[..., d] + UL * mL + L["p"] * S[..., d] - lam * (mR - mL)))
    rows.append(0.5 * (UR * (ER + R["p"]) + UL * (EL + L["p"]) - lam * (ER - EL)))
    rhoYL, rhoYR = L["rho"][..., None] * L["Y"], R["rho"][..., None] * R["Y"]
    species = 0.5 * (rhoYR * UR[..., None] + rhoYL * UL[..., None] - lam[..., None] * (rhoYR - rhoYL))
    return np.concatenate([np.stack(rows, axis=-1), species], axis=-1)


# Liou (2006)
def ausmup(L, R, S, Mco=np.sqrt(0.1)):
    area = np.sqrt(dot(S, S))
    uL, uR = velocity(L), velocity(R)
    UL, UR = dot(uL, S) / area, dot(uR, S) / area
    a12 = 0.5 * (L["c"] + R["c"])
    ML, MR = UL / a12, UR / a12
    Mbarsq = 0.5 * (UL * UL + UR * UR) / (a12 * a12)
    Kp, Ku, sigma, beta = 0.25, 0.75, 1.0, 1.0 / 8.0
    Mo = np.sqrt(np.minimum(1.0, np.maximum(Mbarsq, Mco * Mco)))
    fa = Mo * (2.0 - Mo)
    M1p, M1m = 0.5 * (ML + np.abs(ML)), 0.5 * (MR - np.abs(MR))
    # each split a polynomial of one side's Mach number alone
    M2pL, M2mL = 0.25 * (ML + 1.0) ** 2, -0.25 * (ML - 1.0) ** 2
    M2pR, M2mR = 0.25 * (MR + 1.0) ** 2, -0.25 * (MR - 1.0) ** 2
    M4p = np.where(np.abs(ML) >= 1.0, M1p, M2pL * (1.0 - 16.0 * beta * M2mL))
    M4m = np.where(np.abs(MR) >= 1.0, M1m, M2mR * (1.0 + 16.0 * beta * M2pR))
    alpha = 3.0 / 16.0 * (-4.0 + 5.0 * fa * fa)
    with np.errstate(divide="ignore", invalid="ignore"):
        p5p = np.where(np.abs(ML) >= 1.0, M1p / ML, M2pL * ((2.0 - ML) - 16.0 * alpha * ML * M2mL))
        p5m = np.where(np.abs(MR) >= 1.0, M1m / MR, M2mR * ((-2.0 - MR) + 16.0 * alpha * MR * M2pR))
    rho12 = 0.5 * (L["rho"] + R["rho"])
    M12 = M4p + M4m - Kp / fa * np.maximum(1.0 - sigma * Mbarsq, 0.0) * (R["p"] - L["p"]) / (rho12 * a12 * a12)
    mdot = a12 * M12 * np.where(M12 > 0.0, L["rho"], R["rho"]) * area
    p12 = p5p * L["p"] + p5m * R["p"] - Ku * p5p * p5m * (L["rho"] + R["rho"]) * fa * a12 * (UR - UL)
    up = mdot > 0.0
    U = {k: np.where(up[..., None] if k == "Y" else up, L[k], R[k]) for k in L}
    uU = velocity(U)
    rows = [mdot, *(mdot * uU[..., d] + p12 * S[..., d] for d in range(3))]
    rows.append(mdot * (U["e"] + 0.5 * dot(uU, uU) + U["p"] / U["rho"]))
    return np.concatenate([np.stack(rows, axis=-1), mdot[..., None] * U["Y"]], axis=-1)


# Toro ch. 10.4, with Davis's outer wave speeds
def hllc(L, R, S):
    area = np.sqrt(dot(S, S))
    n = S / area[..., None]
    uL, uR = velocity(L), velocity(R)
    unL, unR = dot(uL, S) / area, dot(uR, S) / area
    SL, SR = unL - L["c"], unR + R["c"]
    Sstar = (R["p"] - L["p"] + L["rho"] * unL * (SL - unL) - R["rho"] * unR * (SR - unR)) / (L["rho"] * (SL - unL) - R["rho"] * (SR - unR))
    left = (SL >= 0.0) | (Sstar >= 0.0)
    star = np.where(left, SL < 0.0, SR >= 0.0)
    X = {k: np.where(left[..., None] if k == "Y" else left, L[k], R[k]) for k in L}
    U, SX = np.where(left, unL, unR), np.where(left, SL, SR)
    uX = velocity(X)
    et = X["e"] + 0.5 * dot(uX, uX)
    mdot = U * X["rho"] * area
    rows = [mdot, *(mdot * uX[..., d] + X["p"] * S[..., d] for d in range(3)), mdot * et + U * X["p"] * area]
    own = np.concatenate([np.stack(rows, axis=-1), mdot[..., None] * X["Y"]], axis=-1)
    # the jump to the star state as the density's jump carrying the side's velocity, energy and fractions, plus the star density carrying the contact's speed along the normal and, in the energy, the pressure's work
    with np.errstate(divide="ignore", invalid="ignore"):
        rhostar = X["rho"] * (SX - U) / (SX - Sstar)
        drho, dU = rhostar - X["rho"], rhostar * (Sstar - U)
        rows = [drho, *(drho * uX[..., d] + dU * n[..., d] for d in range(3))]
        rows.append(drho * et + dU * (Sstar + X["p"] / (X["rho"] * (SX - U))))
        jump = np.concatenate([np.stack(rows, axis=-1), drho[..., None] * X["Y"]], axis=-1) * (SX * area)[..., None]
    return own + np.where(star[..., None], jump, 0.0)


###############################################################################
# The switch
###############################################################################
def jameson(state, pgain=1.0, ggain=1.0):
    p = {o: s[..., fields.index("p")] for o, s in state.items()}
    rho = {o: s[..., fields.index("rho")] for o, s in state.items()}
    c = {o: s[..., fields.index("c")] for o, s in state.items()}

    def sensor(fm, f, fp):
        return np.abs(fp - 2.0 * f + fm) / np.abs(fp + 2.0 * f + fm)

    pressure = pgain * np.maximum(sensor(p[-2], p[-1], p[0]), sensor(p[-1], p[0], p[1]))
    gL, gR = rho[-1] * c[-1] ** 2 / p[-1], rho[0] * c[0] ** 2 / p[0]
    g = ggain * np.abs(gR - gL) / np.abs(gR + gL)
    shock = np.minimum(pressure, 1.0)
    contact = np.minimum(np.maximum(pressure, g), 1.0)
    return shock, contact


# Hendrickson, Kartha and Candler's sensor
def ducros(grads, Q, dijk, nu=0.1, floor=0.0):
    def sensor(g, Qc, d):
        div = g[..., 0, 0] + g[..., 1, 1] + g[..., 2, 2]
        wx = g[..., 2, 1] - g[..., 1, 2]
        wy = g[..., 0, 2] - g[..., 2, 0]
        wz = g[..., 1, 0] - g[..., 0, 1]
        div2, omega2 = div * div, wx * wx + wy * wy + wz * wz
        tiny = np.finfo(div.dtype).tiny
        D = np.minimum(4.0 / 3.0 * div2 / (div2 + omega2 + tiny), 1.0)
        u2 = (Qc[..., 1:4] ** 2).sum(axis=-1) / Qc[..., 0] ** 2
        length = 0.5 * d.min(axis=-1)
        theta = div2 / (div2 + nu * nu * u2 / (length * length) + tiny)
        return np.maximum(D - floor, 0.0) * theta + floor

    w = np.maximum(sensor(grads[-1], Q[-1], dijk[-1]), sensor(grads[0], Q[0], dijk[0]))
    return w, w


###############################################################################
# Corrections: rows added to a primary's F
###############################################################################
def matrixdissipation(gas, L, R, S, shock, contact):
    # Swanson and Turkel: -1/2 Gamma |Gamma^-1 A| dq on the jump in (p, u, T, Y), the matrix at the face's mean state
    MW = np.array([s["MW"] for s in gas.values()])
    cp0 = np.array([s["cp0"] for s in gas.values()])
    area = np.sqrt(dot(S, S))
    n = S / area[..., None]
    uL, uR = velocity(L), velocity(R)
    uf = 0.5 * (uL + uR)
    un, ke = dot(uf, n), 0.5 * dot(uf, uf)
    rho, p, T, c = (0.5 * (L[k] + R[k]) for k in ("rho", "p", "T", "c"))
    Y, dY = 0.5 * (L["Y"] + R["Y"]), R["Y"] - L["Y"]
    # the eos's density derivatives and species enthalpies at the face: rho = p / (R T), h = cp (T - Tref)
    Yall = np.concatenate([Y, 1.0 - Y.sum(axis=-1, keepdims=True)], axis=-1)
    MWmix = 1.0 / (Yall @ (1.0 / MW))
    rho_p, rho_T = rho / p, -rho / T
    rho_Y = -(rho * MWmix)[..., None] * (1.0 / MW[:-1] - 1.0 / MW[-1])
    h = cp0 * (T[..., None] - Tref)
    dh = h[..., :-1] - h[..., -1:]
    cp = 0.5 * (L["cp"] + R["cp"])
    H = 0.5 * (L["e"] + L["p"] / L["rho"] + R["e"] + R["p"] / R["rho"]) + ke
    sE = T * rho_T / rho
    acoustic, convective = c, np.abs(un)
    ws, wc = shock, np.maximum(shock, contact)
    # the damped jump, field by field
    dp = ws * acoustic * (R["p"] - L["p"])
    dus = uR - uL
    dun = dot(dus, n)
    du = ws[..., None] * (convective[..., None] * dus + ((acoustic - convective) * dun)[..., None] * n)
    dTac = -T * rho_T * (R["p"] - L["p"]) / (rho * rho * cp)
    dT = wc * convective * (R["T"] - L["T"]) + ws * (acoustic - convective) * dTac
    dYs = (wc * convective)[..., None] * dY
    # Gamma times it: the density first, the rest on it
    drho = rho_p * dp + rho_T * dT + (rho_Y * dYs).sum(axis=-1)
    dE = (rho_p * H + sE) * dp + rho * dot(uf, du) + (rho_T * H + rho * cp) * dT
    dE = dE + ((H[..., None] * rho_Y + rho[..., None] * dh) * dYs).sum(axis=-1)
    half = -0.5 * area
    rows = [half * drho, *(half * (drho * uf[..., d] + rho * du[..., d]) for d in range(3)), half * dE]
    species = half[..., None] * (drho[..., None] * Y + rho[..., None] * dYs)
    return np.concatenate([np.stack(rows, axis=-1), species], axis=-1)


functions = dict(kepaec=kepaec, rusanov=rusanov, ausmup=ausmup, hllc=hllc)
