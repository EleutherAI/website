#!/usr/bin/env python3
"""
Figures for the "AI 2027 in this model" section of the
dynamical-models-of-ai-governability post.

The model core is a port of the V4 GROWTH-PEGGED delta-general sigma-clock
model (in-app engine: basin-explorer/src/BasinExplorer.jsx, commit b87373a+):

  F_u = (1 - l_k O) Lk + (k_uu - lO) q_u,   Lk = k_cu q_c + k_hu q_h
  F_c = (1-k_cu) q_c + (1-k_hu) q_h + (1-d) O (l_k Lk + l q_u)
  G   = q_c + q_h + k_uu q_u - d O (l_k Lk + l q_u)
  e'  = c_0 + a_E/M c_M q_u / G - e        (passive opacity per doubling)

c_0 is pinned from the long-run observability dial W at the rare-uncooperative
point: c_0/c_M = (1-W)/(W(1 - d*l_k*k_cu*W)), re-solved per parameter cell.
Long-run observability is the minus root of c*d*P*O^2 - D*O + 1 = 0 with
P = l_k k_cu + l r, D = 1 + c + (a + c k_uu) r; fixed points are roots of
g(r) = k_cu + b r - O*(r) P(r) S(r), S = 1 + (1-d) r (a cubic in r).

The port is validated against pinned v4 ground-truth numbers before any
figure is drawn (fixtures from _scratch/review/scripts/v4/calibrations_v4.py
and landmarks_v4.py, digests in calibrations-v4.md / landmarks-v4.md). There
is no eradication regime and, at delta = 1 and k_cu = 0.9, no basin at any
fix rate under the dial pin (floor validity d*k_cu <= 4/(4+l_k)).

Two figures:
  fig1: AI 2027 located as a high-leakage run (q_u rises, O collapses).
  fig2 (lead): at high leakage, observability alone cannot recover
        cooperation; the only cooperative outcome is extreme suppression
        with a high endemic attractor.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import os

OUT = "/Users/davidjohnston/Dropbox/Mac/Documents/Eleuther Docs/website/static/images/blog/dynamical-models"
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------------------
# palette (matches the basin-explorer warm-paper theme)
# ---------------------------------------------------------------------------
C = dict(
    bg="#faf7ef", panel="#fdfbf4", border="#d8d0bb", borderStrong="#a89c80",
    fg="#1d1c17", fgDim="#5a5448", fgMuted="#8a8270",
    accent="#a85d1f", coop="#3a7a4e", uncoop="#a83b2e", human="#6f4585",
    grid="#e3dcc6",
    # region fills
    fill_coop="#cfe0d3", fill_bistable="#ecd9ad", fill_escape="#e7c2ba",
    edge_coop="#3a7a4e", edge_escape="#a83b2e",
)
for fam in ("IBM Plex Sans", "DejaVu Sans"):
    if any(fam in f.name for f in mpl.font_manager.fontManager.ttflist):
        mpl.rcParams["font.family"] = fam
        break
mpl.rcParams.update({
    "figure.facecolor": C["bg"], "axes.facecolor": C["panel"],
    "savefig.facecolor": C["bg"], "axes.edgecolor": C["borderStrong"],
    "text.color": C["fg"], "axes.labelcolor": C["fg"],
    "xtick.color": C["fgDim"], "ytick.color": C["fgDim"],
    "axes.titlecolor": C["fg"], "font.size": 11,
})

# ===========================================================================
# MODEL CORE — v4 growth-peg port of the in-app engine (src/BasinExplorer.jsx)
# ===========================================================================
def pin_c(Omega, delta, lk, k_cu):
    """c = c_0/c_M such that long-run O*(q_u -> 0) equals the dial Omega."""
    den = Omega * (1.0 - delta * lk * k_cu * Omega)
    if den <= 0:
        raise ValueError("O*-dial pin infeasible")
    return (1.0 - Omega) / den

def computeO(m, e):
    m = max(m, 1e-12); e = max(e, 1e-12)
    return m / (m + e)

def make_deriv(p):
    def f(s):
        Q, m, e, eta = s
        O = computeO(m, e)
        Lk = p["k_cu"] * (1 - Q) + p["k_hu"] * eta   # leakage inflow
        Fc = ((1 - Q) * (1 - p["k_cu"]) + eta * (1 - p["k_hu"])
              + (1 - p["delta"]) * O * (p["lk"] * Lk + p["l"] * Q))
        Fu = (1 - p["lk"] * O) * Lk + (p["k_uu"] - p["l"] * O) * Q
        G = Fc + Fu
        if G <= 1e-10:
            return np.array([0.0, 0.0, 0.0, -eta])
        FM = p["c_M"] * ((1 - Q) + eta / p["a_ai_h"])
        FE_active = p["a_e_m"] * p["c_M"] * Q
        # v4: passive opacity is per-doubling (undiluted); only active
        # evasion is a production flow.
        return np.array([Fu / G - Q, FM / G - m,
                         p["c_0"] + FE_active / G - e, -eta])
    return f

def computeG(s, p):
    Q, m, e, eta = s
    O = computeO(m, e)
    Lk = p["k_cu"] * (1 - Q) + p["k_hu"] * eta
    return (1 - Q) + eta + p["k_uu"] * Q - p["delta"] * O * (p["lk"] * Lk + p["l"] * Q)

def rk4_step(state, dt, deriv):
    k1 = deriv(state)
    k2 = deriv(state + 0.5 * dt * k1)
    k3 = deriv(state + 0.5 * dt * k2)
    k4 = deriv(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

def simulate(p, ic, sigma_max=15.0, dt=0.005):
    deriv = make_deriv(p)
    steps = max(1, int(np.ceil(sigma_max / dt)))
    state = np.array(ic, dtype=float)
    traj = [dict(sigma=0.0, Q=state[0], m=state[1], e=state[2], eta=state[3],
                 O=computeO(state[1], state[2]))]
    escaped = False
    for i in range(1, steps + 1):
        if computeG(state, p) <= 1e-6:
            escaped = True
            break
        state = rk4_step(state, dt, deriv)
        state[0] = min(max(state[0], 0.0), 1.0)
        state[1] = max(state[1], 0.0)
        state[2] = max(state[2], 0.0)
        state[3] = max(state[3], 0.0)
        if not np.all(np.isfinite(state)):
            escaped = True; break
        if state[0] > 0.999:
            escaped = True
        traj.append(dict(sigma=i * dt, Q=state[0], m=state[1], e=state[2],
                         eta=state[3], O=computeO(state[1], state[2])))
        if escaped:
            break
    return dict(traj=traj, escaped=escaped, final=traj[-1])

def calibrated_rates(p):
    """c_M from O(0)=1/(1+R0), m0=1, bounded-gate G0 (matches the app);
    c_0 from the per-cell dial pin (v4)."""
    Q, eta = p["q0"], p["eta0"]
    R0 = max(p["R0"], 1e-9)
    O0 = 1 / (1 + R0)
    m0 = 1.0
    Lk0 = p["k_cu"] * (1 - Q) + p["k_hu"] * eta
    G0 = max((1 - Q) + eta + p["k_uu"] * Q
             - p["delta"] * O0 * (p["lk"] * Lk0 + p["l"] * Q), 0.05)
    mon0 = max((1 - Q) + eta / p["a_ai_h"], 1e-9)
    Ostar = min(max(p["Ostar"], 1e-6), 1 - 1e-6)
    c_M = G0 * m0 / mon0
    c_0 = pin_c(Ostar, p["delta"], p["lk"], p["k_cu"]) * c_M
    return c_M, c_0

def ode_params(p):
    c_M, c_0 = calibrated_rates(p)
    return dict(k_uu=p["k_uu"], k_cu=p["k_cu"], k_hu=p["k_hu"], l=p["l"],
                lk=p["lk"], delta=p["delta"], a_ai_h=p["a_ai_h"],
                a_e_m=p["a_e_m"], c_M=c_M, c_0=c_0)

def build_ic(p):
    return [p["q0"], 1.0, max(p["R0"], 1e-9), p["eta0"]]

# --------------------------- analytic layer (v4) ---------------------------
RGRID = np.exp(np.linspace(np.log(1e-7), np.log(1e6), 1600))

def g_on_grid(k_cu, k_uu, l, lk, delta, a, c, rs=RGRID):
    """v4 sign function g(r) = k + b r - O*(r) P S on an r grid (vectorised).
    O*(r) is the minus root of c*delta*P*O^2 - D*O + 1 = 0; NaN where the
    observability sector has no equilibrium (validity-envelope exit)."""
    b = k_cu + k_uu - 1.0
    P = lk * k_cu + l * rs
    D = 1.0 + c + (a + c * k_uu) * rs
    S = 1.0 + (1.0 - delta) * rs
    cdP = c * delta * P
    disc = D * D - 4.0 * cdP
    O = np.where(disc >= 0, 2.0 / (D + np.sqrt(np.maximum(disc, 0.0))), np.nan)
    small = cdP < 1e-14
    if np.any(small):
        O = np.where(small, 1.0 / D, O)
    return k_cu + b * rs - O * P * S

def basin_roots(k_cu, k_uu, l, lk, delta, a, c):
    """Fixed points q* (ascending) from sign changes of g on the log grid,
    refined by bisection. g(0) > 0 always (no eradication)."""
    g = g_on_grid(k_cu, k_uu, l, lk, delta, a, c)
    roots = []
    for i in range(len(RGRID) - 1):
        g0, g1 = g[i], g[i + 1]
        if np.isnan(g0) or np.isnan(g1):
            continue
        if g0 == 0.0:
            roots.append(RGRID[i])
        elif g0 * g1 < 0:
            lo, hi = RGRID[i], RGRID[i + 1]
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                gm = g_on_grid(k_cu, k_uu, l, lk, delta, a, c,
                               rs=np.array([mid]))[0]
                if np.isnan(gm):
                    break
                if gm * g0 <= 0:
                    hi = mid
                else:
                    lo = mid
            roots.append(0.5 * (lo + hi))
    return [r / (1 + r) for r in sorted(roots)]

def classify_basin(ode):
    """Returns (kind, roots): kind in escape / monostable / bistable."""
    k_uu, k_cu, l, lk, delta = (ode["k_uu"], ode["k_cu"], ode["l"],
                                ode["lk"], ode["delta"])
    a = ode["a_e_m"]; c = ode["c_0"] / ode["c_M"]
    if k_cu <= 1e-9:
        Ostar0 = 1 / (1 + c)
        roots = basin_roots(k_cu, k_uu, l, lk, delta, a, c)
        if l * Ostar0 >= k_uu - 1:
            roots = [0.0] + roots
    else:
        roots = basin_roots(k_cu, k_uu, l, lk, delta, a, c)
    roots = [q for q in roots if q < 1 - 1e-9]
    if len(roots) == 0:
        return "escape", roots
    if len(roots) == 1:
        return "monostable", roots
    return "bistable", roots

def basin_exists_grid(k_cu, l, lk, delta, a, Ostar, k_uu=1.0):
    """Fast existence check with the per-cell dial pin (vectorised in r)."""
    try:
        c = pin_c(Ostar, delta, lk, k_cu)
    except ValueError:
        return False
    g = g_on_grid(k_cu, k_uu, l, lk, delta, a, c)
    return bool(np.any(g[~np.isnan(g)] < 0))

def lstar_v4(k_cu, k_uu, delta, lk, a, Ostar, l_hi=400.0):
    """Numeric basin-existence threshold (existence is monotone in l;
    derivation-audit Part A'' V4.8). None if no basin at any l <= l_hi
    (the delta*k_cu > 4/(4+l_k) regime under the dial pin)."""
    if not basin_exists_grid(k_cu, l_hi, lk, delta, a, Ostar, k_uu):
        return None
    lo, hi = 0.0, l_hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if basin_exists_grid(k_cu, mid, lk, delta, a, Ostar, k_uu):
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)

def lstar_delta1_closed(k, c, lk):
    """delta = 1 closed form (audit V4.8): valid only when the floor exists
    (2 c delta k <= 1 + c)."""
    T = 1.0 + c
    A = T - c * k
    if A - lk < 0:
        return None
    return 2 * k * (A + np.sqrt(A * (A - lk)))

# Broad-default preset (identity-calibrated trend-adjusted k; unchanged in
# v4 — audit V4.11)
DEFAULTS = dict(T_auto=0.5, Ostar=0.5, k_uu=1.0, k_cu=0.0618, k_hu=0.0618,
                l=0.2, lk=1.0, delta=0.7, a_e_m=1.0, a_ai_h=1.0, R0=1.0,
                q0=0.05, eta0=5.0)

def P(**over):
    p = dict(DEFAULTS); p.update(over); return p

# ===========================================================================
# VALIDATION against pinned v4 ground-truth numbers
# ===========================================================================
def validate():
    print("=== validating Python port (v4 growth peg) ===")
    ok = True
    # A. named-calibration verdicts (calibrations_v4.py)
    kind, roots = classify_basin(ode_params(P()))  # Broad default
    good = kind == "escape"
    print(f"  Broad default (k=0.0618, l=0.2, l_k=1, d=0.7): {kind} "
          f"(want escape/takeover) {'ok' if good else 'MISMATCH'}")
    if not good: ok = False
    ps = P(k_cu=0.0085, k_hu=0.0085, q0=0.005)  # Strict default
    kind, roots = classify_basin(ode_params(ps))
    good = kind == "monostable" and abs(roots[0] - 0.0471) < 0.001
    print(f"  Strict default (k=0.0085): {kind}, q_u*={roots[0]:.4f} "
          f"(want 0.0471) {'ok' if good else 'MISMATCH'}")
    if not good: ok = False
    kind, _ = classify_basin(ode_params(P(k_cu=0.9)))  # AI 2027
    print(f"  AI 2027 (k_cu=0.9, l=0.2, d=0.7): {kind} (want escape) "
          f"{'ok' if kind == 'escape' else 'MISMATCH'}")
    if kind != "escape": ok = False
    # B. thresholds: numeric bisection vs v4 fixtures
    got = lstar_v4(0.0618, 1.0, 0.7, 1.0, 1.0, 0.5)
    good = got is not None and abs(got - 0.3123) < 2e-3
    print(f"  Broad default l*: {got:.4f} (want 0.3123) "
          f"{'ok' if good else 'MISMATCH'}")
    if not good: ok = False
    # delta=1 closed form agrees with bisection at Broad
    c1 = pin_c(0.5, 1.0, 1.0, 0.0618)
    want = lstar_delta1_closed(0.0618, c1, 1.0)
    got = lstar_v4(0.0618, 1.0, 1.0, 1.0, 1.0, 0.5)
    good = abs(got - want) < 2e-3 and abs(want - 0.4139) < 1e-3
    print(f"  Broad d=1 l*: bisection {got:.4f} vs closed form {want:.4f} "
          f"(want 0.4139) {'ok' if good else 'MISMATCH'}")
    if not good: ok = False
    # C. AI-2027 thresholds (calibrations-v4 section 6): finite at d=0.7,
    #    NO basin at any l at d=1 under the dial pin
    for d, lk, want in ((0.7, 1.0, 3.926), (0.7, 0.5, 3.995),
                        (1.0, 1.0, None), (1.0, 0.5, None)):
        got = lstar_v4(0.9, 1.0, d, lk, 1.0, 0.5)
        if want is None:
            good = got is None
            print(f"  l*(k_cu=0.9, d={d}, l_k={lk}) = {got} (want NO BASIN) "
                  f"{'ok' if good else 'MISMATCH'}")
        else:
            good = got is not None and abs(got - want) < 0.02
            print(f"  l*(k_cu=0.9, d={d}, l_k={lk}) = {got:.3f} (want {want}) "
                  f"{'ok' if good else 'MISMATCH'}")
        if not good: ok = False
    # D. observability cannot save: tolerable k_cu and l* at O*=0.99
    def max_kcu(Ostar, l=0.2):
        lo, hi = 1e-4, 0.5
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            if basin_exists_grid(mid, l, 1.0, 0.7, 1.0, Ostar):
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)
    k05, k99 = max_kcu(0.5), max_kcu(0.99)
    good = abs(k05 - 0.0395) < 1e-3 and abs(k99 - 0.1078) < 2e-3
    print(f"  tolerable k_cu at l=0.2: dial 0.5 -> {k05:.4f} (want 0.0395), "
          f"0.99 -> {k99:.4f} (want 0.1078) {'ok' if good else 'MISMATCH'}")
    if not good: ok = False
    lO99 = lstar_v4(0.9, 1.0, 0.7, 1.0, 1.0, 0.99)
    good = lO99 is not None and abs(lO99 - 1.674) < 0.01
    print(f"  l* at k_cu=0.9, dial 0.99, d=0.7: {lO99:.3f} (want 1.674) "
          f"{'ok' if good else 'MISMATCH'}")
    if not good: ok = False
    # E. AI 2027 trajectory landmarks (landmarks_v4.py section 9): q crosses
    #    0.5 near sigma ~ 3.1, observed peak ~ 0.178 near sigma 3.6
    pp = P(k_cu=0.9)
    sim = simulate(ode_params(pp), build_ic(pp), 12)
    s50 = next((tr["sigma"] for tr in sim["traj"] if tr["Q"] >= 0.5), None)
    obs = [tr["O"] * tr["Q"] for tr in sim["traj"]]
    pk = max(obs)
    good = s50 is not None and 3.0 <= s50 <= 3.25 and 0.17 <= pk <= 0.19
    print(f"  AI 2027 run: sigma@Q=0.5 = {s50:.2f} (want ~3.1); observed "
          f"peak {pk:.3f} (want ~0.178) {'ok' if good else 'MISMATCH'}")
    if not good: ok = False
    print("  PORT VALID" if ok else "  *** PORT INVALID ***")
    print()
    return ok

# ===========================================================================
# FIGURE 1 — AI 2027 as a high-leakage run
# ===========================================================================
def fig1():
    # single AI-2027 run: true q_u vs observed O*q_u (top), observability (bottom).
    # x-axis is total AI-dev labour in multiples of human labour, A/H0 = e^sigma/eta0.
    p = P(k_cu=0.9)
    eta0 = p["eta0"]
    sim = simulate(ode_params(p), build_ic(p), 12)
    sig = np.array([t["sigma"] for t in sim["traj"]])
    xh = np.exp(sig) / eta0
    Q = np.array([t["Q"] for t in sim["traj"]])
    O = np.array([t["O"] for t in sim["traj"]])
    obs = O * Q
    x0, x2, xmax = 1 / eta0, np.exp(2) / eta0, np.exp(12) / eta0

    fig, (axQ, axO) = plt.subplots(2, 1, figsize=(8.4, 6.0), sharex=True,
                                   gridspec_kw=dict(height_ratios=[1.55, 1]))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.10, hspace=0.13)

    # ---- top: true vs observed uncooperative behaviour ----
    axQ.axvspan(x0, x2, color=C["accent"], alpha=0.07, lw=0)
    axQ.text(x0 * 1.06, 0.955, "first ≈7× of AI-labour growth\n"
             "(AI 2027’s adversarial emergence)", ha="left", va="top",
             fontsize=8.3, color=C["accent"])
    axQ.axvline(1.0, color=C["human"], lw=1.0, ls=(0, (1, 2)), alpha=0.7)
    axQ.text(1.0, 0.02, " AI labour = human", color=C["human"], fontsize=7.6,
             rotation=90, va="bottom", ha="left")
    axQ.plot(xh, Q, color=C["uncoop"], lw=3.0, solid_capstyle="round",
             label="true uncooperative share  $q_u$")
    axQ.plot(xh, obs, color=C["accent"], lw=2.4, ls=(0, (5, 2)),
             solid_capstyle="round",
             label="observed uncooperative behaviour  $O\\,q_u$")
    axQ.axhline(0.5, color=C["fgMuted"], lw=0.8, ls=":", alpha=0.7)
    axQ.text(xmax * 0.92, 0.52, "$q_u=0.5$", ha="right", va="bottom",
             fontsize=8, color=C["fgMuted"])
    ip = int(np.argmax(obs))
    axQ.annotate("what labs can measure peaks early (~18%), then\n"
                 "falls — even as true uncooperativeness\napproaches dominance",
                 (xh[ip], obs[ip]), xytext=(xh[ip] * 7, 0.40), fontsize=8.7,
                 color=C["accent"], va="center",
                 arrowprops=dict(arrowstyle="-", color=C["accent"], lw=0.8))
    axQ.set_xscale("log")
    axQ.set_ylim(0, 1.0); axQ.set_xlim(x0, xmax)
    axQ.set_ylabel("share of total automation")
    axQ.grid(True, color=C["grid"], lw=0.7); axQ.set_axisbelow(True)
    axQ.legend(loc="upper left", bbox_to_anchor=(0.30, 0.99), frameon=True,
               facecolor=C["panel"], edgecolor=C["border"], fontsize=9)
    axQ.set_title("AI 2027 located as a high-leakage run\n"
                  "$k_{uu}{=}1,\\ \\ell{=}0.2,\\ \\ell_k{=}1,\\ \\delta{=}0.7,"
                  "\\ O^*{=}0.5$ fixed; "
                  "leakage $k_{cu}{=}0.9$ stands in for prior-driven misalignment",
                  fontsize=10.3, pad=8, loc="left")

    # ---- bottom: observability collapse ----
    axO.axvspan(x0, x2, color=C["accent"], alpha=0.07, lw=0)
    axO.axvline(1.0, color=C["human"], lw=1.0, ls=(0, (1, 2)), alpha=0.7)
    axO.plot(xh, O, color=C["coop"], lw=2.6, solid_capstyle="round")
    axO.set_xscale("log")
    axO.set_ylim(0, 0.8); axO.set_xlim(x0, xmax)
    axO.set_ylabel("observability  $O$")
    axO.set_xlabel("AI-development labour, in multiples of human labour   ($A/H_0$)")
    axO.grid(True, color=C["grid"], lw=0.7); axO.set_axisbelow(True)
    axO.text(x0 * 1.1, 0.74, "oversight holds roughly flat at first, then collapses",
             fontsize=8.7, color=C["coop"], va="top")
    fig.savefig(f"{OUT}/ai2027-high-leakage-run.png", dpi=200)
    plt.close(fig)
    print(f"wrote ai2027-high-leakage-run.png (peak obs {max(obs):.3f}, "
          f"sigma@50% {next(s for s, q in zip(sig, Q) if q >= 0.5):.2f})")

# ===========================================================================
# FIGURE 2 (LEAD) — observability alone cannot recover cooperation
# ===========================================================================
def region_grid_dial(kcu_vals, yvals, ykey, fixed):
    """Integer grid: 1 where a cooperative-side basin exists, 0 where
    takeover is the only outcome (per-cell dial pin throughout)."""
    G = np.zeros((len(yvals), len(kcu_vals)), dtype=int)
    base = dict(DEFAULTS); base.update(fixed)
    for j, y in enumerate(yvals):
        for i, k in enumerate(kcu_vals):
            kw = dict(l=base["l"], lk=base["lk"], delta=base["delta"],
                      a=base["a_e_m"], Ostar=base["Ostar"])
            kw[ykey if ykey != "Ostar" else "Ostar"] = y
            if ykey == "l":
                kw["l"] = y
            G[j, i] = 1 if basin_exists_grid(max(k, 1e-9), kw["l"], kw["lk"],
                                             kw["delta"], kw["a"],
                                             kw["Ostar"]) else 0
    return G

def fig2():
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap([C["fill_escape"], C["fill_coop"]])

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.4, 5.1))
    fig.subplots_adjust(left=0.065, right=0.985, top=0.76, bottom=0.20, wspace=0.23)

    # ---- Panel A: (k_cu, O* dial) at central l = 0.2, l_k = 1, delta = 0.7 ----
    kcu = np.linspace(1e-4, 0.20, 240)
    Ostar = np.linspace(0.5, 0.99, 160)
    axA.pcolormesh(kcu, Ostar, region_grid_dial(kcu, Ostar, "Ostar", dict()),
                   cmap=cmap, vmin=0, vmax=1, shading="auto")
    # v4 boundary: max tolerable k_cu at l = 0.2 for each dial value
    kbound = []
    for o in Ostar:
        lo, hi = 1e-4, 0.5
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if basin_exists_grid(mid, 0.2, 1.0, 0.7, 1.0, o):
                lo = mid
            else:
                hi = mid
        kbound.append(0.5 * (lo + hi))
    axA.plot(kbound, Ostar, color=C["fg"], lw=1.6, ls="--")
    print(f"  panel A tolerable leakage: O*=0.5 -> {kbound[0]:.4f}, "
          f"O*=0.99 -> {kbound[-1]:.4f}")
    axA.annotate("", xy=(0.158, 0.965), xytext=(0.158, 0.52),
                 arrowprops=dict(arrowstyle="->", color=C["fg"], lw=1.4))
    axA.text(0.163, 0.74, "perfect observability\nroughly triples leakage\n"
             "tolerance — no more", fontsize=8.8, color=C["fg"], va="center")
    axA.text(0.012, 0.93, "cooperative\nbasin", fontsize=9, color=C["coop"],
             va="top", fontweight="bold")
    axA.text(0.155, 0.58, "uncooperative\nbasin", fontsize=9, color=C["uncoop"],
             ha="center", fontweight="bold")
    axA.plot(0.0618, 0.5, "o", color=C["fg"], ms=7, zorder=6)
    axA.annotate("Broad default $k_{cu}{=}0.062$:\nalready outside at "
                 "$O^*{=}0.5$", (0.0618, 0.5),
                 xytext=(0.012, 0.62), fontsize=8.5, color=C["fg"],
                 arrowprops=dict(arrowstyle="-", color=C["fg"], lw=0.8))
    axA.annotate("AI 2027 “priors dominate” $k_{cu}{\\approx}0.9$  →  far off-scale, "
                 "deep in escape", (0.20, 0.55),
                 xytext=(0.0, 1.005), xycoords="data", textcoords="axes fraction",
                 fontsize=8.3, color=C["uncoop"])
    axA.set_xlim(0, 0.20); axA.set_ylim(0.5, 0.99)
    axA.set_xlabel("cooperative→uncooperative leakage  $k_{cu}$")
    axA.set_ylabel("best-case observability  $O^*$")
    axA.set_title("(a)  Better observability extends tolerable\n"
                  "leakage ~3× — no more   ($\\ell{=}0.2$, $\\ell_k{=}1$, "
                  "$\\delta{=}0.7$ fixed)",
                  fontsize=10.5, loc="left", pad=18)

    # ---- Panel B: (k_cu, l) at near-perfect O* = 0.99 ----
    kcu2 = np.linspace(1e-4, 1, 240)
    lvals = np.linspace(0.001, 4, 160)
    axB.pcolormesh(kcu2, lvals, region_grid_dial(kcu2, lvals, "l",
                                                 dict(Ostar=0.99)),
                   cmap=cmap, vmin=0, vmax=1, shading="auto")
    # v4 basin boundary l*(k) at the 0.99 dial
    lb = np.array([np.nan if (ls := lstar_v4(k, 1.0, 0.7, 1.0, 1.0, 0.99,
                                             l_hi=40.0)) is None else ls
                   for k in kcu2])
    axB.plot(kcu2, lb, color=C["fg"], lw=1.6, ls="--")
    l_at_09 = lstar_v4(0.9, 1.0, 0.7, 1.0, 1.0, 0.99)
    print(f"  panel B l* at k_cu=0.9, O*=0.99: {l_at_09:.3f}")
    axB.axhline(1.0, color=C["fgMuted"], lw=1.0, ls=(0, (1, 1.5)))
    axB.text(0.02, 1.05, "$\\ell=1$: even at $O^*{=}1$, no $\\ell\\leq1$ "
             "yields a basin at this leakage —\nthere is no eradication "
             "escape hatch (seeding can never be out-intercepted)",
             fontsize=8.0, color=C["fgDim"], va="bottom")
    axB.axhline(0.2, color=C["accent"], lw=1.8)
    axB.text(0.985, 0.24, "central estimate  $\\ell=0.2$", color=C["accent"],
             fontsize=8.7, va="bottom", ha="right")
    axB.text(0.07, 0.62, "cooperative\nbasin", fontsize=9.5, color=C["coop"],
             fontweight="bold")
    axB.text(0.55, 0.55, "uncooperative basin", fontsize=10, color=C["uncoop"],
             fontweight="bold")
    axB.plot(0.9, l_at_09, "s", color=C["uncoop"], ms=9, zorder=6, mec=C["fg"])
    axB.annotate("at AI 2027 leakage a basin needs $\\ell{\\approx}1.7$ even at\n"
                 "$O^*{=}0.99$ — 8–9× the central fix rate — and it\n"
                 "appears with its attractor already at a high\nuncooperative "
                 "share (~44–54% at $\\delta{=}0.7$; at $\\delta{=}1$\nno fix "
                 "rate yields a basin at all)", (0.9, l_at_09),
                 xytext=(0.26, 3.2), fontsize=8.4, color=C["uncoop"], va="top",
                 arrowprops=dict(arrowstyle="-", color=C["uncoop"], lw=0.8))
    axB.set_xlim(0, 1); axB.set_ylim(0, 4)
    axB.set_xlabel("cooperative→uncooperative leakage  $k_{cu}$")
    axB.set_ylabel("fix rate  $\\ell$")
    axB.set_title("(b)  Holding high leakage needs a fix rate scaling\n"
                  "with it — far beyond the evidence   ($O^*=0.99$, "
                  "$\\ell_k{=}1$)",
                  fontsize=10.5, loc="left", pad=18)

    legend_el = [
        Patch(fc=C["fill_coop"], ec=C["edge_coop"],
              label="a cooperative-side basin exists"),
        Patch(fc=C["fill_escape"], ec=C["edge_escape"],
              label="uncooperative dominance is the only outcome"),
        Line2D([0], [0], color=C["fg"], lw=1.6, ls="--",
               label="basin boundary $\\ell^*$ (numeric, growth-peg model)"),
    ]
    fig.legend(handles=legend_el, loc="lower center", ncol=3, frameon=False,
               fontsize=8.6, bbox_to_anchor=(0.5, 0.012))
    fig.suptitle("At high leakage, observability alone cannot recover cooperation",
                 fontsize=13.5, x=0.065, ha="left", y=0.955)
    fig.savefig(f"{OUT}/ai2027-observability-cannot-save.png", dpi=200)
    plt.close(fig)
    print("wrote ai2027-observability-cannot-save.png")

if __name__ == "__main__":
    if not validate():
        raise SystemExit("port validation failed; not drawing figures")
    fig1()
    fig2()
    print("done")
