#!/usr/bin/env python3
"""
Figures for the "AI 2027 in this model" section of the
dynamical-models-of-ai-governability post.

The model core is a port of the PRODUCTION-GATED delta-general sigma-clock
model (in-app engine: basin-explorer/src/BasinExplorer.jsx, commit b5d1952+):

  F_u = (1-lO)(k_cu q_c + k_hu q_h) + (k_uu - lO) q_u
  F_c = (1-k_cu) q_c + (1-k_hu) q_h + (1-d) lO L,  L = k_cu q_c + k_hu q_h + q_u
  G   = q_c + q_h + k_uu q_u - d lO L

The port is validated against pinned gated ground-truth numbers before any
figure is drawn (fixtures from _scratch/review/scripts/gated-calibrations.js
and verify-gated-fixed-points.js; the old pre-gating self-tests were
re-baselined by design when the model changed).

Two figures:
  fig1: AI 2027 located as a high-leakage run (q_u rises, O collapses).
  fig2 (lead): at high leakage, observability alone cannot recover cooperation;
        under gating the only cooperative destination is the eradication regime.
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
# MODEL CORE — production-gated port of the in-app engine (src/BasinExplorer.jsx)
# ===========================================================================
def computeO(m, e):
    m = max(m, 1e-12); e = max(e, 1e-12)
    return m / (m + e)

def make_deriv(p):
    def f(s):
        Q, m, e, eta = s
        O = computeO(m, e)
        Lk = p["k_cu"] * (1 - Q) + p["k_hu"] * eta   # leakage inflow
        L = Lk + Q                                    # full suppression target
        Fc = ((1 - Q) * (1 - p["k_cu"]) + eta * (1 - p["k_hu"])
              + (1 - p["delta"]) * p["l"] * O * L)
        Fu = (1 - p["l"] * O) * Lk + (p["k_uu"] - p["l"] * O) * Q
        G = Fc + Fu
        if G <= 1e-10:
            return np.array([0.0, 0.0, 0.0, -eta])
        FM = p["c_M"] * ((1 - Q) + eta / p["a_ai_h"])
        FE = p["c_0"] + p["a_e_m"] * p["c_M"] * Q
        return np.array([Fu / G - Q, FM / G - m, FE / G - e, -eta])
    return f

def computeG(s, p):
    Q, m, e, eta = s
    O = computeO(m, e)
    L = p["k_cu"] * (1 - Q) + p["k_hu"] * eta + Q
    return (1 - Q) + eta + p["k_uu"] * Q - p["delta"] * p["l"] * O * L

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
    """c_M, c_0 from O(0)=1/(1+R0), m0=1, with the gated G0 (matches the app)."""
    Q, eta = p["q0"], p["eta0"]
    R0 = max(p["R0"], 1e-9)
    O0 = 1 / (1 + R0)
    m0 = 1.0
    L0 = p["k_cu"] * (1 - Q) + p["k_hu"] * eta + Q
    G0 = max((1 - Q) + eta + p["k_uu"] * Q - p["delta"] * p["l"] * O0 * L0, 0.05)
    mon0 = max((1 - Q) + eta / p["a_ai_h"], 1e-9)
    Ostar = min(max(p["Ostar"], 1e-6), 1 - 1e-6)
    c_M = G0 * m0 / mon0
    c_0 = ((1 - Ostar) / Ostar) * c_M
    return c_M, c_0

def ode_params(p):
    c_M, c_0 = calibrated_rates(p)
    return dict(k_uu=p["k_uu"], k_cu=p["k_cu"], k_hu=p["k_hu"], l=p["l"],
                delta=p["delta"], a_ai_h=p["a_ai_h"], a_e_m=p["a_e_m"],
                c_M=c_M, c_0=c_0)

def build_ic(p):
    return [p["q0"], 1.0, max(p["R0"], 1e-9), p["eta0"]]

def lstar_gated(k_cu, k_uu, delta, a, c):
    """Gated basin-existence threshold l* (audit G4/G6; port of the app's
    BASIN_BOUNDARY.lstar / gated-calibrations.js lstarGated). Valid as a
    conventional-basin threshold only while l* < 1+c (else the eradication
    regime arrives first)."""
    b = k_cu + k_uu - 1
    if b < 0 or (abs(b) < 1e-15 and delta < 1):
        return 0.0
    P0 = k_cu * (a + c) + b * (1 + c)
    betaB = 1 + (1 - delta) * k_cu
    lA = b * (a + c) / (1 - delta) if delta < 1 else np.inf
    PB = P0 / betaB
    u = 1 - (1 - delta) * k_cu
    u2 = max(u * u, 1e-18)
    N = abs(k_cu * (a + c) - b * (1 + c))
    V = P0 * betaB - 2 * b * (a + c) * k_cu - 2 * (1 - delta) * k_cu * (1 + c)
    disc = V * V - u2 * N * N
    lPlus = (V + np.sqrt(disc)) / u2 if disc >= 0 else -np.inf
    return min(lA, max(PB, lPlus))

def classify_basin(ode):
    """Gated quadratic classifier in r = q_u/(1-q_u) (audit G3/G9).

    Fixed points solve A r^2 + B r + C = 0 with
      b = k_cu + k_uu - 1, a = a_e_m, c = c_0/c_M,
      A = b(a+c) - (1-d)l              (unchanged by gating)
      B = k_cu(a+c) + b(1+c) - l(1+(1-d)k_cu)
      C = k_cu(1+c-l)                  (gated seeding; delta-free)
    C <= 0 (lO*(0) >= 1) is the eradication regime: q_u -> 0 is reachable.
    Returns (kind, roots): kind in escape / monostable / bistable / eradication.
    """
    k_uu, k_cu, l, delta = ode["k_uu"], ode["k_cu"], ode["l"], ode["delta"]
    a = ode["a_e_m"]; c = ode["c_0"] / ode["c_M"]
    b = k_cu + k_uu - 1
    A = b * (a + c) - (1 - delta) * l
    B = k_cu * (a + c) + b * (1 + c) - l * (1 + (1 - delta) * k_cu)
    Cc = k_cu * (1 + c - l)
    if abs(A) < 1e-12:
        roots_r = [] if abs(B) < 1e-12 else [-Cc / B]
    else:
        disc = B * B - 4 * A * Cc
        roots_r = [] if disc < 0 else [(-B + np.sqrt(disc)) / (2 * A),
                                       (-B - np.sqrt(disc)) / (2 * A)]
    roots = sorted(r / (1 + r) for r in roots_r if r > 1e-12)
    roots = [q for q in roots if q < 1 - 1e-9]
    if k_cu > 1e-9 and Cc <= 0:
        # eradication regime: g(0) <= 0, q_u = 0 reachable in finite time
        return "eradication", roots
    if k_cu <= 1e-9:
        Ostar0 = 1 / (1 + c)
        if l * Ostar0 >= k_uu - 1:
            roots = [0.0] + roots
    if len(roots) == 0:
        return "escape", roots
    if len(roots) == 1:
        return "monostable", roots
    return "bistable", roots

# Broad-default preset (identity-calibrated trend-adjusted k; audit G12)
DEFAULTS = dict(T_auto=0.5, Ostar=0.5, k_uu=1.0, k_cu=0.0407, k_hu=0.0407,
                l=0.4, delta=0.7, a_e_m=1.0, a_ai_h=1.0, R0=1.0, q0=0.05,
                eta0=5.0)

def P(**over):
    p = dict(DEFAULTS); p.update(over); return p

# ===========================================================================
# VALIDATION against pinned gated ground-truth numbers
# ===========================================================================
def validate():
    print("=== validating Python port (production-gated) ===")
    ok = True
    # A. named-calibration verdicts (gated-calibrations.js)
    kind, roots = classify_basin(ode_params(P()))  # Broad default
    good = kind == "monostable" and abs(roots[0] - 0.2053) < 0.002
    print(f"  Broad default (k=0.0407, l=0.4, d=0.7): {kind}, "
          f"q_u*={roots[0]:.4f} (want 0.2053) {'ok' if good else 'MISMATCH'}")
    if not good: ok = False
    ps = P(k_cu=0.00542, k_hu=0.00542, q0=0.005)  # Strict default
    kind, roots = classify_basin(ode_params(ps))
    good = kind == "monostable" and abs(roots[0] - 0.0222) < 0.001
    print(f"  Strict default (k=0.00542): {kind}, q_u*={roots[0]:.4f} "
          f"(want 0.0222) {'ok' if good else 'MISMATCH'}")
    if not good: ok = False
    kind, _ = classify_basin(ode_params(P(k_cu=0.9)))  # AI 2027
    print(f"  AI 2027 (k_cu=0.9, l=0.4, d=0.7): {kind} (want escape) "
          f"{'ok' if kind == 'escape' else 'MISMATCH'}")
    if kind != "escape": ok = False
    # B. gated thresholds: closed form vs bisection on classify
    def exists(l, **over):
        kind, _ = classify_basin(ode_params(P(l=l, **over)))
        return kind in ("monostable", "bistable")
    def bisect_lstar(**over):
        lo, hi = 0.001, 1.9   # stay below the eradication line at O*=0.5
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if exists(mid, **over): hi = mid
            else: lo = mid
        return hi
    got = bisect_lstar()
    want = lstar_gated(0.0407, 1.0, 0.7, 1.0, 1.0)
    good = abs(got - want) < 2e-3 and abs(want - 0.2241) < 1e-3
    print(f"  Broad default l*: bisection {got:.4f} vs closed form {want:.4f} "
          f"(want 0.2241) {'ok' if good else 'MISMATCH'}")
    if not good: ok = False
    got = bisect_lstar(k_cu=0.05, k_hu=0.05, delta=1.0)
    good = abs(got - 0.38) < 2e-3   # delta=1 rule: 8 k (1-k) = 0.38
    print(f"  Broad naive d=1 l* = {got:.4f} (want 0.3800 = 4k(1-k)/O*) "
          f"{'ok' if good else 'MISMATCH'}")
    if not good: ok = False
    # C. B=0 branch at AI 2027 leakage (audit G6 table)
    for d, want in ((0.7, 2.8346), (1.0, 3.60)):
        got = lstar_gated(0.9, 1.0, d, 1.0, 1.0)
        good = abs(got - want) < 0.01
        print(f"  l*(k_cu=0.9, d={d}) = {got:.3f} (want {want}; beyond "
              f"eradication line l_E=2) {'ok' if good else 'MISMATCH'}")
        if not good: ok = False
    # D. eradication regime classification + trajectory
    kind, roots = classify_basin(ode_params(P(k_cu=0.9, l=2.5)))
    good = kind == "eradication" and len(roots) == 1 and abs(roots[0] - 0.326) < 0.01
    print(f"  k_cu=0.9, l=2.5: {kind}, separator q={roots[0]:.3f} (want "
          f"eradication, ~0.326) {'ok' if good else 'MISMATCH'}")
    if not good: ok = False
    pp = P(k_cu=0.9, l=2.5, q0=0.02)
    sim = simulate(ode_params(pp), build_ic(pp), 12)
    qf = sim["final"]["Q"]
    good = qf < 1e-3
    print(f"  eradication trajectory from q0=0.02: q(end)={qf:.5f} (want -> 0) "
          f"{'ok' if good else 'MISMATCH'}")
    if not good: ok = False
    # E. AI 2027 trajectory landmark (gated): q crosses 0.5 at sigma ~ 3.0
    pp = P(k_cu=0.9)
    sim = simulate(ode_params(pp), build_ic(pp), 12)
    s50 = next((tr["sigma"] for tr in sim["traj"] if tr["Q"] >= 0.5), None)
    obs = [tr["O"] * tr["Q"] for tr in sim["traj"]]
    pk = max(obs)
    good = s50 is not None and abs(s50 - 3.02) < 0.08 and abs(pk - 0.236) < 0.01
    print(f"  AI 2027 run: sigma@Q=0.5 = {s50:.2f} (want ~3.02); observed peak "
          f"{pk:.3f} (want ~0.236) {'ok' if good else 'MISMATCH'}")
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
    axQ.annotate("what labs can measure peaks early (~24%), then\n"
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
                  "$k_{uu}{=}1,\\ \\ell{=}0.4,\\ \\delta{=}0.7,\\ O^*{=}0.5$ fixed; "
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
    axO.text(x0 * 1.1, 0.74, "oversight gains ground early, then collapses",
             fontsize=8.7, color=C["coop"], va="top")
    fig.savefig(f"{OUT}/ai2027-high-leakage-run.png", dpi=200)
    plt.close(fig)
    print("wrote ai2027-high-leakage-run.png")

# ===========================================================================
# FIGURE 2 (LEAD) — observability alone cannot recover cooperation
# ===========================================================================
def region_grid(kcu_vals, yvals, ykey, fixed):
    """Integer grid: 1 where some cooperative outcome exists (conventional
    basin or eradication regime), 0 where takeover is the only outcome."""
    G = np.zeros((len(yvals), len(kcu_vals)), dtype=int)
    for j, y in enumerate(yvals):
        for i, k in enumerate(kcu_vals):
            over = dict(fixed); over["k_cu"] = k; over[ykey] = y
            kind, _ = classify_basin(ode_params(P(**over)))
            G[j, i] = 0 if kind == "escape" else 1
    return G

def fig2():
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap([C["fill_escape"], C["fill_coop"]])

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.4, 5.1))
    fig.subplots_adjust(left=0.065, right=0.985, top=0.76, bottom=0.20, wspace=0.23)

    # ---- Panel A: (k_cu, O*) at central l = 0.4, delta = 0.7 ----
    kcu = np.linspace(0, 0.20, 500)
    Ostar = np.linspace(0.5, 0.99, 300)
    axA.pcolormesh(kcu, Ostar, region_grid(kcu, Ostar, "Ostar", dict()),
                   cmap=cmap, vmin=0, vmax=1, shading="auto")
    # gated boundary: invert l*(k; c) = 0.4 in k by bisection for each O*
    kbound = []
    for o in Ostar:
        cc = (1 - o) / o
        lo, hi = 1e-4, 0.5
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            if lstar_gated(mid, 1.0, 0.7, 1.0, cc) > 0.4: hi = mid
            else: lo = mid
        kbound.append(0.5 * (lo + hi))
    axA.plot(kbound, Ostar, color=C["fg"], lw=1.6, ls="--")
    # observability axis does little: arrow straight up at fixed k_cu
    axA.annotate("", xy=(0.158, 0.965), xytext=(0.158, 0.52),
                 arrowprops=dict(arrowstyle="->", color=C["fg"], lw=1.4))
    axA.text(0.163, 0.74, "perfect observability\nroughly doubles leakage\n"
             "tolerance — no more", fontsize=8.8, color=C["fg"], va="center")
    axA.text(0.012, 0.93, "cooperative\nbasin", fontsize=9, color=C["coop"],
             va="top", fontweight="bold")
    axA.text(0.125, 0.58, "uncooperative\nbasin", fontsize=9, color=C["uncoop"],
             ha="center", fontweight="bold")
    axA.plot(0.0407, 0.5, "o", color=C["fg"], ms=7, zorder=6)
    axA.annotate("Broad default $k_{cu}{=}0.041$:\ninside, modest margin",
                 (0.0407, 0.5), xytext=(0.012, 0.60), fontsize=8.5, color=C["fg"],
                 arrowprops=dict(arrowstyle="-", color=C["fg"], lw=0.8))
    axA.annotate("AI 2027 “priors dominate” $k_{cu}{\\approx}0.9$  →  far off-scale, "
                 "deep in escape", (0.20, 0.55),
                 xytext=(0.0, 1.005), xycoords="data", textcoords="axes fraction",
                 fontsize=8.3, color=C["uncoop"])
    axA.set_xlim(0, 0.20); axA.set_ylim(0.5, 0.99)
    axA.set_xlabel("cooperative→uncooperative leakage  $k_{cu}$")
    axA.set_ylabel("best-case observability  $O^*$")
    axA.set_title("(a)  Better observability barely extends\n"
                  "tolerable leakage   ($\\ell=0.4$, $\\delta=0.7$ fixed)",
                  fontsize=10.5, loc="left", pad=18)

    # ---- Panel B: (k_cu, l) at near-perfect O* = 0.99 ----
    kcu2 = np.linspace(0, 1, 500)
    lvals = np.linspace(0, 4, 300)
    cB = (1 - 0.99) / 0.99
    lE = 1 + cB  # eradication line l O*(0) = 1
    axB.pcolormesh(kcu2, lvals, region_grid(kcu2, lvals, "l", dict(Ostar=0.99)),
                   cmap=cmap, vmin=0, vmax=1, shading="auto")
    # gated basin boundary l*(k), drawn only where it undercuts the eradication line
    lb = np.array([lstar_gated(k, 1.0, 0.7, 1.0, cB) for k in kcu2])
    msk = lb <= lE
    axB.plot(kcu2[msk], lb[msk], color=C["fg"], lw=1.6, ls="--")
    # eradication line
    axB.axhline(lE, color=C["uncoop"], lw=1.4, ls=(0, (1, 1.5)))
    axB.text(0.02, lE + 0.06, "eradication line  $\\ell\\,O^*(0)=1$: above it, "
             "suppression intercepts more\nthan all seeding and $q_u\\to0$ "
             "(narrow basin at high $k_{cu}$)", fontsize=8.0,
             color=C["uncoop"], va="bottom")
    axB.axhline(0.4, color=C["accent"], lw=1.8)
    axB.text(0.985, 0.44, "central estimate  $\\ell=0.4$", color=C["accent"],
             fontsize=8.7, va="bottom", ha="right")
    axB.text(0.10, 0.62, "cooperative\nbasin", fontsize=9.5, color=C["coop"],
             fontweight="bold")
    axB.text(0.62, 0.30, "uncooperative basin", fontsize=10, color=C["uncoop"],
             fontweight="bold")
    axB.plot(0.9, lE, "s", color=C["uncoop"], ms=9, zorder=6, mec=C["fg"])
    axB.annotate("at AI 2027 leakage a conventional basin would need\n"
                 "$\\ell{\\approx}1.4$ — but the eradication line arrives first "
                 "($\\ell{\\approx}1.0$):\nthe only cooperative outcome is "
                 "over-suppression,\n~2.5× the central estimate even at "
                 "$O^*{=}0.99$", (0.9, lE),
                 xytext=(0.30, 2.9), fontsize=8.6, color=C["uncoop"], va="top",
                 arrowprops=dict(arrowstyle="-", color=C["uncoop"], lw=0.8))
    axB.set_xlim(0, 1); axB.set_ylim(0, 4)
    axB.set_xlabel("cooperative→uncooperative leakage  $k_{cu}$")
    axB.set_ylabel("suppression strength  $\\ell$")
    axB.set_title("(b)  Holding high leakage needs suppression scaling\n"
                  "with it — into over-suppression   ($O^*=0.99$)",
                  fontsize=10.5, loc="left", pad=18)

    legend_el = [
        Patch(fc=C["fill_coop"], ec=C["edge_coop"],
              label="a cooperative outcome can exist (basin, or eradication beyond the dotted line)"),
        Patch(fc=C["fill_escape"], ec=C["edge_escape"],
              label="uncooperative dominance is the only outcome"),
        Line2D([0], [0], color=C["fg"], lw=1.6, ls="--",
               label="gated basin boundary $\\ell^*$"),
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
