#!/usr/bin/env python3
"""
Figures for the "representing external scenarios" (AI 2027) appendix of the
dynamical-models-of-ai-governability post.

The model core is a faithful port of basin-explorer/model.mjs (the canonical
sigma-clock model). The port is validated against ground-truth numbers emitted
by explore_ai2027.mjs before any figure is drawn.

Two figures:
  fig1: AI 2027 located as a high-leakage run (q_u rises, O collapses).
  fig2 (lead): at high leakage, observability alone cannot recover cooperation.
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
# MODEL CORE  — port of model.mjs (BETA = 1, linear observability)
# ===========================================================================
def computeO(m, e):
    m = max(m, 1e-12); e = max(e, 1e-12)
    return m / (m + e)

def make_deriv(p):
    def f(s):
        Q, m, e, eta = s
        O = computeO(m, e)
        Fc = ((1 - Q) * (1 - p["k_cu"]) + eta * (1 - p["k_hu"])
              + (1 - p["delta"]) * p["l"] * O * Q)
        Fu = p["k_cu"] * (1 - Q) + p["k_hu"] * eta + (p["k_uu"] - p["l"] * O) * Q
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
    return (1 - Q) + eta + (p["k_uu"] - p["delta"] * p["l"] * O) * Q

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
    Q, eta = p["q0"], p["eta0"]
    R0 = max(p["R0"], 1e-9)
    O0 = 1 / (1 + R0)
    m0 = 1.0
    G0 = max((1 - Q) + eta + (p["k_uu"] - p["delta"] * p["l"] * O0) * Q, 0.05)
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

def classify_basin(ode):
    """Delta-general quadratic classifier in r = q_u/(1-q_u).

    Fixed points solve A_d r^2 + B r + C = 0 with
      b = k_cu + k_uu - 1, a = a_e_m, c = c_0/c_M,
      A_d = b(a+c) - (1-delta) l,  B = k_cu(a+c) + b(1+c) - l,  C = k_cu(1+c).
    (See verify-delta-fixed-points.js R3/R4; reduces to the post's delta=1 form.)
    """
    k_uu, k_cu, l, delta = ode["k_uu"], ode["k_cu"], ode["l"], ode["delta"]
    a = ode["a_e_m"]; c = ode["c_0"] / ode["c_M"]
    b = k_cu + k_uu - 1
    A = b * (a + c) - (1 - delta) * l
    B = k_cu * (a + c) + b * (1 + c) - l
    Cc = k_cu * (1 + c)
    if abs(A) < 1e-12:
        roots_r = [] if abs(B) < 1e-12 else [-Cc / B]
    else:
        disc = B * B - 4 * A * Cc
        roots_r = [] if disc < 0 else [(-B + np.sqrt(disc)) / (2 * A),
                                       (-B - np.sqrt(disc)) / (2 * A)]
    roots = sorted(r / (1 + r) for r in roots_r if r > 1e-12)
    roots = [q for q in roots if q < 1 - 1e-9]
    if k_cu <= 1e-9:
        Ostar0 = 1 / (1 + c)
        if l * Ostar0 >= k_uu - 1:
            roots = [0.0] + roots
    if len(roots) == 0:
        return "escape", roots
    if len(roots) == 1:
        return "monostable", roots
    return "bistable", roots

DEFAULTS = dict(T_auto=0.5, Ostar=0.5, k_uu=1.0, k_cu=0.05, k_hu=0.05, l=0.4,
                delta=0.7, a_e_m=1.0, a_ai_h=1.0, R0=1.0, q0=0.05, eta0=5.0)

def P(**over):
    p = dict(DEFAULTS); p.update(over); return p

# ===========================================================================
# VALIDATION against explore_ai2027.mjs ground truth
# ===========================================================================
def validate():
    print("=== validating Python port (delta-general) ===")
    ok = True
    # A. delta=1, l=0.2 reproduces the pre-delta JS ground truth
    expect = {0.0: "monostable", 0.025: "bistable", 0.05: "escape",
              0.3: "escape", 0.95: "escape"}
    for k, want in expect.items():
        got, _ = classify_basin(ode_params(P(k_cu=k, l=0.2, delta=1.0)))
        flag = "ok" if got == want else "MISMATCH"
        if got != want: ok = False
        print(f"  [d=1,l=0.2] basin k_cu={k:<5}: {got:<11} (want {want}) {flag}")
    # B. sigma to Q=0.5 for k_cu=0.3 (delta=1, l=0.2) should be ~4.88
    pp = P(k_cu=0.3, l=0.2, delta=1.0)
    sim = simulate(ode_params(pp), build_ic(pp), 20)
    s50 = next((tr["sigma"] for tr in sim["traj"] if tr["Q"] >= 0.5), None)
    print(f"  sigma@Q=0.5 (k_cu=0.3): {s50:.2f} (want ~4.88) "
          f"{'ok' if abs(s50-4.88)<0.1 else 'MISMATCH'}")
    if abs(s50 - 4.88) >= 0.1: ok = False
    # C. boundary table at delta=1, l=0.2 (old regression row)
    def max_coop(l, Ostar, delta):
        best = 0.0
        for k in np.arange(0, 1.0001, 0.005):
            kind, _ = classify_basin(ode_params(P(k_cu=k, l=l, Ostar=Ostar,
                                                  delta=delta)))
            if kind != "escape": best = k
        return best
    row = [round(max_coop(0.2, o, 1.0), 3) for o in (0.5, 0.7, 0.9, 0.99)]
    want_row = [0.025, 0.035, 0.045, 0.045]
    print(f"  [d=1] max-coop k_cu @ l=0.2 over Ostar: {row} (want {want_row}) "
          f"{'ok' if np.allclose(row, want_row, atol=0.011) else 'MISMATCH'}")
    if not np.allclose(row, want_row, atol=0.011): ok = False
    # D. delta-general verdicts vs calibration-verdicts.js (verified numbers)
    kind, roots = classify_basin(ode_params(P()))  # Broad central
    good = kind == "monostable" and abs(roots[0] - 0.323) < 0.005
    print(f"  Broad (l=0.4, d=0.7): {kind}, q_u*={roots[0]:.3f} (want 0.323) "
          f"{'ok' if good else 'MISMATCH'}")
    if not good: ok = False
    ps = P(k_cu=0.005, k_hu=0.005, q0=0.005)  # Strict central
    kind, roots = classify_basin(ode_params(ps))
    good = kind == "monostable" and abs(roots[0] - 0.0255) < 0.002
    print(f"  Strict (l=0.4, d=0.7): {kind}, q_u*={roots[0]:.4f} (want 0.0255) "
          f"{'ok' if good else 'MISMATCH'}")
    if not good: ok = False
    kind, _ = classify_basin(ode_params(P(k_cu=0.9)))  # AI 2027
    print(f"  AI 2027 (k_cu=0.9, l=0.4, d=0.7): {kind} (want escape) "
          f"{'ok' if kind == 'escape' else 'MISMATCH'}")
    if kind != "escape": ok = False
    # E. piecewise threshold l*O* = 4 d k_cu (d >= 1/2) via bisection on existence
    def exists(l):
        kind, _ = classify_basin(ode_params(P(l=l)))
        return kind != "escape"
    lo, hi = 0.01, 2.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if exists(mid): hi = mid
        else: lo = mid
    want = 4 * 0.7 * 0.05 / 0.5  # = 0.28
    print(f"  Broad basin threshold l* = {hi:.4f} (want {want:.4f}) "
          f"{'ok' if abs(hi - want) < 2e-3 else 'MISMATCH'}")
    if abs(hi - want) >= 2e-3: ok = False
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
    axQ.annotate("what labs can measure peaks early, then\n"
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
    """Return integer code grid: 0 escape, 1 bistable, 2 monostable."""
    code = {"escape": 0, "bistable": 1, "monostable": 2}
    G = np.zeros((len(yvals), len(kcu_vals)), dtype=int)
    for j, y in enumerate(yvals):
        for i, k in enumerate(kcu_vals):
            over = dict(fixed); over["k_cu"] = k; over[ykey] = y
            kind, _ = classify_basin(ode_params(P(**over)))
            G[j, i] = code[kind]
    return G

def fig2():
    from matplotlib.colors import ListedColormap
    # two categories: escape (red) vs cooperative basin can exist (green)
    cmap = ListedColormap([C["fill_escape"], C["fill_coop"]])

    def exists_grid(kcu_vals, yvals, ykey, fixed):
        G = region_grid(kcu_vals, yvals, ykey, fixed)
        return (G >= 1).astype(int)   # 0 escape, 1 basin-exists

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.4, 5.1))
    fig.subplots_adjust(left=0.065, right=0.985, top=0.76, bottom=0.20, wspace=0.23)

    # ---- Panel A: (k_cu, O*) at central l = 0.4, delta = 0.7 ----
    kcu = np.linspace(0, 0.20, 500)
    Ostar = np.linspace(0.5, 0.99, 300)
    axA.pcolormesh(kcu, Ostar, exists_grid(kcu, Ostar, "Ostar", dict()),
                   cmap=cmap, vmin=0, vmax=1, shading="auto")
    # delta-general boundary at a=1, k_uu=1, delta >= 1/2: k_cu = l O* / (4 delta)
    axA.plot(0.4 * Ostar / (4 * 0.7), Ostar, color=C["fg"], lw=1.6, ls="--")
    # observability axis does almost nothing: arrow straight up at fixed k_cu
    axA.annotate("", xy=(0.158, 0.965), xytext=(0.158, 0.52),
                 arrowprops=dict(arrowstyle="->", color=C["fg"], lw=1.4))
    axA.text(0.163, 0.74, "perfect observability\nbuys you almost no\nextra leakage "
             "tolerance", fontsize=8.8, color=C["fg"], va="center")
    axA.text(0.012, 0.93, "cooperative\nbasin", fontsize=9, color=C["coop"],
             va="top", fontweight="bold")
    axA.text(0.125, 0.58, "uncooperative\nbasin", fontsize=9, color=C["uncoop"],
             ha="center", fontweight="bold")
    axA.plot(0.05, 0.5, "o", color=C["fg"], ms=7, zorder=6)
    axA.annotate("central (Broad) $k_{cu}{=}0.05$:\ninside, but with thin margin",
                 (0.05, 0.5), xytext=(0.012, 0.60), fontsize=8.5, color=C["fg"],
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
    axB.pcolormesh(kcu2, lvals, exists_grid(kcu2, lvals, "l", dict(Ostar=0.99)),
                   cmap=cmap, vmin=0, vmax=1, shading="auto")
    # boundary at a=1, k_uu=1, delta=0.7: l = 4 delta k_cu (1+c), c = (1-O*)/O*
    axB.plot(kcu2, 4 * 0.7 * kcu2 * (1 + 0.0101), color=C["fg"], lw=1.6, ls="--")
    axB.axhline(0.4, color=C["accent"], lw=1.8)
    axB.text(0.985, 0.46, "central estimate  $\\ell=0.4$", color=C["accent"],
             fontsize=8.7, va="bottom", ha="right")
    axB.text(0.13, 2.05, "cooperative\nbasin", fontsize=9.5, color=C["coop"],
             fontweight="bold")
    axB.text(0.62, 0.62, "uncooperative basin", fontsize=10, color=C["uncoop"],
             fontweight="bold")
    axB.plot(0.9, 2.55, "s", color=C["uncoop"], ms=9, zorder=6, mec=C["fg"])
    axB.annotate("AI 2027 leakage needs\n$\\ell{\\approx}2.5$ — ~6× the central\n"
                 "estimate, even at $O^*{=}0.99$", (0.9, 2.55),
                 xytext=(0.36, 3.5), fontsize=8.6, color=C["uncoop"], va="top",
                 arrowprops=dict(arrowstyle="-", color=C["uncoop"], lw=0.8))
    axB.set_xlim(0, 1); axB.set_ylim(0, 4)
    axB.set_xlabel("cooperative→uncooperative leakage  $k_{cu}$")
    axB.set_ylabel("suppression strength  $\\ell$")
    axB.set_title("(b)  Holding high leakage needs suppression\n"
                  "scaling with it   (observability near-perfect, $O^*=0.99$)",
                  fontsize=10.5, loc="left", pad=18)

    legend_el = [
        Patch(fc=C["fill_coop"], ec=C["edge_coop"],
              label="a cooperative basin can exist"),
        Patch(fc=C["fill_escape"], ec=C["edge_escape"],
              label="uncooperative basin — dominance is the only outcome"),
        Line2D([0], [0], color=C["fg"], lw=1.6, ls="--", label="basin boundary"),
    ]
    fig.legend(handles=legend_el, loc="lower center", ncol=3, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 0.012))
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
