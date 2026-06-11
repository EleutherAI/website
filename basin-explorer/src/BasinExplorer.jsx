import React, { useState, useMemo, useEffect, useRef, createContext, useContext } from 'react';
import {
  LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
  ReferenceLine, ResponsiveContainer, Legend
} from 'recharts';

// =============================================================================
// FONTS + KATEX LOADING
// =============================================================================
const FONT_LINK_ID = '__basin_explorer_fonts__';
function useFonts() {
  useEffect(() => {
    if (document.getElementById(FONT_LINK_ID)) return;
    const link = document.createElement('link');
    link.id = FONT_LINK_ID;
    link.rel = 'stylesheet';
    link.href = 'https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Serif:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap';
    document.head.appendChild(link);
  }, []);
}

const KaTeXContext = createContext(false);

function useKaTeXLoader() {
  const [ready, setReady] = useState(
    typeof window !== 'undefined' && !!window.katex
  );
  useEffect(() => {
    if (ready) return;
    if (typeof window === 'undefined') return;
    if (window.katex) { setReady(true); return; }

    if (!document.getElementById('__katex_css__')) {
      const css = document.createElement('link');
      css.id = '__katex_css__';
      css.rel = 'stylesheet';
      css.href = 'https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css';
      document.head.appendChild(css);
    }

    let cancelled = false;
    const existing = document.getElementById('__katex_js__');
    if (!existing) {
      const js = document.createElement('script');
      js.id = '__katex_js__';
      js.src = 'https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.js';
      js.onload = () => { if (!cancelled) setReady(true); };
      document.head.appendChild(js);
    } else {
      const id = setInterval(() => {
        if (window.katex) { setReady(true); clearInterval(id); }
      }, 50);
      return () => { cancelled = true; clearInterval(id); };
    }
    return () => { cancelled = true; };
  }, [ready]);
  return ready;
}

function TeX({ children, display = false }) {
  const ready = useContext(KaTeXContext);
  const ref = useRef(null);
  useEffect(() => {
    if (!ref.current) return;
    const tex = typeof children === 'string' ? children : String(children);
    if (ready && window.katex) {
      try {
        window.katex.render(tex, ref.current, {
          displayMode: display,
          throwOnError: false,
          output: 'html',
        });
      } catch {
        ref.current.textContent = tex;
      }
    } else {
      ref.current.textContent = tex;
    }
  }, [children, display, ready]);
  return display
    ? <div ref={ref} style={{ margin: '10px 0', overflowX: 'auto', overflowY: 'hidden' }} />
    : <span ref={ref} style={{ display: 'inline-block' }} />;
}

const FONTS = {
  serif: '"IBM Plex Serif", Georgia, serif',
  sans: '"IBM Plex Sans", system-ui, sans-serif',
  mono: '"IBM Plex Mono", "SF Mono", Menlo, monospace',
};

// Light paper-feel palette
const C = {
  bg: '#faf7ef',
  bgPanel: '#f0ebdc',
  bgInset: '#fdfbf4',
  border: '#d8d0bb',
  borderStrong: '#a89c80',
  fg: '#1d1c17',
  fgDim: '#5a5448',
  fgMuted: '#8a8270',
  accent: '#a85d1f',
  cooperative: '#3a7a4e',
  escape: '#a83b2e',
  bistable: '#a85d1f',
  monostable: '#4a6075',
  trace_q: '#a83b2e',
  trace_O: '#3a7a4e',
  apparent: '#2f6f8f',
  share_coop: '#3a7a4e',
  share_uncoop: '#a83b2e',
  share_human: '#6f4585',
  grid: '#e3dcc6',
  testpath: '#1d1c17',
};

// =============================================================================
// MATH — σ-clock ODE (bounded-gate suppression + growth-pegged passive
// opacity, v4)
// =============================================================================
//
// Observed misbehaviour begets fixes applied to the whole production pipeline,
// so the accumulated detect-and-fix stock acts twice. It removes established
// uncooperative labour from the q_u stock at the observability-gated rate ℓO
// (ℓ = fix rate of known failure modes per σ-unit — a RATE, may exceed 1),
// and it intercepts the LEAKAGE INFLOW before it enters the pool with bounded
// efficacy ℓ_k·O (ℓ_k ∈ [0,1] = interception efficacy at production — a
// FRACTION of the inflow: interception cannot remove more leakage than
// exists). With the leakage inflow L_k = k_cu(1-Q) + k_hu·η, only the
// fraction (1 - ℓ_k·O) of leaked effort enters the pool. Of the intercepted
// flow O·(ℓ_k·L_k + ℓ·Q), fraction δ is destroyed and (1−δ) is redirected
// into cooperative production. Leakage is therefore not a pure transfer:
// the destroyed part of intercepted leakage, δO·ℓ_k·L_k, subtracts from G
// (2–3% of G at the Broad central).
//
//   F_c = (1-Q)(1-k_cu) + η(1-k_hu) + (1-δ)·O·(ℓ_k·L_k + ℓ·Q)
//   F_u = (1-ℓ_k·O)·L_k + (k_uu - ℓO)·Q
//   G   = F_c + F_u = (1-Q) + η + k_uu·Q - δ·O·(ℓ_k·L_k + ℓ·Q)
//
//   F_M = c_M[(1-Q) + η/a_AI/H]     (detect-and-fix coverage production)
//   F_E = c_0·G + a_E/M·c_M·Q       (v4: passive opacity is pegged to
//                                    automation GROWTH — dE_passive = c_0·dA —
//                                    while active evasion stays a labour share)
//
//   Q' = F_u/G - Q
//   η' = -η                         (η here is the blog post's q_h)
//   m' = F_M/G - m
//   e' = F_E/G - e = c_0 + a_E/M·c_M·Q/G - e
//   O  = m^β / (m^β + e^β)
//
// The v4 change is the one F_E term: every new unit of automation arrives
// carrying c_0 units of not-yet-covered failure surface, regardless of how
// much labour produced it. (A production-share spec, e' = (c_0 + a_E/M·c_M·Q)/G
// − e, would say that producing more AI with less AI labour yields LESS
// opacity — and it booked a human-era observability subsidy from the large
// human-era G.) Setting c_0 = 0 reproduces the v3 bounded-gate engine
// derivative-for-derivative — the recorded regression invariant for this pass
// (audit V4.1; checked by _scratch/review/scripts/v4/v4-app-regression.js,
// pinned to the v3 app at 6539bf1). Because ℓ_k ≤ 1 the gated seeding
// g(0) = k_cu(1−ℓ_k·O*(0)) stays strictly positive (re-proven under v4, V4.5):
// the v2 "eradication regime" (q_u → 0 at ℓO*(0) ≥ 1) was an artifact of the
// unbounded gate intercepting more leakage than exists, and is RETRACTED.
// Derivation: _scratch/review/derivation-audit.md Part A″ (V4.1–V4.14).
// NOTE: the sharpness exponent β below is unrelated to the old "ℓ_k = βℓ"
// extension note; that extension is now the explicit ℓ_k parameter.
// =============================================================================

// Observability sharpness is fixed at 1 (soft split O = m/(m+e)); no slider.
const BETA = 1;

function computeO(m, e, beta) {
  const mu = Math.max(m, 1e-12);
  const eps = Math.max(e, 1e-12);
  const lm = beta * Math.log(mu);
  const le = beta * Math.log(eps);
  const mx = Math.max(lm, le);
  const a = Math.exp(lm - mx);
  const b = Math.exp(le - mx);
  return a / (a + b);
}

function makeDeriv(p) {
  return (state) => {
    const [Q, m, e, eta] = state;
    const O = computeO(m, e, BETA);
    const Lk = p.k_cu * (1 - Q) + p.k_hu * eta; // leakage inflow into F_u
    const I = p.lk * Lk + p.l * Q;              // intercepted flow (per unit O)
    const Fc = (1 - Q) * (1 - p.k_cu) + eta * (1 - p.k_hu) + (1 - p.delta) * O * I;
    const Fu = (1 - p.lk * O) * Lk + (p.k_uu - p.l * O) * Q;
    const G = Fc + Fu;  // = (1-Q) + η + k_uu·Q - δ·O·I
    if (G <= 1e-10) {
      // Out of validity envelope (progress clock broken)
      return [0, 0, 0, -eta];
    }
    const FM = p.c_M * ((1 - Q) + eta / p.a_ai_h);
    const FEact = p.a_e_m * p.c_M * Q;  // active evasion (labour share)
    return [
      Fu / G - Q,
      FM / G - m,
      p.c_0 + FEact / G - e,  // v4 growth peg: dE_passive = c_0·dA
      -eta,
    ];
  };
}

function computeG(state, p) {
  const [Q, m, e, eta] = state;
  const O = computeO(m, e, BETA);
  const I = p.lk * (p.k_cu * (1 - Q) + p.k_hu * eta) + p.l * Q;
  return (1 - Q) + eta + p.k_uu * Q - p.delta * O * I;
}

function rk4Step(state, dt, deriv) {
  const k1 = deriv(state);
  const s2 = state.map((v, i) => v + 0.5 * dt * k1[i]);
  const k2 = deriv(s2);
  const s3 = state.map((v, i) => v + 0.5 * dt * k2[i]);
  const k3 = deriv(s3);
  const s4 = state.map((v, i) => v + dt * k3[i]);
  const k4 = deriv(s4);
  return state.map((v, i) => v + (dt / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]));
}

// Long-run observability at odds r = Q/(1−Q) (v4 growth peg, audit A″ V4.2–3).
// The (m, e) QSS no longer cancels total production F:
//   m* = c_M·q_c/F,   e* = c_0 + a_E/M·c_M·q_u/F,
// and F is itself linear in O, so the consistency condition 1/O = 1 + e*/m*
// clears to a per-r QUADRATIC in O:
//   h(O) = cδP·O² − D·O + 1 = 0,
//   P = ℓ_k·k_cu + ℓ·r,  D = 1 + c + (a_E/M + c·k_uu)·r,  c = c_0/c_M.
// O*(r) is the MINUS root, written in the numerically stable form
// 2/(D + √disc) — exact 1/D limit as cδP → 0 (δ = 0 or c_0 = 0; at k_uu = 1
// this is the v3 closed form). Unlike v2/v3, O* now depends on δ, ℓ and ℓ_k:
// suppression slows growth per unit labour, and growth is what brings passive
// opacity, so dO*/dℓ > 0 and dO*/dδ > 0 (V4.9). disc < 0 is the
// observability-sector validity-envelope exit (deep suppression: the QSS has
// no interior equilibrium) — return NaN and let callers skip.
function ostarLongRun(r, p) {
  const c = p.c_0 / Math.max(p.c_M, 1e-12);
  const P = p.lk * p.k_cu + p.l * r;
  const D = 1 + c + (p.a_e_m + c * p.k_uu) * r;
  const cdP = c * p.delta * P;
  const disc = D * D - 4 * cdP;
  if (disc < 0) return NaN;
  return 2 / (D + Math.sqrt(disc));
}

// Sign function for q_u' on the long-run slow manifold (η → 0, m/e at QSS):
//   g(r) = k_cu + b·r − O*(r)·(ℓ_k·k_cu + ℓ·r)(1 + (1−δ)r),  b = k_cu + k_uu − 1.
// Same form as v3 given O; only O*(r) changed (implicit, above). The seeding
// is gated but bounded: g(0) = k_cu·(1 − ℓ_k·O*(0)) > 0 for ℓ_k ≤ 1, since
// O*(0) < 1 strictly whenever c₀ > 0 (h(1)|_{r=0} = c(δℓ_k·k_cu − 1) < 0 —
// re-proven under v4, audit V4.5). Substituting g = 0 into h(O) = 0 makes the
// exact fixed points a CUBIC in r, Φ(r) = cδN² − D·N·S + P·S² (N = k_cu + b·r,
// S = 1 + (1−δ)r, valid on the minus-root branch, V4.4); the app root-finds
// g directly.
function gLongRun(r, p) {
  if (r < 0) return NaN;
  const O = ostarLongRun(r, p);
  if (!Number.isFinite(O)) return NaN;
  const b = p.k_cu + p.k_uu - 1;
  return p.k_cu + b * r - O * (p.lk * p.k_cu + p.l * r) * (1 + (1 - p.delta) * r);
}

// Long-run projection: all positive roots of g (log-grid sign scan +
// bisection, matching the derivation engine's resolution). disc(r) is
// quadratic with positive leading coefficient, so O* is undefined on at most
// ONE interval (r₁, r₂) — the validity-envelope gap. Crossings ADJACENT to
// the gap (g dives through zero just before the envelope edge — a real fixed
// point, common in deep-suppression corners) are caught by bisecting the
// edge itself and testing the sign against the near-edge value. g(0) > 0 by
// construction, so the smallest positive root is always a downward (stable)
// crossing; the v2 eradication branch, which needed g(0) ≤ 0, stays
// retracted (audit V4.5).
function findSteadyStatesR(p) {
  const f = (r) => gLongRun(r, p);
  const roots = [];
  const N = 3000;
  const lnLo = Math.log(1e-9), lnHi = Math.log(300);
  const rs = new Array(N + 2), gs = new Array(N + 2);
  rs[0] = 0; gs[0] = f(0);
  for (let i = 0; i <= N; i++) {
    rs[i + 1] = Math.exp(lnLo + (i / N) * (lnHi - lnLo));
    gs[i + 1] = f(rs[i + 1]);
  }
  const bisectRoot = (lo, hi, gLo) => {
    for (let j = 0; j < 80; j++) {
      const mid = 0.5 * (lo + hi);
      const gm = f(mid);
      if (!Number.isFinite(gm)) return null; // hidden envelope sliver: no certain root
      if (gm * gLo <= 0) hi = mid;
      else lo = mid;
    }
    return 0.5 * (lo + hi);
  };
  for (let i = 0; i + 1 < rs.length; i++) {
    const g0 = gs[i], g1 = gs[i + 1];
    const ok0 = Number.isFinite(g0), ok1 = Number.isFinite(g1);
    if (ok0 && ok1) {
      if (g0 * g1 < 0) {
        const root = bisectRoot(rs[i], rs[i + 1], g0);
        if (root !== null) roots.push(root);
      }
    } else if (ok0 !== ok1) {
      // envelope edge inside this cell: bisect toward the edge from the
      // finite side, then check for a crossing against the near-edge value
      const leftFinite = ok0;
      let fin = leftFinite ? rs[i] : rs[i + 1];
      let bad = leftFinite ? rs[i + 1] : rs[i];
      let edgeR = fin, edgeVal = leftFinite ? g0 : g1;
      for (let j = 0; j < 60; j++) {
        const mid = 0.5 * (fin + bad);
        const gm = f(mid);
        if (Number.isFinite(gm)) { fin = mid; edgeR = mid; edgeVal = gm; }
        else bad = mid;
      }
      const anchor = leftFinite ? g0 : g1;
      if (anchor * edgeVal < 0) {
        const root = leftFinite
          ? bisectRoot(rs[i], edgeR, g0)
          : bisectRoot(edgeR, rs[i + 1], edgeVal);
        if (root !== null) roots.push(root);
      }
    }
  }
  return roots.sort((a, b) => a - b);
}

function rToQ(r) { return r / (1 + r); }

// =============================================================================
// NUMERIC BASIN-EXISTENCE BOUNDARY (v4 growth peg; implicit O*)
// =============================================================================
// Under v4 the fixed points solve a cubic with an implicit O*(r), and the v3
// closed-form threshold chain (quadratic A/B/C, two-branch ℓ*O*₀ rules) is
// SUPERSEDED (audit A″ V4.8). The boundary is computed numerically instead:
// since g(0) > 0 always, an interior fixed point exists iff g dips to ≤ 0
// somewhere, so the margin function is −min g over the cooperative-side
// validity interval — positive exactly when a cooperative-side basin exists,
// with the basin appearance/disappearance (saddle-node) as its zero level
// set. Existence is monotone in ℓ with a
// unique threshold ℓ* (dO*/dℓ > 0 and ∂g/∂ℓ < 0 pointwise, V4.8), found by
// bisection. One closed form survives, used as a cross-check in the
// regression suite: at δ = 1, k_uu = 1, a_E/M = 1,
//   ℓ*(δ=1) = 2k_cu[(T − ck_cu) + √((T − ck_cu)(T − ck_cu − ℓ_k))],  T = 1+c
// — the v3 form with T discounted by the F-feedback term ck_cu. In extreme
// corners (δ·k_cu beyond the V4.6 validity bound, e.g. AI-2027 at δ = 1 under
// the dial pin) NO finite ℓ produces a basin: min g stays positive at any fix
// rate, and lstar() returns Infinity.
// Cross-checked against _scratch/review/scripts/v4/engine_v4.py.
const BASIN_BOUNDARY = {
  id: 'growthPegImplicitOstar',
  label: 'numeric basin boundary (v4 ℓ*)',
  // −min_r g(r) over the COOPERATIVE-SIDE defined interval: the contiguous
  // stretch of defined O* starting at r = 0 (disc is quadratic in r, so the
  // validity envelope opens at most one NaN gap; the cooperative-side basin
  // and its appearance live below it). Log grid + golden-section refinement
  // around the grid minimum, plus an explicit bisection of the envelope edge
  // (g can dive through zero right at the edge in deep-suppression corners).
  // > 0 ⟺ a long-run interior fixed point exists on the cooperative side.
  margin(odeP) {
    const N = 160;
    const lnLo = Math.log(1e-6), lnHi = Math.log(300);
    const lnAt = (i) => lnLo + (i / N) * (lnHi - lnLo);
    const gAt = (ln) => {
      const gv = gLongRun(Math.exp(ln), odeP);
      return Number.isFinite(gv) ? gv : Infinity;
    };
    let best = Infinity, bestI = -1, edgeI = -1;
    for (let i = 0; i <= N; i++) {
      const gv = gLongRun(Math.exp(lnAt(i)), odeP);
      if (!Number.isFinite(gv)) { edgeI = i; break; } // envelope gap: stop here
      if (gv < best) { best = gv; bestI = i; }
    }
    if (bestI < 0) return -Infinity; // cannot happen for r → 0, kept as a guard
    if (edgeI > 0) {
      // refine the envelope edge and include the near-edge value in the min
      let lo = lnAt(edgeI - 1), hi = lnAt(edgeI);
      for (let it = 0; it < 40; it++) {
        const mid = 0.5 * (lo + hi);
        if (gAt(mid) < Infinity) lo = mid;
        else hi = mid;
      }
      best = Math.min(best, gAt(lo));
    }
    // golden-section refinement within the bracketing neighbours
    let lo = lnAt(Math.max(bestI - 1, 0));
    let hi = lnAt(Math.min(bestI + 1, edgeI > 0 ? edgeI : N));
    const phi = 0.5 * (Math.sqrt(5) - 1);
    let x1 = hi - phi * (hi - lo), x2 = lo + phi * (hi - lo);
    let f1 = gAt(x1), f2 = gAt(x2);
    for (let it = 0; it < 24; it++) {
      if (f1 <= f2) { hi = x2; x2 = x1; f2 = f1; x1 = hi - phi * (hi - lo); f1 = gAt(x1); }
      else { lo = x1; x1 = x2; f1 = f2; x2 = lo + phi * (hi - lo); f2 = gAt(x2); }
    }
    return -Math.min(best, f1, f2);
  },
  // Critical fix rate ℓ* above which a long-run interior fixed point exists
  // (bisection on the monotone existence; Infinity when no finite ℓ works).
  lstar(odeP, lHi = 400) {
    const exists = (l) => this.margin({ ...odeP, l }) > 0;
    if (!exists(lHi)) return Infinity;
    if (exists(0)) return 0;
    let lo = 0, hi = lHi;
    for (let it = 0; it < 60; it++) {
      const mid = 0.5 * (lo + hi);
      if (exists(mid)) hi = mid;
      else lo = mid;
    }
    return 0.5 * (lo + hi);
  },
};

function classifyBasin(p) {
  const rRoots = findSteadyStatesR(p);
  const qRoots = rRoots.map(rToQ);
  // When k_cu = 0 there is no leakage seeding, so q = 0 is an exact cooperative
  // fixed point (g(0) = k_cu = 0). The interior root-finder skips this boundary
  // root, so add it explicitly when it is stable: ℓ·O*(0) ≥ k_uu − 1. Without
  // this, k_cu = 0 is misreported as "escape" even though trajectories
  // converge to the cooperative state. O*(0) is now the v4 implicit root
  // (under the dial pin it equals the O* slider exactly); at k_cu = 0 the
  // O-quadratic degenerates (P(0) = 0) and O*(0) = 1/(1 + c_0/c_M) — the
  // stability condition is still δ- and ℓ_k-free there (audit V4.5/V4.9:
  // both gated leakage terms vanish identically at k_cu = 0, q_h = 0).
  const Ostar0 = ostarLongRun(0, p);
  if (p.k_cu <= 1e-9) {
    if (p.l * Ostar0 >= p.k_uu - 1) qRoots.unshift(0);
  }
  // Badge readout: the k_cu → 0 reduction of Condition 1 is ℓ·O*(0) ≥ k_uu − 1.
  // The full v4 C1 has no closed bracket (it is g'(0) < 0 with the implicit
  // O*'(0), audit V4.10); at the central calibrations the effective bracket
  // is ≈ 0.79 (ℓ_k = 1) / 0.93 (ℓ_k = 0.5).
  // For ℓ_k ≤ 1 the seeding g(0) = k_cu(1−ℓ_k·O*(0)) is strictly positive, so
  // the smallest positive root is always a downward (stable) crossing; the v2
  // eradication branch stays retracted (audit V4.5).
  const lOstar0 = p.l * Ostar0;
  const kuuExcess = p.k_uu - 1;
  if (qRoots.length === 0) return { kind: 'escape', qRoots: [], qStable: null, qSaddle: null, lOstar0, kuuExcess };
  if (qRoots.length === 1) return { kind: 'monostable', qRoots, qStable: qRoots[0], qSaddle: null, lOstar0, kuuExcess };
  return { kind: 'bistable', qRoots, qStable: qRoots[0], qSaddle: qRoots[1], lOstar0, kuuExcess };
}

// σ-time simulation. ic = [Q₀, m₀, e₀, η₀].
function simulate(p, ic, sigmaMax, dtBase = 0.005, maxSamples = 500) {
  const deriv = makeDeriv(p);
  const dt = dtBase;
  const steps = Math.max(1, Math.ceil(sigmaMax / dt));
  const sampleEvery = Math.max(1, Math.floor(steps / maxSamples));
  let state = [...ic];
  const O0 = computeO(state[1], state[2], BETA);
  const G0 = computeG(state, p);
  const samples = [{ sigma: 0, state: [...state], O: O0, G: G0 }];
  let escaped = false;
  let escapeReason = null;
  for (let i = 1; i <= steps; i++) {
    const Gnow = computeG(state, p);
    if (Gnow <= 1e-6) {
      escaped = true;
      escapeReason = 'G≤0 (outside validity)';
      const O = computeO(state[1], state[2], p.beta);
      samples.push({ sigma: i * dt, state: [...state], O, G: Gnow });
      break;
    }
    state = rk4Step(state, dt, deriv);
    state[0] = Math.min(Math.max(state[0], 0), 1);
    state[1] = Math.max(state[1], 0);
    state[2] = Math.max(state[2], 0);
    state[3] = Math.max(state[3], 0);
    if (state.some(v => !Number.isFinite(v))) {
      escaped = true; escapeReason = 'numerical';
      break;
    }
    if (state[0] > 0.999) escaped = true;
    if (i % sampleEvery === 0 || i === steps || escaped) {
      const O = computeO(state[1], state[2], p.beta);
      const G = computeG(state, p);
      samples.push({ sigma: i * dt, state: [...state], O, G });
      if (escaped) break;
    }
  }
  return { samples, escaped, escapeReason };
}

// =============================================================================
// PARAMETERS
// =============================================================================

const PARAM_DEFS = {
  T_auto: {
    label: 'Automation doubling time',
    symbol: 'T_auto',
    unit: 'years',
    handle: 'Present-day calendar doubling time of total automation A.',
    est: 'Blog ≈ 3–7 months (OpenAI revenue ~7 mo; Epoch compute×efficiency ~3.6 mo).',
    detail: 'Only scales the calendar-years axis — the dynamics run on the σ-clock, and under the main calibration it no longer enters the rates.',
    math: [String.raw`\text{years} = \sigma \cdot T_{auto}/\ln 2`],
    role: 'Calendar↔σ conversion, exact at σ=0; later approximate since the productivity multiplier Λ is deliberately unmodelled.',
    default: 0.5, min: 0.1, max: 5, step: 0.05,
    tier: 'time',
  },
  Ostar: {
    label: 'Best-case observability',
    symbol: 'O*',
    handle: 'If the uncoop share stays low, where does observability settle in the long run?',
    est: 'Central ≈ 50% (blog sweeps O(0) ∈ 0.3–0.9; the pin back-solves c₀/c_M ≈ 1.02 from O* = ½ at Broad).',
    detail: 'Folds a messy economic judgement — the risk of under-investing in observability versus the cost of pushing it higher — into one long-run level. Splitting it into pieces would not make it easier to estimate, so the model asks for the result directly. The dial is enforced at the long-run rare-uncooperative point (q_h = 0, q_u → 0): under v4 the long-run O* solves a quadratic there, so the pinned c₀/c_M depends (weakly) on δ, ℓ_k and k_cu and is re-solved per parameter cell. "Best case" because it holds while uncoop is rare; true long-run observability falls as q_u grows.',
    math: [String.raw`c\,\delta\,\ell_k k_{cu}\,O^{*2} - (1+c)\,O^* + 1 = 0`, String.raw`c = \frac{c_0}{c_M} = \frac{1-O^*}{O^*\,(1-\delta\,\ell_k k_{cu}\,O^*)}`],
    role: 'Sets passive opacity c₀ relative to monitoring c_M (per-cell pin; ℓ-free). c_M itself is pegged to the automation scale, controlling only how fast O relaxes from O(0).',
    related: ['R0', 'a_e_m', 'a_ai_h'],
    default: 0.5, min: 0.05, max: 0.95, step: 0.01,
    tier: 'time',
  },
  k_uu: {
    label: 'Uncoop. self-productivity',
    symbol: 'k_uu',
    handle: 'For equal effort, how much uncoop AI is produced vs cooperative AI?',
    est: 'Blog anchor = 1 (reproduction at par; upward pressure over the horizon).',
    detail: 'Uncoop self-production rate, relative to cooperative self-production (normalised to 1). > 1 is strongly self-reinforcing.',
    math: [String.raw`F_u \ni (k_{uu} - \ell O)\,q_u`],
    role: 'Condition 2 — the all-uncoop endpoint is stable when k_uu + k_cu − 1 > (1−δ)ℓ/(a + c·k_uu) (δ = 1: k_uu + k_cu > 1). Unchanged by gating; ℓ_k does not enter (at fixed c₀/c_M — the dial pin moves c imperceptibly).',
    related: ['k_cu', 'l'],
    default: 1, min: 0, max: 2, step: 0.01,
    tier: 'primary',
  },
  k_cu: {
    label: 'Coop → uncoop leakage',
    symbol: 'k_cu',
    handle: 'Fraction of cooperative AI labour that ends up enabling uncoop instead.',
    est: 'Human-era identity (level + trend) at ℓ = 0.2, ℓ_k = 1: Broad 0.0618 (trend-adjusted; range 0.062–0.083 across the g-bracket, 0.042–0.056 at ℓ_k = 0.5), Strict 0.0085 (steady-state); naive level proxy 0.05; "high leakage" > 0.9.',
    detail: 'A true transfer: this much of q_c is diverted toward q_u (and removed from q_c) — but only the fraction (1−ℓ_k·O) that escapes interception at production actually enters the pool, so the effective seeding is k_cu(1−ℓ_k·O*(0)). Default values come from the human-era calibration identity, which back-solves k from today\'s observed level q0 (and, for Broad, the observed trend — the instruments bracket g ∈ [−0.22, 0]) given the suppression outflow at today\'s O.',
    math: [String.raw`F_c \ni q_c(1-k_{cu})`, String.raw`F_u \ni (1-\ell_k O)\,k_{cu}\,q_c`],
    role: 'Sets the gated seeding g(0) = k_cu(1−ℓ_k·O*(0)) > 0 and enters Condition 2 (k_uu + k_cu > 1 at δ = 1). Note: through the identity the calibrated k_cu itself depends on ℓ, ℓ_k and O(0) — crediting weaker suppression implies MORE leakage to hold the same observed level — so moving ℓ with k_cu frozen mixes "lever moved" with "evidence re-read".',
    related: ['k_hu', 'k_uu', 'l', 'lk'],
    default: 0.0618, min: 0, max: 1, step: 0.0001,
    tier: 'primary',
  },
  l: {
    label: 'Fix rate of known failure modes',
    symbol: 'ℓ',
    handle: 'How fast are known (covered) failure modes fixed out of the deployed stock, per σ-unit?',
    est: 'Central ℓ ≈ 0.2 (fixed-harness retrospective slope, read raw — NO division by O(0)); nominal range 0.1–0.25.',
    detail: 'The fix throughput of the detect-and-fix pipeline on established behaviour: −ℓO·q_u removes covered misbehaviour from the stock. A rate per σ-unit, not a fraction — it may exceed 1. Estimation (instrument theory): a fixed-harness retrospective series (Petri-style) is run by the developer, so what it flags is a subset of what the developer observes and acts on — its raw decay slope estimates ℓ directly, with no observability division. (The earlier reading divided the slope by O(0) ≈ 0.5 to get ℓ ≈ 0.4; under the instrument theory that division was wrong.) Contemporaneous deployment series instead confound fixing with hiding (the O-trend) and cannot separate them; multi-vintage panels identify the coverage gradient. One conservative caveat: new-arrival leakage into the fixed harness\'s coverage understates the true fix rate. Removal-vs-concealment is carried by O, not by a bias on ℓ — only the product ℓ·O enters suppression.',
    math: [String.raw`F_u = (1-\ell_k O)(k_{cu}q_c + k_{hu}q_h) + (k_{uu} - \ell O)\,q_u`, String.raw`\ell \approx \text{fixed-harness decay slope (raw)}`],
    role: 'Bounded-gate Condition 1 (v4, implicit O*): rare uncoop labour shrinks iff O*(0)·[ℓ + (1−δ)ℓ_k·k_cu] + O*′(0)·ℓ_k·k_cu > k_uu + k_cu − 1, with O*′(0) < 0 from differentiating the O-quadratic (no closed bracket survives the implicit O*; the v3 necessity-at-δ=1 proof is now numerical support only). High ℓ drives the attractor down the endemic floor q* ≈ k_cu(1+c−ℓ_k−cδk_cu)/ℓ — arbitrarily low, never zero (floor valid while 2cδk_cu ≤ 1+c).',
    related: ['lk', 'k_uu', 'k_cu', 'Ostar', 'delta'],
    default: 0.2, min: 0, max: 8, step: 0.05,
    tier: 'primary',
  },
  lk: {
    label: 'Interception efficacy at production',
    symbol: 'ℓ_k',
    handle: 'Of newly produced misbehaviour, what fraction can the detect-and-fix stock intercept at full observability?',
    est: 'Unanchored — no empirical estimate yet. Display values 0.5 and 1; bounded in [0, 1] by construction.',
    detail: 'The bounded leak gate: only the fraction (1−ℓ_k·O) of leaked effort enters the uncooperative pool. Unlike the fix rate ℓ (a rate on the established stock, which may exceed 1), ℓ_k is a FRACTION of the inflow — interception cannot remove more leakage than exists. That bound is what forbids eradication: g(0) = k_cu(1−ℓ_k·O*(0)) > 0 for ℓ_k ≤ 1, so rare uncoop labour always grows, and very high ℓ only pushes the attractor down the endemic floor k_cu(1+c−ℓ_k−cδk_cu)/ℓ, never to zero (at ℓ_k = 1 the surviving seeding arrives through novel modes — c₀ > 0 keeps O* < 1). No empirical anchor yet; in principle calibratable (what fraction of new misbehaviour do pipeline workarounds catch at full observability?).',
    math: [String.raw`F_u = (1-\ell_k O)(k_{cu}q_c + k_{hu}q_h) + (k_{uu} - \ell O)\,q_u`, String.raw`\Phi(0) = -k_{cu}(1+c-\ell_k-c\delta k_{cu}) < 0`],
    role: 'Gates the seeding (the cubic\'s constant term Φ(0) is ℓ-free and strictly negative for ℓ_k ≤ 1). Whether high observability alone can rescue a bad calibration lives or dies on ℓ_k: at ℓ_k = 1, O* → 1 drives q* → 0 (Broad needs O* ≥ 0.88); at ℓ_k = 0.5 half the leak gets through no matter how well you see (Broad floors at q* ≈ 0.14 as O* → 1). Under v4 that split is a recomputed numerical fact — the v3 exact two-product structure (ℓ·O* and ℓ_k·O*) is broken by the growth peg.',
    related: ['l', 'k_cu', 'Ostar'],
    default: 1, min: 0, max: 1, step: 0.05,
    tier: 'primary',
  },
  delta: {
    label: 'Suppression destruction fraction',
    symbol: 'δ',
    handle: 'Of the uncoop labour suppression catches, how much is destroyed vs redirected to cooperative work?',
    est: 'Central ≈ 0.7 via the filtering-fraction heuristic (fresh estimate); range 0.3–1.0.',
    detail: 'δ = 1: caught uncoop labour is destroyed outright (filtered, blocked, discarded). δ = 0: every caught unit is corrected and its labour redirected into cooperative production at full value. Under the bounded gate the intercepted flow is O·(ℓ_k·L_k + ℓ·q_u) with L_k = k_cu q_c + k_hu q_h — leakage interception and stock removal are split by the same δ. Heuristic: weight suppression channels by destructiveness (output filtering ≈ 1, retraining ≈ 0, control-style resampling/editing ≈ ½) and average by suppressed-flow volume; heavy real-time filtering on frontier deployments puts the central value high. Redirection is assumed 1:1, with no lag or quality discount.',
    math: [
      String.raw`F_u = (1-\ell_k O)(k_{cu}q_c{+}k_{hu}q_h) + (k_{uu}-\ell O)\,q_u`,
      String.raw`F_c \ni +(1-\delta)\,O\,(\ell_k L_k + \ell\,q_u)`,
      String.raw`G = q_c + q_h + k_{uu}q_u - \delta\,O\,(\ell_k L_k + \ell\,q_u)`,
    ],
    role: 'Lowers the basin threshold ℓ* (numeric under v4; at δ = 1, k_uu = 1, a_E/M = 1 the closed form is ℓ* = 2k_cu[(T−ck_cu) + √((T−ck_cu)(T−ck_cu−ℓ_k))], T = 1+c) and destabilises the all-uncoop endpoint when k_uu + k_cu − 1 < (1−δ)ℓ/(a + c·k_uu) — a condition neither gate moves. Caution: for δ < 1 the surviving attractor can sit at high q_u — check its location, not just existence. At extreme δ·k_cu (beyond 2cδk_cu ≤ 1+c) deep suppression stops flooring the system at all: no finite ℓ produces a basin.',
    related: ['l', 'k_cu', 'k_uu'],
    default: 0.7, min: 0, max: 1, step: 0.05,
    tier: 'primary',
  },
  a_e_m: {
    label: 'Evasion advantage',
    symbol: 'a_E/M',
    handle: 'Per unit of AI labour, how productive is evasion vs monitoring?',
    est: 'Order 1 (M-Trends dwell-time); growth-advantage scenario pushes > 1.',
    detail: '> 1 means uncoop AI produces evasion faster than cooperative AI produces monitoring, at equal labour. Active evasion is a labour share; passive opacity c₀ is pegged to automation growth instead (v4).',
    math: [String.raw`e' = c_0 + a_{E/M}\,c_M\,q_u/F - e`],
    role: 'Pulls long-run observability O*(q_u) down as q_u grows.',
    related: ['Ostar', 'a_ai_h'],
    default: 1, min: 0.1, max: 10, step: 0.05,
    tier: 'primary',
  },
  a_ai_h: {
    label: 'AI monitoring-production advantage',
    symbol: 'a_AI/H',
    handle: 'How much better is AI than humans at producing monitoring?',
    est: 'Parity (= 1), wide both-way uncertainty (METR, RE-Bench).',
    detail: 'a_AI/H = 10 ⇒ one unit of human labour makes a tenth the monitoring of a unit of cooperative AI. Mostly a scale knob — sets c_M at calibration; the basin sees it only weakly via c₀/c_M.',
    math: [String.raw`F_M = c_M\left(q_c + q_h/a_{AI/H}\right)`],
    role: 'Scales the human contribution to monitoring during the q_h (human) era.',
    related: ['Ostar', 'eta0'],
    default: 1, min: 0.1, max: 10, step: 0.05,
    tier: 'primary',
  },
  k_hu: {
    label: 'Human → uncoop leakage',
    symbol: 'k_hu',
    handle: 'Fraction of human labour that ends up enabling uncoop AI.',
    est: 'Set jointly with k_cu by the human-era identity at ℓ = 0.2, ℓ_k = 1: Broad 0.0618, Strict 0.0085 (naive level proxy ~0.05).',
    detail: 'Like k_cu but for human labour, and equally gated: only the fraction (1−ℓ_k·O) of the leaked flow enters the pool. Active only while q_h > 0; fades as humans become economically irrelevant.',
    math: [String.raw`F_u \ni (1-\ell_k O)\,k_{hu}\,q_h`],
    role: 'Reproduction-relevant leakage (property ii′); plausibly rarer than observed misbehaviour.',
    related: ['k_cu', 'eta0'],
    default: 0.0618, min: 0, max: 1, step: 0.0001,
    tier: 'primary',
  },
  // Initial conditions
  q0: {
    label: 'Initial uncoop share',
    symbol: 'q_u(0)',
    handle: 'Starting fraction of total automation that is uncooperative.',
    est: 'Observed share ÷ O(0): ~1–67%, central ~5%.',
    detail: 'Below the stable q_u* it converges; above the saddle it escapes (long-run projection). The full 4D basin can differ from this 1-D projection.',
    math: [String.raw`q_u = \frac{A_u}{A_c + A_u}`],
    role: 'Initial condition; matters only in the bistable regime.',
    related: ['Ostar', 'R0'],
    default: 0.05, min: 0, max: 0.99, step: 0.005,
    tier: 'ic',
  },
  R0: {
    label: 'Misses per catch today',
    symbol: 'R(0)',
    scale: 'log',
    handle: 'For every uncoop behaviour monitoring catches today, how many slip past?',
    est: 'O(0) ∈ 0.3–0.9 (central 0.5) ⇒ R(0) ≈ 0.1–2.3, central 1.',
    detail: 'Present missed-to-caught odds. Sets the initial observability O(0) the trajectory starts from, before it drifts toward O*.',
    math: [String.raw`R = \frac{1-O}{O} = \frac{e}{m}`, String.raw`O(0) = \frac{1}{1+R(0)}`],
    role: 'With m(0) = 1 fixing stock units, e(0) = R(0) directly.',
    related: ['Ostar'],
    default: 1, min: 0.01, max: 100, step: 0.001,
    tier: 'ic',
  },
  eta0: {
    label: 'Initial human stock',
    symbol: 'q_h(0)',
    handle: 'Human labour as a multiple of total automation at the start.',
    est: 'Starts ≫ 1; default 5, decays as AI labour grows.',
    detail: 'q_h = H₀/(A_c+A_u). q_h = 1 is AI–human parity (A = H₀); below 1 automation already exceeds humans, past the model\'s intended start.',
    math: [String.raw`q_h' = -q_h \;\Rightarrow\; q_h = q_{h,0}\,e^{-\sigma}`],
    role: 'The human era. Parity (q_h = 1) is marked on the σ-plots at σ = ln q_{h,0}.',
    related: ['k_hu', 'a_ai_h'],
    default: 5, min: 1, max: 50, step: 0.5,
    tier: 'ic',
  },
};

function defaultParams() {
  const p = {};
  for (const [k, def] of Object.entries(PARAM_DEFS)) p[k] = def.default;
  return p;
}

// =============================================================================
// NAMED CALIBRATION PRESETS
// =============================================================================
// Broad and Strict share every structural parameter and differ only in how the
// observed-misbehaviour evidence is read. The leakage rates are no longer the
// raw level q0: they come from the human-era calibration identity (bounded
// gate, SELF-CONSISTENT F0 convention, audit V10/V4.11; with S = 1−q0+η0, k_uu = 1)
//   k = q0[(1+g)(1+η0−δℓO0·q0) − (1−ℓO0)] / (S[(1−ℓ_k·O0) + (1+g)δℓ_k·O0·q0])
// at ℓ = 0.2, δ = 0.7, O0 = 0.5, η0 = 5. Broad uses the TREND-ADJUSTED
// identity (level q0 = 0.05 plus trend g = −0.22/σ; the instruments bracket
// g ∈ [−0.22, 0], so the post presents Broad's k as the range 0.062–0.083 at
// ℓ_k = 1): k = 0.0618. Strict uses the STEADY-STATE identity at its own
// level q0 = 0.005 (the falling broad series doesn't measure the (ii′) pool,
// so its trend is not licensed): k = 0.0085. q0 keeps the LEVEL — k ≠ q0.
// AI 2027 is Broad with k_cu pushed to the high-leakage reading of the
// AI 2027 scenario.
// ℓ_k is UNANCHORED; presets use the ℓ_k = 1 display value (headline choice
// D16 pending with David — at ℓ_k = 0.5 the identity gives Broad 0.0419 /
// Strict 0.0057 instead).
// The identity values are EXACTLY unchanged under the v4 growth peg (the
// identity never touches the e-equation — provable, audit V4.11), so the
// presets carry the same numbers as v3.
// Values must stay in sync with _scratch/review/drafts/calibrations-v4.md.
const PRESETS = {
  broad: {
    label: 'Broad',
    blurb: 'observed misbehaviour as proxy; trend-adjusted identity k = 0.0618 (ℓ_k = 1)',
    params: {
      T_auto: 0.5, Ostar: 0.5, k_uu: 1, k_cu: 0.0618, l: 0.2, lk: 1, delta: 0.7,
      a_e_m: 1, a_ai_h: 1, k_hu: 0.0618, q0: 0.05, R0: 1, eta0: 5,
    },
  },
  strict: {
    label: 'Strict',
    blurb: 'reproduction-relevant (ii′) rates; steady-state identity k = 0.0085 (ℓ_k = 1)',
    params: {
      T_auto: 0.5, Ostar: 0.5, k_uu: 1, k_cu: 0.0085, l: 0.2, lk: 1, delta: 0.7,
      a_e_m: 1, a_ai_h: 1, k_hu: 0.0085, q0: 0.005, R0: 1, eta0: 5,
    },
  },
  ai2027: {
    label: 'AI 2027',
    blurb: 'Broad with high leakage k_cu = 0.9 at handoff',
    params: {
      T_auto: 0.5, Ostar: 0.5, k_uu: 1, k_cu: 0.9, l: 0.2, lk: 1, delta: 0.7,
      a_e_m: 1, a_ai_h: 1, k_hu: 0.0618, q0: 0.05, R0: 1, eta0: 5,
    },
  },
};

function presetIsActive(params, preset) {
  return Object.keys(PARAM_DEFS).every(
    k => Math.abs(params[k] - preset.params[k]) < 1e-12
  );
}

// Trajectory-view integration horizon (in σ). Not a model parameter.
const SIGMA_MAX_DEFAULT = 15;
const SIGMA_MAX_MIN = 5;
const SIGMA_MAX_MAX = 60;

// e(0) = R(0) directly (missed-to-caught odds), with m(0)=1 fixing the units.
function deriveIC(p) {
  const m0 = 1;
  const e0 = Math.max(p.R0, 1e-9);
  return { m0, e0 };
}

function calibratedRates(p) {
  const Q = p.q0;
  const eta = p.eta0;
  const R0 = Math.max(p.R0, 1e-9);
  const O0 = 1 / (1 + R0);
  const { m0 } = deriveIC(p);
  // Bounded-gate total at σ = 0 (must match computeG: the destroyed part of
  // intercepted leakage also subtracts from G).
  const Lk0 = p.k_cu * (1 - Q) + p.k_hu * eta;
  const G0 = Math.max((1 - Q) + eta + p.k_uu * Q - p.delta * O0 * (p.lk * Lk0 + p.l * Q), 0.05);
  const monitoringSource0 = Math.max((1 - Q) + eta / p.a_ai_h, 1e-9);

  // Main calibration method (blog): fix present observability O(0) via R(0)
  // (which sets m0=1, e0=R0), peg the monitoring scale c_M to the automation
  // production scale so monitoring starts at quasi-steady-state, then set
  // passive opacity from the best-case long-run observability O* via the v4
  // dial pin (audit A″ A-pin): O* is enforced at the long-run
  // rare-uncooperative point (q_h = 0, q_u → 0), where the implicit
  // O-quadratic h(O) = cδℓ_k·k_cu·O² − (1+c)O + 1 back-solves to
  //   c_0/c_M = (1 − O*) / (O*·(1 − δ·ℓ_k·k_cu·O*)),
  // re-solved per parameter cell (depends on δ, ℓ_k, k_cu; ℓ-free). With the
  // pin, the badge's O*(0) equals the O* slider exactly. (v3's
  // c_0/c_M = (1−O*)/O* is the cδ → 0 limit; the denominator is positive
  // throughout the slider ranges since δ·ℓ_k·k_cu·O* < 1.) c_M's magnitude
  // affects only how fast O relaxes from O(0) toward O*. (T_M,½ / T_E,2 rate
  // judgements are the blog's unused alternative.)
  const Ostar = Math.min(Math.max(p.Ostar, 1e-6), 1 - 1e-6);
  const c_M = G0 * m0 / monitoringSource0;
  const pinDen = Ostar * (1 - p.delta * p.lk * p.k_cu * Ostar);
  const c_0 = c_M * (1 - Ostar) / Math.max(pinDen, 1e-9);
  return { c_M, c_0, c0Raw: c_0 };
}

function odeParams(p) {
  const rates = calibratedRates(p);
  return {
    k_uu: p.k_uu, k_cu: p.k_cu, k_hu: p.k_hu,
    l: p.l, lk: p.lk, delta: p.delta, beta: BETA,
    a_ai_h: p.a_ai_h, a_e_m: p.a_e_m,
    c_M: rates.c_M, c_0: rates.c_0, c0Raw: rates.c0Raw,
  };
}

function buildInitialState(p) {
  const { m0, e0 } = deriveIC(p);
  return [p.q0, m0, e0, p.eta0];
}

// Calendar-year conversion (linear, intentionally — see assumptions doc).
function sigmaToYears(sigma, T_auto) {
  return sigma * T_auto / Math.log(2);
}

// =============================================================================
// URL-ENCODED STATE (shareable links)
// =============================================================================
// Parameters are encoded in the query string under their PARAM_DEFS keys
// (e.g. ?v=3&k_uu=1.2&l=0.2), plus sigmaMax / view / time for the view state.
// Only values that differ from defaults are written; values read from the URL
// are clamped to each parameter's [min, max].
//
// Schema version: v=4 marks links written by the GROWTH-PEG engine (passive
// opacity per σ-unit of automation growth, O*-dial pin for c₀; 2026-06).
// v=3 links (the bounded-gate production-share e′ engine), v=2 links (the
// production-gated ℓ_k ≡ ℓ engine) and unversioned links (the ungated-δ
// engine) carry the same parameter names but mean a different model, so they
// fall back to defaults instead of being silently re-interpreted (established
// policy: HARD fallback for v < current).
const URL_SCHEMA_VERSION = '4';

function trimNum(v) {
  return String(+v.toPrecision(6));
}

function readStateFromURL() {
  const out = {
    params: defaultParams(),
    sigmaMax: SIGMA_MAX_DEFAULT,
    view: 'trajectory',
    displayMode: 'sigma',
  };
  if (typeof window === 'undefined') return out;
  const sp = new URLSearchParams(window.location.search);
  if (sp.get('v') !== URL_SCHEMA_VERSION) return out; // legacy/unversioned link
  for (const [k, def] of Object.entries(PARAM_DEFS)) {
    const raw = sp.get(k);
    if (raw === null) continue;
    const v = parseFloat(raw);
    if (Number.isFinite(v)) out.params[k] = Math.min(Math.max(v, def.min), def.max);
  }
  const sm = parseFloat(sp.get('sigmaMax'));
  if (Number.isFinite(sm)) {
    out.sigmaMax = Math.min(Math.max(sm, SIGMA_MAX_MIN), SIGMA_MAX_MAX);
  }
  if (sp.get('view') === 'outcome') out.view = 'outcome';
  if (sp.get('time') === 'years') out.displayMode = 'years';
  return out;
}

function writeStateToURL(params, sigmaMax, view, displayMode) {
  if (typeof window === 'undefined' || !window.history?.replaceState) return;
  const sp = new URLSearchParams();
  const defaults = defaultParams();
  for (const k of Object.keys(PARAM_DEFS)) {
    if (Math.abs(params[k] - defaults[k]) > 1e-12) sp.set(k, trimNum(params[k]));
  }
  if (sigmaMax !== SIGMA_MAX_DEFAULT) sp.set('sigmaMax', trimNum(sigmaMax));
  if (view !== 'trajectory') sp.set('view', view);
  if (displayMode !== 'sigma') sp.set('time', displayMode);
  const qs = sp.toString();
  const url = window.location.pathname
    + (qs ? `?v=${URL_SCHEMA_VERSION}&` + qs : '')
    + window.location.hash;
  window.history.replaceState(null, '', url);
}

// =============================================================================
// UI COMPONENTS
// =============================================================================

function Slider({ pkey, value, onChange, extraDetail = null, onPin }) {
  const def = PARAM_DEFS[pkey];
  const hasCard = !!(def.math || def.related || def.role);
  const [hover, setHover] = useState(false);
  const rowRef = useRef(null);
  const [pop, setPop] = useState(null);
  const handleEnter = () => {
    setHover(true);
    const r = rowRef.current?.getBoundingClientRect();
    if (r) {
      const popW = 270;
      let left = r.right + 8;                       // beside the panel, over the content area
      if (left + popW > window.innerWidth - 8) left = Math.max(8, r.left - popW - 8);
      const top = Math.max(8, Math.min(r.top, window.innerHeight - 170)); // keep on-screen
      setPop({ left, top, width: popW });
    }
  };
  const isLog = def.scale === 'log';
  const sMin = isLog ? Math.log10(def.min) : def.min;
  const sMax = isLog ? Math.log10(def.max) : def.max;
  const sStep = isLog ? (sMax - sMin) / 240 : def.step;
  const sVal = isLog ? Math.log10(Math.min(Math.max(value, def.min), def.max)) : value;
  const fromSlider = (s) => isLog ? Math.pow(10, s) : s;
  return (
    <div
      ref={rowRef}
      style={{
        padding: '10px 14px',
        borderBottom: `1px solid ${C.border}`,
        position: 'relative',
        background: C.bgPanel,
      }}
      onMouseEnter={handleEnter}
      onMouseLeave={() => setHover(false)}
    >
      <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', gap: 8 }}>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, minWidth: 0 }}>
          <span style={{
            fontFamily: FONTS.sans, fontSize: 12.5, color: C.fg, fontWeight: 500,
            whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
          }}>{def.label}</span>
          <span style={{ fontFamily: FONTS.mono, fontSize: 11, color: C.fgMuted }}>{def.symbol}</span>
          {hasCard && onPin && (
            <button
              onClick={(e) => { e.stopPropagation(); onPin(pkey); }}
              title="equations & related parameters"
              style={{
                background: 'transparent', border: 'none', cursor: 'pointer', padding: 0,
                fontFamily: FONTS.mono, fontSize: 12, lineHeight: 1, color: C.accent,
              }}
            >ⓘ</button>
          )}
        </div>
        <span style={{
          fontFamily: FONTS.mono, fontSize: 12, color: C.accent, fontWeight: 600,
          whiteSpace: 'nowrap',
        }}>
          {value.toFixed(value < 0.01 ? 4 : value < 0.1 ? 3 : value < 1 ? 3 : 2)}{def.unit ? ' ' + def.unit : ''}
        </span>
      </div>
      <input
        type="range"
        min={sMin}
        max={sMax}
        step={sStep}
        value={sVal}
        onChange={e => onChange(fromSlider(parseFloat(e.target.value)))}
        style={{ width: '100%', marginTop: 6, accentColor: C.accent }}
      />
      <div style={{
        fontFamily: FONTS.sans, fontSize: 11, color: C.fgDim, marginTop: 2,
        lineHeight: 1.4,
      }}>{def.handle}</div>
      {extraDetail && (
        <div style={{
          fontFamily: FONTS.mono, fontSize: 10.5, color: C.fgMuted, marginTop: 4,
          paddingTop: 4, borderTop: `1px dashed ${C.border}`,
        }}>{extraDetail}</div>
      )}
      {hover && pop && (def.est || hasCard) && (
        <div style={{
          position: 'fixed', left: pop.left, top: pop.top, width: pop.width, zIndex: 50,
          pointerEvents: 'none',
          background: C.bgInset, border: `1px solid ${C.borderStrong}`,
          padding: '7px 10px', fontFamily: FONTS.sans, fontSize: 11,
          color: C.fgDim, lineHeight: 1.45,
          boxShadow: '0 6px 16px rgba(60,40,10,0.18)',
        }}>
          {def.est && <div style={{ color: C.accent, fontWeight: 600 }}>{def.est}</div>}
          {hasCard && <div style={{ marginTop: def.est ? 5 : 0, color: C.fgMuted, fontSize: 10 }}>ⓘ click for details &amp; equations ▸</div>}
        </div>
      )}
    </div>
  );
}

function ParamCard({ pkey, onClose, onNavigate }) {
  if (!pkey) return null;
  const def = PARAM_DEFS[pkey];
  if (!def) return null;
  return (
    <div style={{
      position: 'fixed', top: 86, right: 24, width: 360, maxHeight: 'calc(100vh - 120px)',
      overflowY: 'auto', zIndex: 60,
      background: C.bgInset, border: `1px solid ${C.borderStrong}`,
      boxShadow: '0 10px 32px rgba(60,40,10,0.28)',
    }}>
      <div style={{
        display: 'flex', justifyContent: 'space-between', alignItems: 'baseline',
        padding: '12px 14px', borderBottom: `1px solid ${C.border}`, background: C.bgPanel,
      }}>
        <div>
          <div style={{ fontFamily: FONTS.serif, fontSize: 15, fontWeight: 700, color: C.fg }}>{def.label}</div>
          <div style={{ fontFamily: FONTS.mono, fontSize: 11, color: C.accent, marginTop: 2 }}>{def.symbol}</div>
        </div>
        <button onClick={onClose} title="close" style={{
          background: 'transparent', border: 'none', cursor: 'pointer',
          fontFamily: FONTS.mono, fontSize: 14, color: C.fgMuted,
        }}>✕</button>
      </div>
      <div style={{ padding: '12px 14px', fontFamily: FONTS.sans, fontSize: 12, color: C.fgDim, lineHeight: 1.55 }}>
        {def.est && <div style={{ color: C.accent, fontWeight: 600, marginBottom: 8 }}>{def.est}</div>}
        {def.detail && <div style={{ marginBottom: 8 }}>{def.detail}</div>}
        {def.math && def.math.length > 0 && (
          <div style={{ margin: '10px 0', padding: '4px 10px', background: C.bgPanel, border: `1px solid ${C.border}` }}>
            {def.math.map((m, i) => <TeX key={i} display>{m}</TeX>)}
          </div>
        )}
        {def.role && <div style={{ fontSize: 11.5, marginBottom: 8 }}>{def.role}</div>}
        <div style={{ fontFamily: FONTS.mono, fontSize: 10.5, color: C.fgMuted }}>
          range {def.min} – {def.max}{def.unit ? ' ' + def.unit : ''} · default {def.default}
        </div>
        {def.related && def.related.length > 0 && (
          <div style={{ marginTop: 10, paddingTop: 8, borderTop: `1px solid ${C.border}` }}>
            <div style={{ fontFamily: FONTS.mono, fontSize: 10, letterSpacing: 1, color: C.fgMuted, textTransform: 'uppercase', marginBottom: 6 }}>related</div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
              {def.related.map(rk => PARAM_DEFS[rk] && (
                <button key={rk} onClick={() => onNavigate(rk)} style={{
                  background: C.bgPanel, border: `1px solid ${C.border}`, cursor: 'pointer',
                  padding: '3px 8px', fontFamily: FONTS.mono, fontSize: 10.5, color: C.fg,
                }}>{PARAM_DEFS[rk].symbol}<span style={{ color: C.fgMuted }}> · {PARAM_DEFS[rk].label}</span></button>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// Slider for the trajectory horizon (σ_max). Lives outside PARAM_DEFS because
// it is a view control, not a model parameter.
function HorizonSlider({ value, onChange, T_auto }) {
  return (
    <div style={{ padding: '10px 14px', borderBottom: `1px solid ${C.border}`, background: C.bgPanel }}>
      <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', gap: 8 }}>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, minWidth: 0 }}>
          <span style={{
            fontFamily: FONTS.sans, fontSize: 12.5, color: C.fg, fontWeight: 500,
            whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
          }}>Trajectory horizon</span>
          <span style={{ fontFamily: FONTS.mono, fontSize: 11, color: C.fgMuted }}>σ_max</span>
        </div>
        <span style={{ fontFamily: FONTS.mono, fontSize: 12, color: C.accent, fontWeight: 600, whiteSpace: 'nowrap' }}>
          {value.toFixed(0)}
        </span>
      </div>
      <input
        type="range"
        min={SIGMA_MAX_MIN}
        max={SIGMA_MAX_MAX}
        step={1}
        value={value}
        onChange={e => onChange(parseFloat(e.target.value))}
        style={{ width: '100%', marginTop: 6, accentColor: C.accent }}
      />
      <div style={{ fontFamily: FONTS.sans, fontSize: 11, color: C.fgDim, marginTop: 2, lineHeight: 1.4 }}>
        How many e-foldings of A the trajectory view simulates
        (≈ {fmtYears(sigmaToYears(value, T_auto))} calendar years at current T_auto).
      </div>
    </div>
  );
}

function GroupHeader({ title }) {
  return (
    <div style={{
      fontFamily: FONTS.mono, fontSize: 10.5, letterSpacing: 1.2,
      color: C.fgMuted, textTransform: 'uppercase',
      padding: '14px 14px 6px', borderBottom: `1px solid ${C.border}`,
      background: C.bgInset,
    }}>{title}</div>
  );
}

function PresetBar({ params, onApply }) {
  return (
    <div style={{ padding: '10px 14px', borderBottom: `1px solid ${C.border}`, background: C.bgPanel }}>
      <div style={{ display: 'flex', gap: 6 }}>
        {Object.entries(PRESETS).map(([id, preset]) => {
          const active = presetIsActive(params, preset);
          return (
            <button
              key={id}
              onClick={() => onApply(preset)}
              title={preset.blurb}
              style={{
                flex: 1,
                background: active ? C.accent : C.bgInset,
                color: active ? C.bg : C.fg,
                border: `1px solid ${active ? C.accent : C.borderStrong}`,
                padding: '6px 4px',
                fontFamily: FONTS.mono, fontSize: 11, fontWeight: 600,
                cursor: 'pointer', letterSpacing: 0.3,
              }}
            >{preset.label}</button>
          );
        })}
      </div>
      <div style={{ fontFamily: FONTS.sans, fontSize: 11, color: C.fgDim, marginTop: 6, lineHeight: 1.4 }}>
        Named calibrations set every slider. Broad/Strict differ only in how the
        observed-misbehaviour evidence is read ({'{'}k_cu, k_hu, q_u(0){'}'}; leakage
        rates via the human-era identity at ℓ = 0.2, ℓ_k = 1 — Broad trend-adjusted
        k = 0.0618, Strict steady-state k = 0.0085); AI 2027 is Broad with high
        leakage (k_cu = 0.9). ℓ_k itself is unanchored — sweep it.
      </div>
    </div>
  );
}

function BasinBadge({ basin, p }) {
  const { kind, qStable, qSaddle, lOstar0, kuuExcess } = basin;
  const palette = {
    escape: { c: C.escape, label: 'NO COOPERATIVE BASIN' },
    monostable: { c: C.monostable, label: 'MONOSTABLE' },
    bistable: { c: C.cooperative, label: 'COOPERATIVE BASIN EXISTS' },
  };
  const { c, label } = palette[kind] || palette.escape;
  const saddleText = 'escapes if q_u > ';
  return (
    <div style={{
      display: 'flex', alignItems: 'stretch', gap: 0,
      border: `1px solid ${c}`, background: C.bgInset,
    }}>
      <div style={{
        background: c, color: C.bg, padding: '8px 14px',
        fontFamily: FONTS.mono, fontSize: 11, fontWeight: 600, letterSpacing: 0.5,
        display: 'flex', alignItems: 'center',
      }}>{label}</div>
      <div style={{
        padding: '8px 14px', fontFamily: FONTS.mono, fontSize: 11.5, color: C.fg,
        display: 'flex', alignItems: 'center', gap: 18, flexWrap: 'wrap',
      }}>
        {qStable !== null && (
          <span><span style={{ color: C.fgMuted }}>stable at q_u = </span><span style={{ color: C.cooperative, fontWeight: 600 }}>{qStable.toFixed(3)}</span></span>
        )}
        {qSaddle !== null && (
          <span><span style={{ color: C.fgMuted }}>{saddleText}</span><span style={{ color: C.bistable, fontWeight: 600 }}>{qSaddle.toFixed(3)}</span></span>
        )}
        <span style={{ color: C.fgMuted, fontSize: 10.5 }} title={`k_cu → 0 reduction of Condition 1: cooperative basin needs ℓ·O*(0) ≥ k_uu − 1. O*(0) is the v4 implicit long-run root — under the dial pin it equals the O* slider exactly. Here ℓ·O*(0) = ${lOstar0.toFixed(2)}, k_uu − 1 = ${kuuExcess.toFixed(2)}; c_M = ${p.c_M?.toFixed?.(3) ?? 'n/a'}; c_0 = ${p.c_0?.toFixed?.(3) ?? 'n/a'} (c_0/c_M from the per-cell pin)`}>
          ℓ·O*(0) = {lOstar0.toFixed(2)} vs k_uu−1 = {kuuExcess.toFixed(2)},  c_M = {p.c_M.toFixed(3)},  c_0 = {p.c_0.toFixed(3)}
        </span>
      </div>
    </div>
  );
}

function ViewTabs({ view, setView }) {
  const tabs = [
    { id: 'trajectory', label: 'Trajectory' },
    { id: 'outcome', label: 'Outcome map' },
  ];
  return (
    <div style={{ display: 'flex', borderBottom: `1px solid ${C.borderStrong}`, background: C.bgPanel }}>
      {tabs.map(t => (
        <button
          key={t.id}
          onClick={() => setView(t.id)}
          style={{
            background: view === t.id ? C.bg : 'transparent',
            color: view === t.id ? C.accent : C.fgDim,
            border: 'none',
            borderBottom: view === t.id ? `2px solid ${C.accent}` : '2px solid transparent',
            padding: '10px 18px',
            fontFamily: FONTS.serif, fontSize: 13.5,
            cursor: 'pointer',
            letterSpacing: 0.2,
          }}
        >{t.label}</button>
      ))}
    </div>
  );
}

function PlotTitle({ title, subtitle }) {
  return (
    <div style={{ marginBottom: 6 }}>
      <div style={{ fontFamily: FONTS.serif, fontSize: 13, color: C.fg, fontWeight: 600 }}>{title}</div>
      {subtitle && <div style={{ fontFamily: FONTS.mono, fontSize: 10.5, color: C.fgMuted, marginTop: 1 }}>{subtitle}</div>}
    </div>
  );
}

const tooltipStyle = {
  background: C.bgInset,
  border: `1px solid ${C.borderStrong}`,
  fontFamily: FONTS.mono,
  fontSize: 11,
  color: C.fg,
};

// Format helpers
const fmt2 = v => v.toFixed(2);
const fmt3 = v => v.toFixed(3);
const fmtYears = v => v < 10 ? v.toFixed(1) : Math.round(v).toString();
const fmtPct = v => (v * 100).toFixed(1) + '%';
const fmtSigma = v => v < 10 ? v.toFixed(1) : Math.round(v).toString();

// =============================================================================
// VIEW: TRAJECTORY
// =============================================================================

function TrajectoryView({ params, basin, sigmaMax, displayMode }) {
  const ode = useMemo(() => odeParams(params), [params]);
  const ic = useMemo(() => buildInitialState(params), [params]);
  const traj = useMemo(() => simulate(ode, ic, sigmaMax), [ode, ic, sigmaMax]);

  const data = useMemo(() => traj.samples.map(s => {
    const [Q, m, e, eta] = s.state;
    const x = displayMode === 'years' ? sigmaToYears(s.sigma, params.T_auto) : s.sigma;
    const denom = 1 + eta;
    return {
      x,
      Q,
      O: s.O,
      apparent: s.O * Q,   // observed/apparent uncoop rate = observability × true share
      coopShare: (1 - Q) / denom,
      uncoopShare: Q / denom,
      humanShare: eta / denom,
      G: s.G,
    };
  }), [traj, params.T_auto, displayMode]);

  const xLabel = displayMode === 'years' ? 'calendar years' : 'σ (e-foldings of A)';
  const xTickFormatter = displayMode === 'years' ? fmtYears : fmtSigma;

  const plotStyle = {
    background: C.bgInset,
    border: `1px solid ${C.border}`,
    padding: '14px 14px 8px',
  };

  const refLines = [];
  if (basin.qStable !== null) refLines.push({ y: basin.qStable, c: C.cooperative, label: 'stable' });
  if (basin.qSaddle !== null) refLines.push({ y: basin.qSaddle, c: C.bistable, label: 'saddle' });

  // AI–human parity (A = H₀): η = η₀·e^{-σ} = 1 at σ = ln η₀. Mark it on the
  // σ-axis; if it falls past the right edge, indicate with an arrow not a line.
  const paritySigma = Math.log(Math.max(params.eta0, 1e-9));
  const parityX = displayMode === 'years' ? sigmaToYears(paritySigma, params.T_auto) : paritySigma;
  const xMax = data.length ? data[data.length - 1].x : 0;
  const parityOffRight = parityX > xMax + 1e-9;
  const parityRef = (key) => (
    <ReferenceLine key={key}
      x={parityOffRight ? xMax : parityX}
      stroke={C.share_human}
      strokeOpacity={parityOffRight ? 0.5 : 0.85}
      strokeDasharray={parityOffRight ? '1 3' : '5 3'}
      label={{
        value: parityOffRight ? `AI=H₀ at σ=${fmtSigma(paritySigma)} →` : 'AI=H₀',
        position: parityOffRight ? 'insideTopRight' : 'top',
        fill: C.share_human, fontSize: 9, fontFamily: FONTS.mono,
      }} />
  );

  const finalSample = traj.samples[traj.samples.length - 1];
  const [qF, , , etaF] = finalSample.state;
  const yearsTotal = sigmaToYears(finalSample.sigma, params.T_auto);
  const finalDenom = 1 + etaF;
  const finalCoop = (1 - qF) / finalDenom;
  const finalUncoop = qF / finalDenom;
  const finalHuman = etaF / finalDenom;

  const outcome = traj.escaped ? `escape (${traj.escapeReason})`
    : basin.qStable !== null && qF < Math.min(basin.qStable * 2 + 0.02, (basin.qStable + (basin.qSaddle || 1)) / 2) ? 'converging to cooperative'
    : basin.qSaddle !== null && qF > basin.qSaddle ? 'beyond saddle (escaping)'
    : 'transient';

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
      <div style={plotStyle}>
        <PlotTitle title="Uncooperative share: true vs apparent" subtitle="solid = true q_u · dashed = apparent O·q_u (what monitoring sees)" />
        <ResponsiveContainer width="100%" height={180}>
          <LineChart data={data} margin={{ top: 18, right: 12, left: 0, bottom: 20 }}>
            <CartesianGrid stroke={C.grid} strokeDasharray="2 4" />
            <XAxis dataKey="x" type="number" domain={['dataMin', 'dataMax']} stroke={C.fgMuted} tick={{ fontSize: 10, fontFamily: FONTS.mono, fill: C.fgDim }}
                   tickFormatter={xTickFormatter}
                   label={{ value: xLabel, position: 'insideBottom', offset: -10, fontSize: 10, fontFamily: FONTS.mono, fill: C.fgMuted }} />
            <YAxis stroke={C.fgMuted} tick={{ fontSize: 10, fontFamily: FONTS.mono, fill: C.fgDim }}
                   domain={[0, 1]} tickFormatter={v => v.toFixed(1)} />
            <Tooltip contentStyle={tooltipStyle}
                     labelFormatter={v => xTickFormatter(+v) + ' ' + xLabel}
                     formatter={(v) => v.toFixed(3)} />
            <Legend verticalAlign="top" align="right" iconType="plainline" iconSize={14}
                    wrapperStyle={{ fontFamily: FONTS.mono, fontSize: 10, color: C.fgDim, paddingBottom: 2 }} />
            {refLines.map((rl, i) => (
              <ReferenceLine key={i} y={rl.y} stroke={rl.c} strokeDasharray="3 3"
                             label={{ value: rl.label, fill: rl.c, fontSize: 10, fontFamily: FONTS.mono, position: 'right' }} />
            ))}
            {parityRef('par-q')}
            <Line type="monotone" dataKey="Q" name="true q_u" stroke={C.trace_q} strokeWidth={1.8} dot={false} isAnimationActive={false} />
            <Line type="monotone" dataKey="apparent" name="apparent O·q_u" stroke={C.apparent} strokeWidth={1.6} strokeDasharray="4 3" dot={false} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div style={plotStyle}>
        <PlotTitle title="Observability O(σ)" subtitle="fraction of uncoop activity that monitoring catches" />
        <ResponsiveContainer width="100%" height={180}>
          <LineChart data={data} margin={{ top: 5, right: 12, left: 0, bottom: 20 }}>
            <CartesianGrid stroke={C.grid} strokeDasharray="2 4" />
            <XAxis dataKey="x" type="number" domain={['dataMin', 'dataMax']} stroke={C.fgMuted} tick={{ fontSize: 10, fontFamily: FONTS.mono, fill: C.fgDim }}
                   tickFormatter={xTickFormatter}
                   label={{ value: xLabel, position: 'insideBottom', offset: -10, fontSize: 10, fontFamily: FONTS.mono, fill: C.fgMuted }} />
            <YAxis stroke={C.fgMuted} tick={{ fontSize: 10, fontFamily: FONTS.mono, fill: C.fgDim }}
                   domain={[0, 1]} tickFormatter={v => v.toFixed(1)} />
            <Tooltip contentStyle={tooltipStyle}
                     labelFormatter={v => xTickFormatter(+v) + ' ' + xLabel}
                     formatter={(v) => v.toFixed(3)} />
            {parityRef('par-o')}
            <Line type="monotone" dataKey="O" stroke={C.trace_O} strokeWidth={1.8} dot={false} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div style={plotStyle}>
        <PlotTitle title="Labour shares" subtitle="coop AI · uncoop AI · humans, as fractions of total labour" />
        <ResponsiveContainer width="100%" height={180}>
          <AreaChart data={data} margin={{ top: 20, right: 12, left: 0, bottom: 20 }}>
            <CartesianGrid stroke={C.grid} strokeDasharray="2 4" />
            <XAxis dataKey="x" type="number" domain={['dataMin', 'dataMax']} stroke={C.fgMuted} tick={{ fontSize: 10, fontFamily: FONTS.mono, fill: C.fgDim }}
                   tickFormatter={xTickFormatter}
                   label={{ value: xLabel, position: 'insideBottom', offset: -10, fontSize: 10, fontFamily: FONTS.mono, fill: C.fgMuted }} />
            <YAxis stroke={C.fgMuted} tick={{ fontSize: 10, fontFamily: FONTS.mono, fill: C.fgDim }}
                   domain={[0, 1]} allowDataOverflow={false} tickFormatter={v => v.toFixed(1)} />
            <Tooltip contentStyle={tooltipStyle}
                     labelFormatter={v => xTickFormatter(+v) + ' ' + xLabel}
                     formatter={(v) => (v * 100).toFixed(1) + '%'} />
            <Legend verticalAlign="top" align="right" iconType="square" iconSize={9}
                    wrapperStyle={{ fontFamily: FONTS.mono, fontSize: 10, color: C.fgDim, paddingBottom: 6 }} />
            {parityRef('par-s')}
            <Area type="monotone" dataKey="uncoopShare" name="uncoop" stackId="1"
                  stroke={C.share_uncoop} fill={C.share_uncoop} fillOpacity={0.55} isAnimationActive={false} />
            <Area type="monotone" dataKey="coopShare" name="coop" stackId="1"
                  stroke={C.share_coop} fill={C.share_coop} fillOpacity={0.55} isAnimationActive={false} />
            <Area type="monotone" dataKey="humanShare" name="humans" stackId="1"
                  stroke={C.share_human} fill={C.share_human} fillOpacity={0.55} isAnimationActive={false} />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      <div style={{ ...plotStyle, display: 'flex', flexDirection: 'column' }}>
        <PlotTitle title="Trajectory readout" subtitle="final state & outcome" />
        <div style={{ padding: '8px 0', fontFamily: FONTS.mono, fontSize: 12, color: C.fg, lineHeight: 1.8 }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr', gap: '4px 18px' }}>
            <span style={{ color: C.fgMuted }}>final σ</span><span>{fmt2(finalSample.sigma)}</span>
            <span style={{ color: C.fgMuted }}>final years</span><span>{fmtYears(yearsTotal)}</span>
            <span style={{ color: C.fgMuted }}>final q_u (true)</span><span style={{ color: C.trace_q }}>{fmt3(qF)}</span>
            <span style={{ color: C.fgMuted }}>apparent O·q_u</span><span style={{ color: C.apparent }}>{fmt3(finalSample.O * qF)}</span>
            <span style={{ color: C.fgMuted }}>final O</span><span style={{ color: C.trace_O }}>{fmt3(finalSample.O)}</span>
            <span style={{ color: C.fgMuted }}>coop labour</span><span style={{ color: C.share_coop }}>{fmtPct(finalCoop)}</span>
            <span style={{ color: C.fgMuted }}>uncoop labour</span><span style={{ color: C.share_uncoop }}>{fmtPct(finalUncoop)}</span>
            <span style={{ color: C.fgMuted }}>human labour</span><span style={{ color: C.share_human }}>{fmtPct(finalHuman)}</span>
            <span style={{ color: C.fgMuted }}>growth rate G</span><span style={{ color: finalSample.G > 0 ? C.fg : C.escape }}>{fmt2(finalSample.G)}</span>
          </div>
          <div style={{ marginTop: 14, paddingTop: 10, borderTop: `1px solid ${C.border}`, fontFamily: FONTS.sans, fontSize: 12 }}>
            <div style={{ color: C.fgMuted, fontSize: 11, fontFamily: FONTS.mono, letterSpacing: 1, textTransform: 'uppercase' }}>Outcome</div>
            <div style={{
              marginTop: 4,
              color: outcome.startsWith('escape') || outcome.includes('escaping') ? C.escape : outcome.includes('cooperative') ? C.cooperative : C.fgDim,
              fontWeight: 500, fontSize: 14,
            }}>{outcome}</div>
          </div>
        </div>
      </div>
    </div>
  );
}

// =============================================================================
// VIEW: OUTCOME MAP
// =============================================================================

const SWEEPABLE_KEYS = ['Ostar', 'k_uu', 'k_cu', 'k_hu', 'l', 'lk', 'delta', 'a_e_m', 'a_ai_h'];

function classifySim(traj, basin) {
  if (traj.escaped) return 'escape';
  const qF = traj.samples[traj.samples.length - 1].state[0];
  if (basin.qStable !== null) {
    const threshold = basin.qSaddle !== null
      ? (basin.qStable + basin.qSaddle) / 2
      : basin.qStable * 2 + 0.02;
    if (qF < threshold) return 'cooperative';
  }
  if (basin.qSaddle !== null && qF > basin.qSaddle) return 'escape';
  return 'unclear';
}

function OutcomeMapView({ params }) {
  const [xKey, setXKey] = useState('k_uu');
  const [yKey, setYKey] = useState('l');
  const [resolution, setResolution] = useState(25);
  const sigmaMax = 40;  // hardcoded horizon for the outcome map

  const xDef = PARAM_DEFS[xKey];
  const yDef = PARAM_DEFS[yKey];

  const grid = useMemo(() => {
    const N = resolution;
    const cells = new Array(N);
    for (let i = 0; i < N; i++) {
      cells[i] = new Array(N);
      const y = yDef.min + (i / (N - 1)) * (yDef.max - yDef.min);
      for (let j = 0; j < N; j++) {
        const x = xDef.min + (j / (N - 1)) * (xDef.max - xDef.min);
        const p = { ...params, [xKey]: x, [yKey]: y };
        const ode = odeParams(p);
        const b = classifyBasin(ode);
        const ic = buildInitialState(p);
        const traj = simulate(ode, ic, sigmaMax, 0.02, 30);
        const outcome = classifySim(traj, b);
        cells[i][j] = { x, y, outcome, bistable: b.kind === 'bistable' };
      }
    }
    return cells;
  }, [params, xKey, yKey, resolution, sigmaMax]);

  // Numeric boundary overlay: zero level set of the margin function
  // −min_r g(r) over the sweep plane (each sample recalibrated like the grid
  // cells, including the per-cell c_0 pin). Traced by scanning columns and
  // rows for sign changes, then bisecting. One curve: the basin-existence
  // threshold (black). (The v2 eradication overlay stays retracted — no
  // eradication regime exists under the bounded gate, audit V4.5.)
  const traceBoundary = (marginFn) => {
    const margAt = (x, y) =>
      marginFn(odeParams({ ...params, [xKey]: x, [yKey]: y }));
    const pts = [];
    const trace = (FIXED, SCAN, fixedDef, scanDef, fixedIsX) => {
      for (let j = 0; j <= FIXED; j++) {
        const u = fixedDef.min + (j / FIXED) * (fixedDef.max - fixedDef.min);
        let vPrev = scanDef.min;
        let mPrev = fixedIsX ? margAt(u, vPrev) : margAt(vPrev, u);
        for (let i = 1; i <= SCAN; i++) {
          const v = scanDef.min + (i / SCAN) * (scanDef.max - scanDef.min);
          const m = fixedIsX ? margAt(u, v) : margAt(v, u);
          // sign change, treating exact zeros as crossings (a boundary can
          // land exactly on a scan grid point)
          const crossed = (mPrev < 0 && m >= 0) || (mPrev > 0 && m <= 0);
          if (Number.isFinite(mPrev) && Number.isFinite(m) && crossed) {
            let lo = vPrev, hi = v, mLo = mPrev;
            for (let it = 0; it < 25; it++) {
              const mid = 0.5 * (lo + hi);
              const mm = fixedIsX ? margAt(u, mid) : margAt(mid, u);
              if (mLo * mm <= 0) hi = mid;
              else { lo = mid; mLo = mm; }
            }
            const vc = 0.5 * (lo + hi);
            pts.push(fixedIsX ? { x: u, y: vc } : { x: vc, y: u });
          }
          vPrev = v; mPrev = m;
        }
      }
    };
    trace(150, 160, xDef, yDef, true);   // column scan (crossings in y)
    trace(150, 160, yDef, xDef, false);  // row scan (crossings in x)
    return pts;
  };
  const boundaryPts = useMemo(
    () => traceBoundary(p => BASIN_BOUNDARY.margin(p)),
    [params, xKey, yKey, xDef, yDef]);

  const W = 540, H = 540;
  const cellW = W / resolution, cellH = H / resolution;

  const colorFor = (cell) => {
    if (cell.outcome === 'cooperative') return C.cooperative;
    if (cell.outcome === 'escape') return C.escape;
    return C.fgMuted;
  };

  const xPos = (params[xKey] - xDef.min) / (xDef.max - xDef.min);
  const yPos = (params[yKey] - yDef.min) / (yDef.max - yDef.min);
  const inRange = xPos >= 0 && xPos <= 1 && yPos >= 0 && yPos <= 1;

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 280px', gap: 12 }}>
      <div style={{ background: C.bgInset, border: `1px solid ${C.border}`, padding: 14 }}>
        <PlotTitle title="Outcome over parameter sweep" subtitle={`each cell recalibrates c_M and c_0 at its (x, y) values; final q_u after σ = ${sigmaMax}`} />
        <svg width={W + 70} height={H + 50} style={{ display: 'block', marginTop: 8 }}>
          <g transform={`translate(58, 10)`}>
            {grid.map((row, i) =>
              row.map((cell, j) => (
                <rect
                  key={`${i}-${j}`}
                  x={j * cellW}
                  y={(resolution - 1 - i) * cellH}
                  width={cellW + 0.5}
                  height={cellH + 0.5}
                  fill={colorFor(cell)}
                  fillOpacity={cell.outcome === 'cooperative' ? 0.75 : cell.outcome === 'escape' ? 0.65 : 0.3}
                />
              ))
            )}
            {grid.map((row, i) =>
              row.map((cell, j) => cell.bistable && (
                <rect
                  key={`b-${i}-${j}`}
                  x={j * cellW + cellW * 0.38}
                  y={(resolution - 1 - i) * cellH + cellH * 0.38}
                  width={cellW * 0.24}
                  height={cellH * 0.24}
                  fill="none"
                  stroke={C.bg}
                  strokeWidth={0.6}
                />
              ))
            )}
            {boundaryPts.map((pt, i) => (
              <circle
                key={`bd-${i}`}
                cx={((pt.x - xDef.min) / (xDef.max - xDef.min)) * W}
                cy={H - ((pt.y - yDef.min) / (yDef.max - yDef.min)) * H}
                r={1.2}
                fill={C.testpath}
                fillOpacity={0.9}
              />
            ))}
            <rect x={0} y={0} width={W} height={H} fill="none" stroke={C.borderStrong} />
            <text x={W / 2} y={H + 32} textAnchor="middle" fill={C.fg} fontFamily={FONTS.mono} fontSize={11}>
              {xDef.label} ({xDef.symbol})
            </text>
            <text x={-H / 2} y={-44} textAnchor="middle" transform="rotate(-90)" fill={C.fg} fontFamily={FONTS.mono} fontSize={11}>
              {yDef.label} ({yDef.symbol})
            </text>
            <text x={0} y={H + 14} textAnchor="middle" fill={C.fgMuted} fontFamily={FONTS.mono} fontSize={10}>{xDef.min}</text>
            <text x={W} y={H + 14} textAnchor="middle" fill={C.fgMuted} fontFamily={FONTS.mono} fontSize={10}>{xDef.max}</text>
            <text x={-6} y={H + 3} textAnchor="end" fill={C.fgMuted} fontFamily={FONTS.mono} fontSize={10}>{yDef.min}</text>
            <text x={-6} y={6} textAnchor="end" fill={C.fgMuted} fontFamily={FONTS.mono} fontSize={10}>{yDef.max}</text>
            {inRange && (
              <circle cx={xPos * W} cy={H - yPos * H} r={5} fill="none" stroke={C.accent} strokeWidth={2} />
            )}
          </g>
        </svg>
      </div>

      <div style={{ background: C.bgPanel, border: `1px solid ${C.border}` }}>
        <GroupHeader title="Sweep axes" />
        <div style={{ padding: '10px 14px' }}>
          <AxisSelect label="X axis" pkey={xKey} setPkey={setXKey} />
          <div style={{ height: 8 }} />
          <AxisSelect label="Y axis" pkey={yKey} setPkey={setYKey} />
        </div>
        <div style={{ padding: '10px 14px', borderTop: `1px solid ${C.border}` }}>
          <div style={{ fontFamily: FONTS.sans, fontSize: 12, color: C.fg, marginBottom: 6 }}>
            Resolution <span style={{ fontFamily: FONTS.mono, color: C.accent, marginLeft: 6 }}>{resolution} × {resolution}</span>
          </div>
          <input type="range" min={10} max={40} step={5} value={resolution}
                 onChange={e => setResolution(parseInt(e.target.value))}
                 style={{ width: '100%', accentColor: C.accent }} />
          <div style={{ fontFamily: FONTS.sans, fontSize: 11, color: C.fgDim }}>
            {resolution * resolution} cells. Higher = slower.
          </div>
        </div>
        <div style={{ padding: '10px 14px', borderTop: `1px solid ${C.border}` }}>
          <div style={{ fontFamily: FONTS.mono, fontSize: 10.5, letterSpacing: 1.2, color: C.fgMuted, textTransform: 'uppercase', marginBottom: 6 }}>Legend</div>
          <LegendDot color={C.cooperative} label="Cooperative basin reached" />
          <LegendDot color={C.escape} label="Escape" />
          <LegendDot color={C.fgMuted} label="Unclear / transient" />
          <div style={{ marginTop: 6, fontFamily: FONTS.sans, fontSize: 11, color: C.fgDim }}>
            Hollow square ⇒ parameters admit a bistable basin (long-run projection).
          </div>
          <div style={{ marginTop: 6, fontFamily: FONTS.sans, fontSize: 11, color: C.fgDim }}>
            Dotted black curve: {BASIN_BOUNDARY.label} — where the long-run
            interior fixed point appears/disappears. Under v4 the long-run O*
            is implicit, so the curve is the numeric zero level set of
            −min<sub>r</sub> g(r) (g(0) &gt; 0 always, so a basin exists exactly
            when g dips to zero); the v3 closed-form threshold chain is
            superseded — only the δ = 1, k_uu = 1, a_E/M = 1 closed form
            ℓ* = 2k_cu[(T−ck_cu) + √((T−ck_cu)(T−ck_cu−ℓ_k))] survives. The
            attractor inside the boundary can sit at high q_u — it appears at
            the saddle-node, not near zero — check the badge, not just the curve.
            Past the boundary at very high ℓ the attractor falls like
            k_cu(1+c−ℓ_k−cδk_cu)/ℓ: an endemic floor, never exactly zero
            (while 2cδk_cu ≤ 1+c; beyond that bound no finite ℓ produces a
            basin at all).
          </div>
          <div style={{ marginTop: 8, fontFamily: FONTS.sans, fontSize: 11, color: C.fgDim }}>
            Amber ring marks current parameter values.
          </div>
        </div>
      </div>
    </div>
  );
}

function AxisSelect({ label, pkey, setPkey }) {
  return (
    <div>
      <label style={{ fontFamily: FONTS.sans, fontSize: 12, color: C.fgDim, display: 'block', marginBottom: 4 }}>{label}</label>
      <select value={pkey} onChange={e => setPkey(e.target.value)}
              style={{
                width: '100%', background: C.bgInset, color: C.fg, border: `1px solid ${C.border}`,
                padding: '6px 8px', fontFamily: FONTS.mono, fontSize: 12,
              }}>
        {SWEEPABLE_KEYS.map(k => (
          <option key={k} value={k}>{PARAM_DEFS[k].label} ({PARAM_DEFS[k].symbol})</option>
        ))}
      </select>
    </div>
  );
}

function LegendDot({ color, label }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
      <div style={{ width: 12, height: 12, background: color, opacity: 0.75 }} />
      <span style={{ fontFamily: FONTS.sans, fontSize: 11.5, color: C.fg }}>{label}</span>
    </div>
  );
}


// =============================================================================
// ASSUMPTIONS / DOCUMENTATION
// =============================================================================

function DocSection({ title, children }) {
  return (
    <section style={{ marginBottom: 28 }}>
      <h2 style={{
        fontFamily: FONTS.serif, fontSize: 17, fontWeight: 700, color: C.fg,
        margin: '0 0 10px 0', paddingBottom: 6, borderBottom: `1px solid ${C.border}`,
      }}>{title}</h2>
      <div style={{ fontFamily: FONTS.serif, fontSize: 14, color: C.fg, lineHeight: 1.65 }}>
        {children}
      </div>
    </section>
  );
}

function P({ children }) {
  return <p style={{ margin: '0 0 10px 0' }}>{children}</p>;
}

function MV({ children }) {
  // Inline mono for non-LaTeX variable mentions in prose
  return <span style={{ fontFamily: FONTS.mono, fontSize: 13, color: C.fgDim }}>{children}</span>;
}

function AssumptionsPanel({ open, setOpen }) {
  if (!open) return null;
  return (
    <div style={{
      position: 'fixed', inset: 0, background: 'rgba(40,30,10,0.45)', zIndex: 100,
      display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 20,
    }} onClick={() => setOpen(false)}>
      <div onClick={e => e.stopPropagation()} style={{
        background: C.bgInset, border: `1px solid ${C.borderStrong}`,
        width: 'min(820px, 92vw)', maxHeight: '88vh',
        display: 'flex', flexDirection: 'column',
        boxShadow: '0 12px 40px rgba(60,40,10,0.25)',
      }}>
        <div style={{
          padding: '20px 28px 14px', borderBottom: `1px solid ${C.border}`,
          background: C.bgPanel,
          display: 'flex', justifyContent: 'space-between', alignItems: 'baseline',
        }}>
          <div>
            <div style={{ fontFamily: FONTS.serif, fontSize: 22, color: C.fg, fontWeight: 700 }}>
              Model documentation
            </div>
            <div style={{ fontFamily: FONTS.mono, fontSize: 11, color: C.fgMuted, marginTop: 4, letterSpacing: 0.3 }}>
              progress-clocked basin model for cooperative vs uncooperative reproductive automation
            </div>
          </div>
          <button onClick={() => setOpen(false)} style={{
            background: C.accent, color: C.bg, border: 'none',
            padding: '6px 14px', fontFamily: FONTS.mono, fontSize: 11, fontWeight: 600,
            cursor: 'pointer', letterSpacing: 0.5,
          }}>CLOSE</button>
        </div>

        <div style={{ padding: '24px 28px', overflowY: 'auto', flex: 1 }}>

          <DocSection title="What this models">
            <P>
              We posit two reproducing slices of AI activity: a cooperative core <MV>A_c</MV> and an
              uncooperative core <MV>A_u</MV>. These are <em>not</em> "intended behaviour" and "unintended
              behaviour" in a general sense — they are the parts of the automation ecosystem that participate
              in self-reproduction, persistence, and suppression dynamics. Ordinary jank, errors, and bias matter
              here only insofar as they leak into or proxy for <MV>A_u</MV>.
            </P>
            <P>
              The model does <em>not</em> explicitly track AI growth in calendar time. Instead it uses
              a progress clock — log of total automation — that factors out absolute takeoff speed. The
              question it answers is: given an observed mix of monitoring, evasion, leakage, and human input,
              does the cooperative slice retain the majority share as automation grows, or does it lose ground?
            </P>
          </DocSection>

          <DocSection title="State variables">
            <P>
              All four are dimensionless ratios of stocks to total automation <MV>A = A_c + A_u</MV>:
            </P>
            <TeX display>{String.raw`Q = \frac{A_u}{A_c + A_u}, \quad \eta = \frac{H_0}{A_c + A_u}, \quad m = \frac{M}{A_c + A_u}, \quad e = \frac{E}{A_c + A_u}`}</TeX>
            <P>
              <MV>Q</MV> is the uncooperative share (the blog post writes <MV>q_u</MV>). <MV>η</MV> is human
              production as a multiple of automated production (humans are assumed constant <MV>H_0</MV> in
              absolute terms) — this is the same variable the blog post writes as <MV>q_h</MV>; the app keeps
              the <MV>η</MV> name internally and they are identical. <MV>m</MV> and <MV>e</MV>
              are the detect-and-fix coverage and evasion stocks; individually they have no intuitive
              interpretation, but together they determine observability:
            </P>
            <TeX display>{String.raw`O = \frac{m}{m + e}`}</TeX>
            <P>
              <MV>O</MV> is the covered fraction of the current misbehaviour-space — the share of uncoop
              activity that the accumulated detect-and-fix stock catches (and can act on) in real time.
              <MV> m</MV> is <em>not</em> raw monitoring effort: it is the accumulated coverage that
              observation buys. Each observed behaviour adds coverage of its mode; how much coverage one
              observation buys (fix-generalization quality) is the productivity <MV>c_M</MV>. A novel mode
              arrives uncovered — every new unit of automation ships with <MV>c_0</MV> units of
              not-yet-covered surface, which keeps <MV>O*</MV> below 1 — gets observed,
              fixed, and subsequently intercepted — that sequencing is exactly the <MV>m</MV>-vs-<MV>e</MV>
              stock race. (A sharpness exponent <MV>β</MV> on <MV>m</MV> and <MV>e</MV>
              is held fixed at 1 here; tuning it added interpretive burden without changing the qualitative basin
              structure.)
            </P>
          </DocSection>

          <DocSection title="Calendar-time dynamics">
            <P>
              The starting point is a system in calendar time with a shared productivity multiplier
              <MV>Λ(t, …)</MV> that captures takeoff speed:
            </P>
            <TeX display>{String.raw`\dot A_c = \Lambda \cdot A \cdot F_c, \quad \dot A_u = \Lambda \cdot A \cdot F_u, \quad \dot M = \Lambda \cdot A \cdot F_M, \quad \dot E = \Lambda \cdot A \cdot F_E`}</TeX>
            <P>
              The key simplifying assumption is that each <MV>F</MV> depends only on the dimensionless state
              <MV>(Q, η, m, e, O)</MV>, not on time directly and not on <MV>Λ</MV>. <MV>Λ</MV> sets the absolute
              pace; the F's encode how productive capacity is allocated.
            </P>
          </DocSection>

          <DocSection title="σ-clock substitution">
            <P>
              Define the progress clock as the log expansion of total automation since <MV>t = 0</MV>:
            </P>
            <TeX display>{String.raw`\sigma = \log\!\left(\frac{A}{A(0)}\right), \qquad \frac{d\sigma}{dt} = \frac{\dot A}{A} = \Lambda \cdot G \quad \text{where } G = F_c + F_u`}</TeX>
            <P>
              Switching to <MV>σ</MV> as the independent variable cancels <MV>Λ</MV> from the dynamics. Writing
              <MV> ′</MV> for <MV>d/dσ</MV>:
            </P>
            <TeX display>{String.raw`Q' = \frac{F_u}{G} - Q, \quad \eta' = -\eta, \quad m' = \frac{F_M}{G} - m, \quad e' = \frac{F_E}{G} - e`}</TeX>
            <P>
              The model now commits only to how production is <em>allocated</em>, not how fast it happens in
              calendar time.
            </P>
          </DocSection>

          <DocSection title="Functional forms">
            <P>
              Production allocation, written with the leakage inflow <MV>L_k</MV>:
            </P>
            <TeX display>{String.raw`L_k = k_{cu}(1-Q) + k_{hu}\,\eta`}</TeX>
            <TeX display>{String.raw`F_c = (1-Q)(1 - k_{cu}) + \eta(1 - k_{hu}) + (1-\delta)\,O\,(\ell_k L_k + \ell\,Q)`}</TeX>
            <TeX display>{String.raw`F_u = (1 - \ell_k O)\,L_k + (k_{uu} - \ell O)\,Q`}</TeX>
            <P>
              Cooperative AI takes its share <MV>(1-Q)</MV> of productive capacity less leakage <MV>k_cu</MV>;
              humans contribute their share <MV>η</MV> less leakage <MV>k_hu</MV>. We choose units of automation
              production so that one unit equals one unit of human labour — i.e. <MV>H_0 = 1</MV>.
            </P>
            <P>
              Suppression is gated by observability and acts in two places, with two different
              parameters. The <em>stock</em> term <MV>−ℓO·Q</MV> removes established uncoop labour:
              <MV> ℓ</MV> is the fix rate of known failure modes per σ-unit — a <em>rate</em>, which may
              exceed 1. The <em>inflow</em> term gates leakage at production: only the
              fraction <MV>(1−ℓ_k·O)</MV> of leaked effort enters the pool, where <MV>ℓ_k ∈ [0,1]</MV> is
              the interception efficacy at production — a bounded <em>fraction</em>, because interception
              cannot remove more leakage than exists. <MV>ℓ_k</MV> is empirically unanchored (the post
              displays results at 0.5 and 1).
            </P>
            <P>
              The intercepted flow is split by the destruction fraction <MV>δ</MV>: of the intercepted
              flow <MV>O·(ℓ_k·L_k + ℓ·Q)</MV>, fraction <MV>δ</MV> is destroyed outright (filtered,
              blocked, discarded) and fraction <MV>(1−δ)</MV> is redirected into cooperative production
              (retrained, corrected, resampled) — the <MV>+(1−δ)·O·(ℓ_k·L_k + ℓ·Q)</MV> term
              in <MV>F_c</MV>. Redirected labour converts 1:1 with no lag or quality discount; if you think
              filtered-then-retrained labour is worth less than one unit, that would be an extra parameter,
              not a reinterpretation of <MV>δ</MV>.
            </P>
            <P>
              Under the gate, leakage is <em>not a pure transfer</em> between <MV>F_c</MV> and
              <MV> F_u</MV>: the redirected part of suppression still cancels in the sum, but the destroyed
              fraction of the <em>intercepted</em> flow — including intercepted leakage <MV>δO·ℓ_k·L_k</MV> —
              subtracts from <MV>G = F_c + F_u</MV>. The total growth rate is:
            </P>
            <TeX display>{String.raw`G = (1-Q) + \eta + k_{uu}\,Q - \delta\,O\,(\ell_k L_k + \ell\,Q)`}</TeX>
            <P>
              Relative to the ungated model the difference is the <MV>−δO·ℓ_k·L_k</MV> term; at the central
              calibration it is 2–3% of <MV>G</MV>, and it means aggregate growth now depends (weakly)
              on leakage.
            </P>
            <P>Detect-and-fix coverage and evasion dynamics:</P>
            <TeX display>{String.raw`F_M = c_M\left((1-Q)+\frac{\eta}{a_{\mathrm{AI/H}}}\right), \qquad F_E = c_0\,G + a_{\mathrm{E/M}}c_M\,Q`}</TeX>
            <P>
              Coverage is produced by cooperative AI and humans. The arbitrary units of <MV>m</MV> and
              <MV>e</MV> are fixed by setting cooperative AI as the baseline producer:
              <MV>c_M</MV> is the calibrated coverage-production rate (how much detect-and-fix coverage one
              unit of cooperative labour — and the observations it makes — buys), and
              <MV>a_AI/H</MV> says how much better AI is than humans at that work. Evasion has two parts
              with deliberately different units. Active evasion <MV>a_E/M·c_M·Q</MV> is a labour share,
              like monitoring. Passive opacity is instead <em>pegged to automation growth</em>: every new
              unit of automation arrives carrying <MV>c_0</MV> units of not-yet-covered failure
              surface — <MV>dE_passive = c_0·dA</MV>, which is the <MV>c_0·G</MV> term (it is also where
              novel, not-yet-covered modes enter). Substituting into the state equation gives the form the
              engine integrates:
            </P>
            <TeX display>{String.raw`e' = \frac{F_E}{G} - e = c_0 + \frac{a_{\mathrm{E/M}}c_M\,Q}{G} - e`}</TeX>
            <P>
              The growth peg is a substantive specification choice. If we produce more AI with less AI
              labour going into it, a labour-share spec would say we get <em>less</em> opacity coming
              out — that seems wrong; opacity should be pegged to the amount of automation. A
              production-share alternative (<MV>e′ = (c_0 + a_E/M·c_M·Q)/G − e</MV>, passive opacity
              diluted by total production) implies a human-era observability subsidy — the large human-era
              <MV> G</MV> dilutes passive opacity several-fold — and a post-handoff decline in
              observability even at low <MV>q_u</MV>; the growth peg removes both, and it is what makes
              the σ-clock's timeline-robustness fully carry over to the observability sector.
            </P>
            <P>
              The <MV>-m</MV> and <MV>-e</MV> terms in the state equations are not an additional obsolescence
              parameter. They just come from measuring coverage and evasion stocks relative to total
              automation: a fixed stock shrinks in <MV>M/A</MV> units as <MV>A</MV> grows.
            </P>
          </DocSection>

          <DocSection title="Asymmetries">
            <P>
              <strong>Leakage is one-way.</strong> There is no <MV>k_uc</MV> term — nothing flows back from
              uncoop into cooperative. The justification is that cooperative is initially dominant, so many
              distinct uncoop configurations could each appropriate from <MV>A_c</MV>, but only one specific
              configuration of <MV>A_u</MV> (the cooperative target) could appropriate back. This is a real
              substantive assumption, and gets shakier once <MV>F_u</MV> and <MV>F_c</MV> are similar in
              magnitude.
            </P>
            <P>
              <strong>Only the uncooperative side faces observability-gated suppression.</strong> There is no
              analogous <MV> -ℓO</MV> term acting on cooperative production itself. Similar reasoning:
              cooperative AI operates in the open while uncoop operates covertly, so observability is the
              bottleneck on how much of the uncoop slice we can suppress. The suppression acts on both the
              stock and the flows into it, but the two terms are different kinds of object by construction:
              the self-term <MV>−ℓO·Q</MV> is a <em>rate</em> on the stock and can exceed the
              self-inflow <MV>k_uu·Q</MV> whenever <MV>ℓO &gt; k_uu</MV>, while interception of leakage is a
              bounded <em>fraction</em> <MV>ℓ_k·O ≤ 1</MV> of the flow and can never remove more than the
              leakage itself. That bound is why exact eradication is impossible here: the
              seeding <MV>g(0) = k_cu(1 − ℓ_k·O*(0))</MV> stays strictly positive (novel modes
              keep <MV>O*(0) &lt; 1</MV>), so very high <MV>ℓ</MV> pushes the endemic attractor down
              like <MV>k_cu(1+c−ℓ_k−cδk_cu)/ℓ</MV> but never to zero.
            </P>
          </DocSection>

          <DocSection title="Calibration">
            <P>
              The observability sector needs three inputs: present observability <MV>O(0)</MV>, the long-run
              passive-opacity ratio <MV>c_0/c_M</MV>, and an overall scale for <MV>c_M</MV>. The blog's main
              method fixes the first two from evidence/priors and pegs the third to the automation scale:
            </P>
            <ul style={{ margin: '0 0 10px 18px', padding: 0 }}>
              <li><MV>R(0)</MV> — misses per catch today, i.e. present observability. With <MV>m_0 = 1</MV> fixing the arbitrary stock units this is the initial evasion stock, <MV>e_0 = R_0</MV>, so <MV>O(0) = 1/(1+R_0)</MV>. Central estimate <MV>O(0) = 1/2</MV> (so <MV>R(0) = 1</MV>).</li>
              <li><MV>O*</MV> — best-case long-run observability, set directly. Observability drifts from its present value <MV>O(0)</MV> toward this; internally it pins the passive-opacity ratio at the long-run rare-uncooperative point (<MV>q_h = 0</MV>, <MV>q_u → 0</MV>), where the implicit long-run observability back-solves to <MV>c_0/c_M = (1−O*)/(O*(1−δ·ℓ_k·k_cu·O*))</MV> — re-solved per parameter cell (it depends on <MV>δ</MV>, <MV>ℓ_k</MV>, <MV>k_cu</MV> but not <MV>ℓ</MV>); central <MV>O* = 1/2</MV> gives <MV>c_0/c_M ≈ 1.02</MV> at Broad. "Best case" because it holds while uncoop stays low — the actual long-run <MV>O*(q_u)</MV> falls as <MV>q_u</MV> grows.</li>
              <li><MV>T_auto</MV> — present calendar doubling time of <MV>A</MV>. Under the main method this only scales the calendar-years axis; it no longer enters <MV>c_M</MV>.</li>
            </ul>
            <P>The calibrated rates the simulator uses are then:</P>
            <TeX display>{String.raw`c_M = \frac{G_0\,m_0}{(1-Q_0)+\eta_0/a_{\mathrm{AI/H}}}, \qquad c_0 = \frac{1-O^{\ast}}{O^{\ast}\,(1-\delta\,\ell_k k_{cu}\,O^{\ast})}\,c_M`}</TeX>
            <P>
              where <MV>G_0</MV> is the <em>gated</em> total at <MV>σ = 0</MV>,
              <MV> G_0 = (1−Q_0) + η_0 + k_uu·Q_0 − δ·O(0)·(ℓ_k·L_k(0) + ℓ·Q_0)</MV>. The first equation
              pegs <MV>c_M</MV> to the automation production scale, so coverage starts at
              quasi-steady-state; its magnitude affects only how quickly <MV>O</MV> relaxes from <MV>O(0)</MV>
              toward <MV>O*</MV>, not the long-run basin. Observability therefore drifts from its measured
              present value <MV>O(0)</MV> to its prior-set long-run value <MV>O*</MV>.
            </P>
            <P>
              The "calendar years" toggle uses <MV>years = σ · T_auto / ln 2</MV>, exact at <MV>σ = 0</MV> and
              approximate later because we deliberately do not model how <MV>Λ</MV> evolves — what the σ-clock
              factors out.
            </P>
            <P>
              <strong>Alternative route (not used by default).</strong> The blog keeps a second elicitation in
              an appendix that instead derives <MV>c_M</MV>, <MV>c_0</MV> from two calendar rate judgements —
              <MV> T_M,½</MV> (time for missed-to-caught odds <MV>R = (1−O)/O</MV> to halve with evasion frozen)
              and <MV>T_E,2</MV> (time to double with monitoring frozen). Odds are the natural target there
              because, differentiating along the σ-clock, the dilution terms cancel:
            </P>
            <TeX display>{String.raw`\frac{d\log R}{d\sigma} = \frac{F_E}{F\,e} - \frac{F_M}{F\,m}`}</TeX>
            <P>
              so passive dilution does not change which side is ahead — and, as a corollary, a <em>constant</em>
              leakage rate drops out of the odds dynamics entirely; only a <em>state-dependent</em> leakage
              (e.g. a policy response to observed misbehaviour) would move the basin.
            </P>
          </DocSection>

          <DocSection title="Long-run behaviour & basin conditions">
            <P>
              After the transient (<MV>η → 0</MV>, monitoring and evasion at quasi-steady-state), the
              growth peg keeps total production <MV>F</MV> in the evasion stock:
              <MV> m* = c_M·q_c/F</MV> but <MV>e* = c_0 + a_E/M·c_M·q_u/F</MV>, so <MV>F</MV> no longer
              cancels in <MV>e*/m*</MV>. Because <MV>F</MV> is itself linear in <MV>O</MV>, long-run
              observability at a fixed uncoop share is <em>implicit</em> — the root of a quadratic
              (with <MV>r = q_u/(1−q_u)</MV>, <MV>c = c_0/c_M</MV>):
            </P>
            <TeX display>{String.raw`c\,\delta P\,O^2 - D\,O + 1 = 0, \qquad P = \ell_k k_{cu} + \ell r, \quad D = 1 + c + (a_{\mathrm{E/M}} + c\,k_{uu})\,r`}</TeX>
            <P>
              <MV>O*(q_u)</MV> is the minus root <MV>2/(D + √(D² − 4cδP))</MV> (the plus root is
              infeasible — it sits at <MV>O &gt; 1</MV> or outside the <MV>F &gt; 0</MV> envelope). It
              falls when the uncoop share grows, when active evasion is more productive, or when passive
              opacity <MV>c_0</MV> is large relative to monitoring productivity <MV>c_M</MV> — and, new
              with the peg, it <em>rises</em> slightly with suppression (<MV>dO*/dℓ &gt; 0</MV>,
              <MV> dO*/dδ &gt; 0</MV>): heavier suppression slows growth per unit labour, and growth is
              what brings passive opacity. When <MV>D² &lt; 4cδP</MV> (deep-suppression corners) the
              observability sector has no interior equilibrium at all — a validity-envelope exit.
            </P>
            <P>
              Two thresholds organise the qualitative outcome. <strong>Condition 1</strong> — can rare uncoop
              labour be suppressed while observability is still high? The seeding itself is gated, but
              bounded — when rare, uncoop production starts from
              <MV> g(0) = k_cu(1−ℓ_k·O*(0))</MV>, which is <em>strictly positive</em> for <MV>ℓ_k ≤ 1</MV> —
              and the initial-slope condition for the uncoop share to shrink while rare becomes
            </P>
            <TeX display>{String.raw`O^{\ast}(0)\Big[\ell + (1-\delta)\,\ell_k k_{cu}\Big] + O^{\ast\prime}(0)\,\ell_k k_{cu} > k_{uu} + k_{cu} - 1`}</TeX>
            <P>
              where <MV>O*′(0) &lt; 0</MV> comes from implicit differentiation of the quadratic — no
              closed bracket survives the implicit <MV>O*</MV> (at the central calibrations the effective
              bracket multiplying <MV>O*(0)</MV> is ≈ 0.79 at <MV>ℓ_k = 1</MV>, ≈ 0.93 at
              <MV> ℓ_k = 0.5</MV>). At <MV>δ = 1</MV> this condition is necessary (though not sufficient)
              for a cooperative basin in every case checked numerically — the production-share model's
              symbolic proof does not carry over. For
              <MV> δ &lt; 1</MV> it loses necessity: redirected suppression can sustain an interior attractor
              even where rare uncoop labour initially grows — though that attractor may sit at high <MV>q_u</MV>.
              <strong> Condition 2</strong> — can uncoop labour hold the whole system? Redirection makes this
              strictly harder, because what little is still caught keeps re-seeding the cooperative pool. The
              all-uncoop endpoint is stable when
            </P>
            <TeX display>{String.raw`k_{uu} + k_{cu} - 1 > \frac{(1-\delta)\,\ell}{a_{\mathrm{E/M}} + (c_0/c_M)\,k_{uu}}`}</TeX>
            <P>
              (at <MV>δ = 1</MV> this reduces to <MV>k_uu + k_cu &gt; 1</MV>; at <MV>k_uu = 1</MV> the
              denominator is the familiar <MV>a_E/M + c_0/c_M</MV>). Neither gate moves this
              condition: the gated terms vanish to first order at the all-uncoop endpoint, and
              <MV> ℓ_k</MV> does not appear at fixed <MV>c_0/c_M</MV> (the dial pin moves <MV>c_0</MV>
              imperceptibly). The combinations:
            </P>
            <ul style={{ margin: '0 0 10px 18px', padding: 0 }}>
              <li><strong>Interior attractor exists and C2 ✓</strong> → <em>bistable</em>: a cooperative-side basin coexists with an uncooperative-dominant one, and initial conditions decide where you land.</li>
              <li><strong>No interior attractor and C2 ✓</strong> → the cooperative basin disappears, leaving uncoop dominance (escape).</li>
              <li><strong>C2 ✗</strong> → the all-uncoop endpoint is unstable and a single interior attractor is the only long-run outcome (monostable) — but for <MV>δ &lt; 1</MV> it can sit anywhere from a few percent to near-total uncoop share, so check its location.</li>
            </ul>
            <P>
              There is no eradication case: because <MV>ℓ_k ≤ 1</MV> (and novel modes
              keep <MV>O*(0) &lt; 1</MV> — under the implicit quadratic this is
              <MV> h(1) = c(δℓ_k·k_cu − 1) &lt; 0</MV>), the seeding <MV>g(0) &gt; 0</MV> always,
              so <MV>q_u = 0</MV> is never reachable. At very high <MV>ℓ</MV> the attractor instead falls
              along the deep-suppression endemic floor <MV>q* ≈ k_cu(1+c−ℓ_k−cδk_cu)/ℓ</MV> — arbitrarily
              low, never zero. The floor itself has a validity bound, <MV>2cδk_cu ≤ 1+c</MV> (at the
              central <MV>O* = ½</MV> pin: <MV>δ·k_cu ≤ 4/(4+ℓ_k)</MV>); beyond it the would-be floor sits
              on the quadratic's infeasible branch and deep suppression cannot floor the system at all —
              no finite <MV>ℓ</MV> produces a basin (the AI-2027 <MV>δ = 1</MV> corner).
            </P>
            <P>
              The exact interior fixed points solve a <em>cubic</em> in the odds <MV>r = q_u/(1−q_u)</MV>:
              substituting <MV>g = 0</MV> into the <MV>O</MV>-quadratic gives
              <MV> Φ(r) = cδN² − D·N·S + P·S² = 0</MV> with <MV>N = k_cu + b·r</MV>,
              <MV> S = 1 + (1−δ)r</MV>, <MV>b = k_uu + k_cu − 1</MV>, valid on the minus-root branch; the
              app root-finds <MV>g</MV> with the implicit <MV>O*</MV> directly. The seeding constant
              <MV> Φ(0) = −k_cu(1+c−ℓ_k−cδk_cu)</MV> is ℓ-free and strictly negative for
              <MV> ℓ_k ≤ 1</MV> — the no-eradication statement in cubic form. The positive roots are
              the stable attractor (lower) and the basin-separating saddle (higher) — which is what the badge
              and the dashed lines in the trajectory views report. Existence is monotone in <MV>ℓ</MV>,
              with threshold <MV>ℓ*</MV> drawn on the outcome map — computed numerically under v4 (the
              production-share model's closed-form threshold chain is superseded); the surviving closed
              form, at <MV>δ = 1</MV>, <MV>k_uu = 1</MV>, <MV>a_E/M = 1</MV>:
              <MV> ℓ* = 2k_cu[(T−ck_cu) + √((T−ck_cu)(T−ck_cu−ℓ_k))]</MV>, <MV>T = 1+c</MV> — the old form
              with <MV>T</MV> discounted by the F-feedback term <MV>ck_cu</MV>.
              (The badge compares
              <MV> ℓ·O*(0)</MV> against <MV>k_uu − 1</MV> — the <MV>k_cu → 0</MV> reduction of Condition 1,
              which stays δ-free at <MV>k_cu = 0</MV>.)
            </P>
          </DocSection>

          <DocSection title="Validity envelope">
            <P>
              The progress clock requires <MV>G &gt; 0</MV>, i.e. total automation must be growing. When
              <MV> G ≤ 0</MV>, total automation is stagnant or contracting — the σ-clock framing no longer
              applies. Trajectories that hit this are flagged as escape with reason "G ≤ 0" and the
              integrator halts. This is not a numerical artefact to suppress; it is the model honestly
              saying "you are outside the regime I was designed for".
            </P>
          </DocSection>

          <DocSection title="Caveats">
            <P>
              <strong>Constant leakage.</strong> For simplicity we hold <MV>k_cu</MV> and <MV>k_hu</MV> fixed.
              In reality they could shrink (improved ability to make non-leaky systems) or grow (larger
              attack surface). Time-varying leakage is a natural extension.
            </P>
            <P>
              <strong>What counts as A_u.</strong> We propose using mundane examples of AI acting in costly
              unintended ways as a crude proxy for the <MV>A_u</MV> share — and by extension for parameters
              like <MV>k_hu</MV>. This is valid if there is rough independence between (a) the rate at which
              AI takes unintended actions and (b) the degree to which those actions are concerned with
              self-reproduction. (a) and (b) are coupled by <em>incentives</em> — reproductive misbehaviour
              is differently incentivised from non-reproductive — but not obviously by <em>algorithms</em>:
              the underlying learned patterns that produce one might produce the other at proportional rates.
              The proxy is valid iff the algorithmic channel dominates the incentive channel as the actual
              cause of unintended self-reproducing behaviour. This is empirically open.
            </P>
            <P>
              <strong>1D vs 4D basin.</strong> The displayed fixed points <MV>q_u*_stable</MV> and
              <MV> q_u*_saddle</MV> come from the long-run projection where <MV>q_h → 0</MV> and monitoring /
              evasion are at quasi-steady-state. The full basin boundary in <MV>(q_u, m, e, q_h)</MV> is a 3D
              manifold; trajectories starting above the projected saddle but with high observability and
              significant <MV>q_h</MV> can still return to the cooperative basin.
            </P>
          </DocSection>

        </div>
      </div>
    </div>
  );
}

// =============================================================================
// MAIN
// =============================================================================

export default function BasinExplorer() {
  useFonts();
  const katexReady = useKaTeXLoader();

  // Initial state comes from the URL query string (shareable links).
  const initialState = useMemo(readStateFromURL, []);
  const [params, setParams] = useState(initialState.params);
  const [view, setView] = useState(initialState.view);
  const [displayMode, setDisplayMode] = useState(initialState.displayMode);
  const [assumptionsOpen, setAssumptionsOpen] = useState(false);
  const [pinnedKey, setPinnedKey] = useState(null);
  const [sigmaMax, setSigmaMax] = useState(initialState.sigmaMax);

  // Keep the URL in sync so the current view is always shareable.
  useEffect(() => {
    writeStateToURL(params, sigmaMax, view, displayMode);
  }, [params, sigmaMax, view, displayMode]);

  const ode = useMemo(() => odeParams(params), [params]);
  const basin = useMemo(() => classifyBasin(ode), [ode]);

  const setParam = (k, v) => setParams(p => ({ ...p, [k]: v }));
  const resetDefaults = () => {
    setParams(defaultParams());
    setSigmaMax(SIGMA_MAX_DEFAULT);
  };

  const primaryKeys = ['k_uu', 'k_cu', 'k_hu', 'l', 'lk', 'delta', 'a_e_m', 'a_ai_h'];
  const applyPreset = (preset) => setParams({ ...preset.params });

  return (
    <KaTeXContext.Provider value={katexReady}>
      <div style={{
        minHeight: '100vh', background: C.bg, color: C.fg,
        fontFamily: FONTS.sans,
      }}>
        <AssumptionsPanel open={assumptionsOpen} setOpen={setAssumptionsOpen} />
        <ParamCard pkey={pinnedKey} onClose={() => setPinnedKey(null)} onNavigate={setPinnedKey} />

        <div style={{
          borderBottom: `1px solid ${C.borderStrong}`,
          padding: '16px 22px',
          display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12,
          background: C.bgPanel,
        }}>
          <div>
            <div style={{ fontFamily: FONTS.serif, fontSize: 22, fontWeight: 700, letterSpacing: -0.2, color: C.fg }}>
              cooperativeness basin explorer
            </div>
            <div style={{ fontFamily: FONTS.mono, fontSize: 11, color: C.fgMuted, marginTop: 2, letterSpacing: 0.3 }}>
              progress-clocked basin model · cooperative vs uncooperative reproductive automation
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <Toggle label="time in years" value={displayMode === 'years'} onChange={v => setDisplayMode(v ? 'years' : 'sigma')} />
            <button onClick={() => setAssumptionsOpen(true)} style={btnStyle}>DOCUMENTATION</button>
            <button onClick={resetDefaults} style={btnStyle}>RESET</button>
          </div>
        </div>

        <div style={{ padding: '12px 22px', borderBottom: `1px solid ${C.border}`, background: C.bg }}>
          <BasinBadge basin={basin} p={ode} />
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '320px 1fr', gap: 0 }}>
          <div style={{
            borderRight: `1px solid ${C.border}`,
            background: C.bgPanel,
            maxHeight: 'calc(100vh - 130px)',
            overflowY: 'auto',
          }}>
            <GroupHeader title="Named calibrations" />
            <PresetBar params={params} onApply={applyPreset} />

            <GroupHeader title="Calibration" />
            <Slider pkey="T_auto" value={params.T_auto} onChange={v => setParam('T_auto', v)} onPin={setPinnedKey} />
            <Slider pkey="R0" value={params.R0} onChange={v => setParam('R0', v)} onPin={setPinnedKey} />
            <Slider pkey="Ostar" value={params.Ostar} onChange={v => setParam('Ostar', v)} onPin={setPinnedKey} />

            <GroupHeader title="Primary dynamics" />
            {primaryKeys.map(k => <Slider key={k} pkey={k} value={params[k]} onChange={v => setParam(k, v)} onPin={setPinnedKey} />)}

            <GroupHeader title="Initial conditions" />
            <Slider pkey="q0" value={params.q0} onChange={v => setParam('q0', v)} onPin={setPinnedKey} />
            <Slider pkey="eta0" value={params.eta0} onChange={v => setParam('eta0', v)} onPin={setPinnedKey} />

            <GroupHeader title="View" />
            <HorizonSlider value={sigmaMax} onChange={setSigmaMax} T_auto={params.T_auto} />
          </div>

          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <ViewTabs view={view} setView={setView} />
            <div style={{ padding: 14, flex: 1, overflow: 'auto' }}>
              {view === 'trajectory' && <TrajectoryView params={params} basin={basin} sigmaMax={sigmaMax} displayMode={displayMode} />}
              {view === 'outcome' && <OutcomeMapView params={params} />}
            </div>
          </div>
        </div>
      </div>
    </KaTeXContext.Provider>
  );
}

const btnStyle = {
  background: 'transparent', color: C.fgDim, border: `1px solid ${C.border}`,
  padding: '5px 10px', fontFamily: FONTS.mono, fontSize: 11, cursor: 'pointer',
  letterSpacing: 0.5,
};

function Toggle({ label, value, onChange }) {
  return (
    <button
      onClick={() => onChange(!value)}
      style={{
        background: 'transparent',
        border: `1px solid ${value ? C.accent : C.border}`,
        color: value ? C.accent : C.fgDim,
        padding: '5px 10px',
        fontFamily: FONTS.mono, fontSize: 11, letterSpacing: 0.5,
        cursor: 'pointer',
        display: 'flex', alignItems: 'center', gap: 6,
      }}
    >
      <span style={{
        width: 8, height: 8,
        background: value ? C.accent : 'transparent',
        border: `1px solid ${value ? C.accent : C.fgMuted}`,
      }} />
      {label}
    </button>
  );
}
