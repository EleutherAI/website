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
// MATH — σ-clock ODE
// =============================================================================
//
// Leakage is TRUE transfer between F_c and F_u (not bonus production), so it
// cancels in G. Of the suppressed flow ℓO·Q leaving F_u, a fraction δ is
// destroyed and (1−δ) is redirected into cooperative production, so the only
// term that can reduce G is the destroyed part -δℓO·Q.
//
//   F_c = (1-Q)(1-k_cu) + η(1-k_hu) + (1-δ)ℓO·Q
//   F_u = k_cu(1-Q) + k_hu·η + (k_uu - ℓO)Q
//   G   = F_c + F_u = (1-Q) + η + (k_uu - δℓO)Q
//
//   F_M = c_M[(1-Q) + η/a_AI/H]     (monitoring production)
//   F_E = c_0 + a_E/M·c_M·Q         (passive opacity + active evasion)
//
//   Q' = F_u/G - Q
//   η' = -η
//   m' = F_M/G - m
//   e' = F_E/G - e
//   O  = m^β / (m^β + e^β)
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
    const Fc = (1 - Q) * (1 - p.k_cu) + eta * (1 - p.k_hu) + (1 - p.delta) * p.l * O * Q;
    const Fu = p.k_cu * (1 - Q) + p.k_hu * eta + (p.k_uu - p.l * O) * Q;
    const G = Fc + Fu;  // = (1-Q) + η + (k_uu - δℓO)Q
    if (G <= 1e-10) {
      // Out of validity envelope (progress clock broken)
      return [0, 0, 0, -eta];
    }
    const FM = p.c_M * ((1 - Q) + eta / p.a_ai_h);
    const FE = p.c_0 + p.a_e_m * p.c_M * Q;
    return [
      Fu / G - Q,
      FM / G - m,
      FE / G - e,
      -eta,
    ];
  };
}

function computeG(state, p) {
  const [Q, m, e, eta] = state;
  const O = computeO(m, e, BETA);
  return (1 - Q) + eta + (p.k_uu - p.delta * p.l * O) * Q;
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

// Long-run projection (η → 0, m/e at quasi-steady-state).
// Q' = 0 reduces to f(r) = 0 in r = Q/(1-Q) (δ-general, derivation audit R3):
//   f(r) = k_cu + (k_cu + k_uu - 1 - ℓ·O*(r))·r - (1-δ)·ℓ·O*(r)·r²
// where O*(r) is computed from the long-run monitoring/evasion QSS ratio
// (δ-independent, R2). At δ = 1 the r² term vanishes and this is exactly the
// previous fixed-point equation.
function findSteadyStatesR(p) {
  const { k_uu, k_cu, l } = p;
  const f = (r) => {
    if (r < 0) return NaN;
    const Q = r / (1 + r);
    const mSource = p.c_M * Math.max(1 - Q, 1e-12);
    const eSource = p.c_0 + p.a_e_m * p.c_M * Q;
    const ratio = eSource / Math.max(mSource, 1e-12);
    const O = 1 / (1 + Math.pow(Math.max(ratio, 1e-12), BETA));
    return (k_cu + k_uu - 1 - l * O) * r + k_cu - (1 - p.delta) * l * O * r * r;
  };
  const roots = [];
  const rMax = 300;
  const N = 5000;
  let prev = f(0);
  for (let i = 1; i <= N; i++) {
    const r = (i / N) * rMax;
    const cur = f(r);
    if (Number.isFinite(prev) && Number.isFinite(cur) && prev * cur < 0) {
      let lo = ((i - 1) / N) * rMax;
      let hi = (i / N) * rMax;
      for (let j = 0; j < 60; j++) {
        const mid = 0.5 * (lo + hi);
        if (f(lo) * f(mid) <= 0) hi = mid;
        else lo = mid;
      }
      roots.push(0.5 * (lo + hi));
    }
    prev = cur;
  }
  return roots;
}

function rToQ(r) { return r / (1 + r); }

// =============================================================================
// ANALYTIC BASIN-EXISTENCE BOUNDARY (δ-general)
// =============================================================================
// Long-run fixed points solve A_δ·r² + B·r + C = 0 in r = q_u/(1−q_u), with
//   a = a_E/M,  c = c_0/c_M,  b = k_cu + k_uu − 1,
//   A_δ = b(a+c) − (1−δ)ℓ,  B = k_cu(a+c) + b(1+c) − ℓ,  C = k_cu(1+c).
// Existence of an interior fixed point is monotone in ℓ, with a unique
// threshold (derivation audit R4):
//   ℓ* = min( ℓ_A , max(P, ℓ₊) ), where
//     ℓ_A = b(a+c)/(1−δ)            (A_δ = 0; ∞ at δ = 1)
//     P   = k_cu(a+c) + b(1+c)      (B = 0)
//     ℓ₊  = M̃ + √(M̃² − N²),  M̃ = P − 2(1−δ)k_cu(1+c),  N = |k_cu(a+c) − b(1+c)|
//   and ℓ* = 0 when b < 0, or when b = 0 and δ < 1.
// At δ = 1 this reduces to ℓ* = (√(k_cu(a+c)) + √(b(1+c)))² for b > 0, the
// previous boundary; at k_uu = 1, a = 1: ℓ*O* = 4δk_cu (δ ≥ ½) or
// k_cu/(1−δ) (δ ≤ ½).
const BASIN_BOUNDARY = {
  id: 'deltaGeneral',
  label: 'analytic basin boundary (δ-general ℓ*)',
  // Critical suppression ℓ* above which a long-run interior fixed point exists.
  lstar(odeP) {
    const a = odeP.a_e_m;
    const c = odeP.c_0 / Math.max(odeP.c_M, 1e-12);
    const b = odeP.k_cu + odeP.k_uu - 1;
    const delta = odeP.delta;
    if (b < 0 || (Math.abs(b) < 1e-15 && delta < 1)) return 0;
    const lA = delta < 1 ? b * (a + c) / (1 - delta) : Infinity;
    const P = odeP.k_cu * (a + c) + b * (1 + c);
    const Mt = P - 2 * (1 - delta) * odeP.k_cu * (1 + c);
    const N = Math.abs(odeP.k_cu * (a + c) - b * (1 + c));
    const disc = Mt * Mt - N * N;
    const lPlus = disc >= 0 ? Mt + Math.sqrt(disc) : -Infinity;
    return Math.min(lA, Math.max(P, lPlus));
  },
  // Returns > 0 when a cooperative-side basin exists; the zero level set is the
  // boundary curve drawn on the outcome map.
  margin(odeP) {
    return odeP.l - this.lstar(odeP);
  },
};

function classifyBasin(p) {
  const rRoots = findSteadyStatesR(p);
  const qRoots = rRoots.map(rToQ);
  // When k_cu = 0 there is no leakage seeding, so q = 0 is an exact cooperative
  // fixed point (g(0) = k_cu = 0). The interior root-finder skips this boundary
  // root, so add it explicitly when it is stable: ℓ·O*(0) ≥ k_uu − 1, with
  // O*(0) = 1/(1 + c_0/c_M). Without this, k_cu = 0 is misreported as "escape"
  // even though trajectories converge to the cooperative state. This stability
  // condition is δ-independent (derivation audit R9: δ enters F only at O(Q),
  // so it contributes at O(Q²) to the linearisation).
  const Ostar0 = 1 / (1 + p.c_0 / Math.max(p.c_M, 1e-12));
  if (p.k_cu <= 1e-9) {
    if (p.l * Ostar0 >= p.k_uu - 1) qRoots.unshift(0);
  }
  // Badge readout: the k_cu → 0 reduction of Condition 1 is ℓ·O*(0) ≥ k_uu − 1.
  // (An earlier readout α = 1 + ℓ − k_uu omitted the O*(0) factor on ℓ.)
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
    est: 'Central ≈ 50% (blog sweeps O(0) ∈ 0.3–0.9; prior c₀ = c_M ⇒ O* = ½).',
    detail: 'Folds a messy economic judgement — the risk of under-investing in observability versus the cost of pushing it higher — into one long-run level. Splitting it into pieces would not make it easier to estimate, so the model asks for the result directly. "Best case" because it holds while uncoop is rare; true long-run observability falls as q_u grows.',
    math: [String.raw`O^*(0) = \frac{1}{1 + c_0/c_M}`, String.raw`\frac{c_0}{c_M} = \frac{1-O^*}{O^*}`],
    role: 'Sets passive opacity c₀ relative to monitoring c_M. c_M itself is pegged to the automation scale, controlling only how fast O relaxes from O(0).',
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
    role: 'k_uu + k_cu > 1 is Condition 2 — the all-uncoop endpoint becomes stable.',
    related: ['k_cu', 'l'],
    default: 1, min: 0, max: 2, step: 0.01,
    tier: 'primary',
  },
  k_cu: {
    label: 'Coop → uncoop leakage',
    symbol: 'k_cu',
    handle: 'Fraction of cooperative AI labour that ends up enabling uncoop instead.',
    est: 'Weakly anchored; central ~0.05 (q_{u,0} proxy), "high leakage" > 0.9.',
    detail: 'A true transfer: this much of q_c moves into q_u (and is removed from q_c). Seeds uncoop steadily even under perfect monitoring, so it sets a floor on q_u.',
    math: [String.raw`F_c \ni q_c(1-k_{cu})`, String.raw`F_u \ni k_{cu}\,q_c`],
    role: 'Enters Condition 1 (ℓ·O*(0) > k_uu+k_cu−1) and Condition 2.',
    related: ['k_hu', 'k_uu', 'l'],
    default: 0.05, min: 0, max: 1, step: 0.005,
    tier: 'primary',
  },
  l: {
    label: 'Suppression strength',
    symbol: 'ℓ',
    handle: 'When detected (O ≈ 1), how strongly is uncoop AI suppressed per σ-unit?',
    est: 'Central ℓ ≈ 0.4 (Petri trend identifies ℓ·O, ÷ O(0) ≈ 0.5); working range 0.2–1.0.',
    detail: 'Acts only on the observed fraction, entering q_u\'s production as −ℓ·O. So it is efficacy against already-known misbehaviour. Identification caveat: a measured decay of misbehaviour identifies the product ℓ·O, not ℓ alone, so the Petri-trend rate is divided by present observability O(0) — and the trend may partly reflect a falling inflow (improving k_hu) rather than suppression, which biases this estimate up.',
    math: [String.raw`F_u \ni (k_{uu} - \ell O)\,q_u`, String.raw`\ell\,O(0) \approx T_{auto}/T_{half}`],
    role: 'Condition 1 (ℓ·O*(0) > k_uu + k_cu − 1) decides whether rare uncoop labour shrinks; only necessary for the basin when δ = 1.',
    related: ['k_uu', 'k_cu', 'Ostar', 'delta'],
    default: 0.4, min: 0, max: 8, step: 0.05,
    tier: 'primary',
  },
  delta: {
    label: 'Suppression destruction fraction',
    symbol: 'δ',
    handle: 'Of the uncoop labour suppression catches, how much is destroyed vs redirected to cooperative work?',
    est: 'Central ≈ 0.7 via the filtering-fraction heuristic (fresh estimate); range 0.3–1.0.',
    detail: 'δ = 1: caught uncoop labour is destroyed outright (filtered, blocked, discarded) — the model\'s old implicit assumption. δ = 0: every caught unit is corrected and its labour redirected into cooperative production at full value. Heuristic: weight suppression channels by destructiveness (output filtering ≈ 1, retraining ≈ 0, control-style resampling/editing ≈ ½) and average by suppressed-flow volume; heavy real-time filtering on frontier deployments puts the central value high. Redirection is assumed 1:1, with no lag or quality discount.',
    math: [
      String.raw`F_u \ni -\ell O\,q_u`,
      String.raw`F_c \ni +(1-\delta)\,\ell O\,q_u`,
      String.raw`G = (1-q_u) + q_h + (k_{uu} - \delta\ell O)\,q_u`,
    ],
    role: 'Lowers the basin threshold ℓ* (at k_uu = 1, a_E/M = 1: ℓ*O* = 4δk_cu for δ ≥ ½, k_cu/(1−δ) for δ ≤ ½) and destabilises the all-uncoop endpoint when k_uu + k_cu − 1 < (1−δ)ℓ/(a+c). Caution: for δ < 1 the surviving attractor can sit at high q_u — check its location, not just existence.',
    related: ['l', 'k_cu', 'k_uu'],
    default: 0.7, min: 0, max: 1, step: 0.05,
    tier: 'primary',
  },
  a_e_m: {
    label: 'Evasion advantage',
    symbol: 'a_E/M',
    handle: 'Per unit of AI labour, how productive is evasion vs monitoring?',
    est: 'Order 1 (M-Trends dwell-time); growth-advantage scenario pushes > 1.',
    detail: '> 1 means uncoop AI produces evasion faster than cooperative AI produces monitoring, at equal labour.',
    math: [String.raw`F_E = c_0 + a_{E/M}\,c_M\,q_u`],
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
    est: 'Proxied by q_{u,0}: central ~5–15%, up to ~67%.',
    detail: 'Like k_cu but for human labour. Active only while q_h > 0; fades as humans become economically irrelevant.',
    math: [String.raw`F_u \ni k_{hu}\,q_h`],
    role: 'Reproduction-relevant leakage (property ii′); plausibly rarer than observed misbehaviour.',
    related: ['k_cu', 'eta0'],
    default: 0.05, min: 0, max: 1, step: 0.005,
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
// observed-misbehaviour evidence is read ({k_cu, k_hu, q0} move together — the
// same evidence does double duty as the q_u(0) and leakage-rate estimate).
// AI 2027 is Broad with the leakage k_cu pushed to the high-leakage reading of
// the AI 2027 scenario (prior-driven misalignment at handoff).
// Values must stay in sync with _scratch/review/drafts/named-calibrations.md.
const PRESETS = {
  broad: {
    label: 'Broad',
    blurb: 'behavioural misbehaviour rates at face value',
    params: {
      T_auto: 0.5, Ostar: 0.5, k_uu: 1, k_cu: 0.05, l: 0.4, delta: 0.7,
      a_e_m: 1, a_ai_h: 1, k_hu: 0.05, q0: 0.05, R0: 1, eta0: 5,
    },
  },
  strict: {
    label: 'Strict',
    blurb: 'reproduction-relevant (ii′) rates only, ~10× lower',
    params: {
      T_auto: 0.5, Ostar: 0.5, k_uu: 1, k_cu: 0.005, l: 0.4, delta: 0.7,
      a_e_m: 1, a_ai_h: 1, k_hu: 0.005, q0: 0.005, R0: 1, eta0: 5,
    },
  },
  ai2027: {
    label: 'AI 2027',
    blurb: 'high leakage k_cu = 0.9 at handoff',
    params: {
      T_auto: 0.5, Ostar: 0.5, k_uu: 1, k_cu: 0.9, l: 0.4, delta: 0.7,
      a_e_m: 1, a_ai_h: 1, k_hu: 0.05, q0: 0.05, R0: 1, eta0: 5,
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
  const G0 = Math.max((1 - Q) + eta + (p.k_uu - p.delta * p.l * O0) * Q, 0.05);
  const monitoringSource0 = Math.max((1 - Q) + eta / p.a_ai_h, 1e-9);

  // Main calibration method (blog): fix present observability O(0) via R(0)
  // (which sets m0=1, e0=R0), peg the monitoring scale c_M to the automation
  // production scale so monitoring starts at quasi-steady-state, then set passive
  // opacity from the best-case long-run observability O* (central 1/2) via
  // c_0/c_M = (1-O*)/O*. c_M's magnitude affects only how fast O relaxes from
  // O(0) toward O*. (T_M,½ / T_E,2 rate judgements are the blog's unused alternative.)
  const Ostar = Math.min(Math.max(p.Ostar, 1e-6), 1 - 1e-6);
  const c_M = G0 * m0 / monitoringSource0;
  const c_0 = ((1 - Ostar) / Ostar) * c_M;
  return { c_M, c_0, c0Raw: c_0 };
}

function odeParams(p) {
  const rates = calibratedRates(p);
  return {
    k_uu: p.k_uu, k_cu: p.k_cu, k_hu: p.k_hu,
    l: p.l, delta: p.delta, beta: BETA,
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
// (e.g. ?k_uu=1.2&l=0.4), plus sigmaMax / view / time for the view state.
// Only values that differ from defaults are written; values read from the URL
// are clamped to each parameter's [min, max].

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
  const url = window.location.pathname + (qs ? '?' + qs : '') + window.location.hash;
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
        Named calibrations set every slider. Broad/Strict differ only in how much
        observed misbehaviour counts as reproduction-relevant ({'{'}k_cu, k_hu, q_u(0){'}'});
        AI 2027 is Broad with high leakage (k_cu = 0.9).
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
          <span><span style={{ color: C.fgMuted }}>escapes if q_u &gt; </span><span style={{ color: C.bistable, fontWeight: 600 }}>{qSaddle.toFixed(3)}</span></span>
        )}
        <span style={{ color: C.fgMuted, fontSize: 10.5 }} title={`k_cu → 0 reduction of Condition 1: cooperative basin needs ℓ·O*(0) ≥ k_uu − 1. Here ℓ·O*(0) = ${lOstar0.toFixed(2)}, k_uu − 1 = ${kuuExcess.toFixed(2)}; c_M = ${p.c_M?.toFixed?.(3) ?? 'n/a'}; c_0 = ${p.c_0?.toFixed?.(3) ?? 'n/a'}`}>
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

const SWEEPABLE_KEYS = ['Ostar', 'k_uu', 'k_cu', 'k_hu', 'l', 'delta', 'a_e_m', 'a_ai_h'];

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

  // Analytic basin-existence boundary: zero level set of BASIN_BOUNDARY.margin
  // over the sweep plane (each sample recalibrated like the grid cells).
  // Traced by scanning columns and rows for sign changes, then bisecting.
  const boundaryPts = useMemo(() => {
    const margAt = (x, y) =>
      BASIN_BOUNDARY.margin(odeParams({ ...params, [xKey]: x, [yKey]: y }));
    const pts = [];
    const trace = (FIXED, SCAN, fixedDef, scanDef, fixedIsX) => {
      for (let j = 0; j <= FIXED; j++) {
        const u = fixedDef.min + (j / FIXED) * (fixedDef.max - fixedDef.min);
        let vPrev = scanDef.min;
        let mPrev = fixedIsX ? margAt(u, vPrev) : margAt(vPrev, u);
        for (let i = 1; i <= SCAN; i++) {
          const v = scanDef.min + (i / SCAN) * (scanDef.max - scanDef.min);
          const m = fixedIsX ? margAt(u, v) : margAt(v, u);
          if (Number.isFinite(mPrev) && Number.isFinite(m) && mPrev * m < 0) {
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
  }, [params, xKey, yKey, xDef, yDef]);

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
            interior fixed point appears/disappears (ℓ = ℓ* from the δ-general
            fixed-point quadratic; at k_uu = 1, a_E/M = 1: ℓ*O* = 4δk_cu for
            δ ≥ ½, k_cu/(1−δ) for δ ≤ ½). For δ &lt; 1 the attractor inside
            the boundary can sit at high q_u — check the badge, not just the curve.
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
              <MV>Q</MV> is the uncooperative share. <MV>η</MV> is human production as a multiple of automated
              production (humans are assumed constant <MV>H_0</MV> in absolute terms). <MV>m</MV> and <MV>e</MV>
              are monitoring and evasion stocks; individually they have no intuitive interpretation, but together
              they determine observability:
            </P>
            <TeX display>{String.raw`O = \frac{m}{m + e}`}</TeX>
            <P>
              <MV>O</MV> is the fraction of uncoop activity that monitoring catches in real time — a soft split of
              effort between monitoring and evasion. (A sharpness exponent <MV>β</MV> on <MV>m</MV> and <MV>e</MV>
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
            <P>Production allocation:</P>
            <TeX display>{String.raw`F_c = (1-Q)(1 - k_{cu}) + \eta(1 - k_{hu}) + (1-\delta)\,\ell O\,Q`}</TeX>
            <TeX display>{String.raw`F_u = k_{cu}(1-Q) + k_{hu}\,\eta + (k_{uu} - \ell O)\,Q`}</TeX>
            <P>
              Cooperative AI takes its share <MV>(1-Q)</MV> of productive capacity less leakage <MV>k_cu</MV>;
              humans contribute their share <MV>η</MV> less leakage <MV>k_hu</MV>. We choose units of automation
              production so that one unit equals one unit of human labour — i.e. <MV>H_0 = 1</MV>.
            </P>
            <P>
              Uncoop production gets leakage from cooperative and human pools, plus its own self-production
              <MV>k_uu·Q</MV>, minus observability-gated suppression <MV>ℓO·Q</MV>.
            </P>
            <P>
              Suppression is split by the destruction fraction <MV>δ</MV>: of the suppressed flow
              <MV> ℓO·Q</MV> leaving <MV>F_u</MV>, fraction <MV>δ</MV> is destroyed outright (filtered,
              blocked, discarded) and fraction <MV>(1−δ)</MV> is redirected into cooperative production
              (retrained, corrected, resampled) — the <MV>+(1−δ)ℓO·Q</MV> term in <MV>F_c</MV>. Redirected
              labour converts 1:1 with no lag or quality discount; if you think filtered-then-retrained
              labour is worth less than one unit, that would be an extra parameter, not a reinterpretation
              of <MV>δ</MV>.
            </P>
            <P>
              Notice that leakage cancels in the sum <MV>G = F_c + F_u</MV> — it is a transfer, not a source —
              and so does the redirected part of suppression. The only term that can pull <MV>G</MV> down is
              the destroyed fraction <MV>-δℓO·Q</MV>. The total growth rate is:
            </P>
            <TeX display>{String.raw`G = (1-Q) + \eta + (k_{uu} - \delta\ell O)\,Q`}</TeX>
            <P>Monitoring and evasion dynamics are analogous:</P>
            <TeX display>{String.raw`F_M = c_M\left((1-Q)+\frac{\eta}{a_{\mathrm{AI/H}}}\right), \qquad F_E = c_0 + a_{\mathrm{E/M}}c_M\,Q`}</TeX>
            <P>
              Monitoring is produced by cooperative AI and humans. The arbitrary units of <MV>m</MV> and
              <MV>e</MV> are fixed by setting cooperative AI as the baseline producer of monitoring:
              <MV>c_M</MV> is the calibrated monitoring-production rate, and
              <MV>a_AI/H</MV> says how much better AI is than humans at monitoring work. Evasion has a
              passive source <MV>c_0</MV> from ordinary automation opacity, plus active evasion from uncoop
              AI scaled by <MV>a_E/M</MV>.
            </P>
            <P>
              The <MV>-m</MV> and <MV>-e</MV> terms in the state equations are not an additional obsolescence
              parameter. They just come from measuring monitoring and evasion stocks relative to total
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
              <strong>Only F_u faces observability-gated suppression.</strong> There is no analogous
              <MV> -ℓO</MV> term in <MV>F_c</MV>. Similar reasoning: cooperative AI operates in the open
              while uncoop operates covertly, so observability is the bottleneck on how much of the
              uncoop slice we can suppress.
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
              <li><MV>O*</MV> — best-case long-run observability, set directly. Observability drifts from its present value <MV>O(0)</MV> toward this; internally it pins the passive-opacity ratio <MV>c_0/c_M = (1−O*)/O*</MV>, with central <MV>O* = 1/2</MV> (so <MV>c_0 = c_M</MV>). "Best case" because it holds while uncoop stays low — the actual long-run <MV>O*(q_u)</MV> falls as <MV>q_u</MV> grows.</li>
              <li><MV>T_auto</MV> — present calendar doubling time of <MV>A</MV>. Under the main method this only scales the calendar-years axis; it no longer enters <MV>c_M</MV>.</li>
            </ul>
            <P>The calibrated rates the simulator uses are then:</P>
            <TeX display>{String.raw`c_M = \frac{G_0\,m_0}{(1-Q_0)+\eta_0/a_{\mathrm{AI/H}}}, \qquad c_0 = \frac{c_0}{c_M}\cdot c_M`}</TeX>
            <P>
              The first pegs <MV>c_M</MV> to the automation production scale, so monitoring starts at
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
              After the transient (<MV>η → 0</MV>, monitoring and evasion at quasi-steady-state), long-run
              observability at a fixed uncoop share <MV>q_u</MV> is
            </P>
            <TeX display>{String.raw`O^*(q_u) = \frac{q_c}{q_c + a_{\mathrm{E/M}}\,q_u + c_0/c_M}`}</TeX>
            <P>
              It falls when the uncoop share grows, when active evasion is more productive, or when passive
              opacity <MV>c_0</MV> is large relative to monitoring productivity <MV>c_M</MV>.
            </P>
            <P>
              Two thresholds organise the qualitative outcome. <strong>Condition 1</strong> — can rare uncoop
              labour be suppressed while observability is still high? Uncoop labour shrinks when rare iff
              observability-gated suppression beats self-reproduction plus leakage:
            </P>
            <TeX display>{String.raw`\ell\,O^*(0) > k_{uu} + k_{cu} - 1, \qquad O^*(0) = \frac{1}{1 + c_0/c_M}`}</TeX>
            <P>
              At <MV>δ = 1</MV> this is necessary (though not sufficient) for a cooperative basin. For
              <MV> δ &lt; 1</MV> it loses necessity: redirected suppression can sustain an interior attractor
              even where rare uncoop labour initially grows — though that attractor may sit at high <MV>q_u</MV>.
              <strong> Condition 2</strong> — can uncoop labour hold the whole system? Redirection makes this
              strictly harder, because what little is still caught keeps re-seeding the cooperative pool. The
              all-uncoop endpoint is stable when
            </P>
            <TeX display>{String.raw`k_{uu} + k_{cu} - 1 > \frac{(1-\delta)\,\ell}{a_{\mathrm{E/M}} + c_0/c_M}`}</TeX>
            <P>
              (at <MV>δ = 1</MV> this reduces to <MV>k_uu + k_cu &gt; 1</MV>). The combinations:
            </P>
            <ul style={{ margin: '0 0 10px 18px', padding: 0 }}>
              <li><strong>Interior attractor exists and C2 ✓</strong> → <em>bistable</em>: a cooperative-side basin coexists with an uncooperative-dominant one, and initial conditions decide where you land.</li>
              <li><strong>No interior attractor and C2 ✓</strong> → the cooperative basin disappears, leaving uncoop dominance (escape).</li>
              <li><strong>C2 ✗</strong> → the all-uncoop endpoint is unstable and a single interior attractor is the only long-run outcome (monostable) — but for <MV>δ &lt; 1</MV> it can sit anywhere from a few percent to near-total uncoop share, so check its location.</li>
            </ul>
            <P>
              The exact interior fixed points solve a quadratic in the odds <MV>r = q_u/(1−q_u)</MV>, with
              <MV> A_δ = b(a+c) − (1−δ)ℓ</MV>, <MV>B = k_cu(a+c) + b(1+c) − ℓ</MV>, <MV>C = k_cu(1+c)</MV>, where
              <MV> b = k_uu + k_cu − 1</MV>, <MV>a = a_E/M</MV> and <MV>c = c_0/c_M</MV>. Condition 2 above is
              exactly <MV>A_δ &gt; 0</MV>. The positive roots are
              the stable attractor (lower) and the basin-separating saddle (higher) — which is what the badge
              and the dashed lines in the IC / trajectory views report. Existence is monotone in <MV>ℓ</MV>,
              with threshold <MV>ℓ*</MV> drawn on the outcome map; at <MV>k_uu = 1</MV>, <MV>a_E/M = 1</MV> it
              takes the piecewise form <MV>ℓ*O* = 4δk_cu</MV> for <MV>δ ≥ ½</MV> and
              <MV> k_cu/(1−δ)</MV> for <MV>δ ≤ ½</MV>. (The badge compares
              <MV> ℓ·O*(0)</MV> against <MV>k_uu − 1</MV> — the <MV>k_cu → 0</MV> reduction of Condition 1,
              which is δ-independent.)
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

  const primaryKeys = ['k_uu', 'k_cu', 'k_hu', 'l', 'delta', 'a_e_m', 'a_ai_h'];
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
              σ-clock alignment basin explorer
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
