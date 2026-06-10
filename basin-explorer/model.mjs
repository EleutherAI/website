// Shared core: exact replication of the σ-clock model from src/BasinExplorer.jsx.
export const BETA = 1;

export function computeO(m, e) {
  const mu = Math.max(m, 1e-12);
  const eps = Math.max(e, 1e-12);
  return mu / (mu + eps);
}

export function makeDeriv(p) {
  return (s) => {
    const [Q, m, e, eta] = s;
    const O = computeO(m, e);
    const Fc = (1 - Q) * (1 - p.k_cu) + eta * (1 - p.k_hu);
    const Fu = p.k_cu * (1 - Q) + p.k_hu * eta + (p.k_uu - p.l * O) * Q;
    const G = Fc + Fu;
    if (G <= 1e-10) return [0, 0, 0, -eta];
    const FM = p.c_M * ((1 - Q) + eta / p.a_ai_h);
    const FE = p.c_0 + p.a_e_m * p.c_M * Q;
    return [Fu / G - Q, FM / G - m, FE / G - e, -eta];
  };
}

export function computeG(s, p) {
  const [Q, m, e, eta] = s;
  const O = computeO(m, e);
  return (1 - Q) + eta + (p.k_uu - p.l * O) * Q;
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

export function simulate(p, ic, sigmaMax = 15, dt = 0.005) {
  const deriv = makeDeriv(p);
  const steps = Math.max(1, Math.ceil(sigmaMax / dt));
  let state = [...ic];
  const traj = [{ sigma: 0, Q: state[0], m: state[1], e: state[2], eta: state[3], O: computeO(state[1], state[2]) }];
  let escaped = false, escapeReason = null, peakQ = state[0];
  for (let i = 1; i <= steps; i++) {
    if (computeG(state, p) <= 1e-6) { escaped = true; escapeReason = 'G<=0'; break; }
    state = rk4Step(state, dt, deriv);
    state[0] = Math.min(Math.max(state[0], 0), 1);
    state[1] = Math.max(state[1], 0);
    state[2] = Math.max(state[2], 0);
    state[3] = Math.max(state[3], 0);
    if (state.some(v => !Number.isFinite(v))) { escaped = true; escapeReason = 'numerical'; break; }
    peakQ = Math.max(peakQ, state[0]);
    if (state[0] > 0.999) escaped = true;
    traj.push({ sigma: i * dt, Q: state[0], m: state[1], e: state[2], eta: state[3], O: computeO(state[1], state[2]) });
    if (escaped) break;
  }
  return { traj, escaped, escapeReason, peakQ, final: traj[traj.length - 1] };
}

// Numeric long-run roots in r = Q/(1-Q), as the app does it.
export function findSteadyStatesR(p) {
  const { k_uu, k_cu, l } = p;
  const f = (r) => {
    if (r < 0) return NaN;
    const Q = r / (1 + r);
    const mSource = p.c_M * Math.max(1 - Q, 1e-12);
    const eSource = p.c_0 + p.a_e_m * p.c_M * Q;
    const ratio = eSource / Math.max(mSource, 1e-12);
    const O = 1 / (1 + Math.pow(Math.max(ratio, 1e-12), BETA));
    return (k_cu + k_uu - 1 - l * O) * r + k_cu;
  };
  const roots = [];
  const rMax = 300, N = 5000;
  let prev = f(0);
  for (let i = 1; i <= N; i++) {
    const r = (i / N) * rMax;
    const cur = f(r);
    if (Number.isFinite(prev) && Number.isFinite(cur) && prev * cur < 0) {
      let lo = ((i - 1) / N) * rMax, hi = (i / N) * rMax;
      for (let j = 0; j < 60; j++) {
        const mid = 0.5 * (lo + hi);
        if (f(lo) * f(mid) <= 0) hi = mid; else lo = mid;
      }
      roots.push(0.5 * (lo + hi));
    }
    prev = cur;
  }
  return roots;
}

export const rToQ = (r) => r / (1 + r);

export function classifyBasinNumeric(p) {
  const qRoots = findSteadyStatesR(p).map(rToQ);
  if (p.k_cu <= 1e-9) {
    const Ostar0 = 1 / (1 + p.c_0 / Math.max(p.c_M, 1e-12));
    if (p.l * Ostar0 >= p.k_uu - 1) qRoots.unshift(0);
  }
  if (qRoots.length === 0) return { kind: 'escape', qRoots, qStable: null, qSaddle: null };
  if (qRoots.length === 1) return { kind: 'monostable', qRoots, qStable: qRoots[0], qSaddle: null };
  return { kind: 'bistable', qRoots, qStable: qRoots[0], qSaddle: qRoots[1] };
}

export function calibratedRates(p) {
  const Q = p.q0, eta = p.eta0;
  const R0 = Math.max(p.R0, 1e-9);
  const O0 = 1 / (1 + R0);
  const m0 = 1;
  const G0 = Math.max((1 - Q) + eta + (p.k_uu - p.l * O0) * Q, 0.05);
  const monitoringSource0 = Math.max((1 - Q) + eta / p.a_ai_h, 1e-9);
  // Main method (blog): peg c_M to the automation production scale so monitoring
  // starts at quasi-steady-state (m0 = 1), then set passive opacity from the
  // best-case long-run observability O* (central 1/2), via c_0/c_M = (1-O*)/O*.
  const Ostar = Math.min(Math.max(p.Ostar, 1e-6), 1 - 1e-6);
  const c_M = G0 * m0 / monitoringSource0;
  const c_0 = ((1 - Ostar) / Ostar) * c_M;
  return { c_M, c_0 };
}

export function odeParams(p) {
  const r = calibratedRates(p);
  return { k_uu: p.k_uu, k_cu: p.k_cu, k_hu: p.k_hu, l: p.l, a_ai_h: p.a_ai_h, a_e_m: p.a_e_m, c_M: r.c_M, c_0: r.c_0 };
}
export function buildIC(p) { return [p.q0, 1, Math.max(p.R0, 1e-9), p.eta0]; }

export const DEFAULTS = {
  T_auto: 0.5, Ostar: 0.5,
  k_uu: 1, k_cu: 0.05, k_hu: 0.05, l: 0.2, a_e_m: 1, a_ai_h: 1,
  R0: 1, q0: 0.05, eta0: 5,
};

export function run(over = {}) {
  const p = { ...DEFAULTS, ...over };
  const ode = odeParams(p);
  return { p, ode, basin: classifyBasinNumeric(ode), sim: simulate(ode, buildIC(p), 15) };
}

export const fmt = (x, d = 3) => (Number.isFinite(x) ? x.toFixed(d) : String(x));

// ----------------------------------------------------------------------------
// ANALYTIC LAYER (slow manifold, η→0, m/e at quasi-steady state)
// ----------------------------------------------------------------------------
// O*(Q) = (1-Q) / (1 + β0 + (a-1)Q),  β0 = c_0/c_M,  a = a_E/M.
export function oStar(Q, beta0, a) {
  return (1 - Q) / (1 + beta0 + (a - 1) * Q);
}

// Fixed points solve  k_cu = Q(1 - k_uu + ℓ·O*(Q)).
// Clearing the denominator gives a quadratic A Q² + B Q + C = 0:
export function basinQuadratic(ode) {
  const { k_uu, k_cu, l } = ode;
  const a = ode.a_e_m;
  const beta0 = ode.c_0 / ode.c_M;
  const A = (1 - k_uu) * (a - 1) - l;
  const B = (1 - k_uu) * (1 + beta0) + l - k_cu * (a - 1);
  const C = -k_cu * (1 + beta0);
  return { A, B, C, a, beta0 };
}

// Returns roots in (0,1), the discriminant, and a classification, all in closed form.
export function classifyBasinAnalytic(ode) {
  const { A, B, C, beta0, a } = basinQuadratic(ode);
  let rootsAll;
  if (Math.abs(A) < 1e-12) {
    rootsAll = Math.abs(B) < 1e-12 ? [] : [-C / B];
  } else {
    const disc = B * B - 4 * A * C;
    if (disc < 0) rootsAll = [];
    else {
      const sq = Math.sqrt(disc);
      rootsAll = [(-B + sq) / (2 * A), (-B - sq) / (2 * A)];
    }
  }
  const roots = rootsAll.filter(q => q > 1e-9 && q < 1 - 1e-9).sort((x, y) => x - y);
  // k_cu = 0: q = 0 is an exact cooperative fixed point (C = 0 ⇒ a root at q=0
  // that the >1e-9 filter drops). Add it when stable: ℓ·O*(0) ≥ k_uu − 1.
  if (ode.k_cu <= 1e-9) {
    const Ostar0 = 1 / (1 + beta0);
    if (ode.l * Ostar0 >= ode.k_uu - 1) roots.unshift(0);
  }
  const disc = B * B - 4 * A * C;
  let kind;
  if (roots.length === 0) kind = 'escape';
  else if (roots.length === 1) kind = 'monostable';
  else kind = 'bistable';
  return { kind, qStable: roots[0] ?? null, qSaddle: roots[1] ?? null, roots, disc, A, B, C, beta0, a };
}
