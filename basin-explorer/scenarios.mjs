// Standalone replication of the σ-clock model from src/BasinExplorer.jsx.
// Used to probe parameter regimes, including ranges beyond the UI sliders.

const BETA = 1;

function computeO(m, e) {
  const mu = Math.max(m, 1e-12);
  const eps = Math.max(e, 1e-12);
  return mu / (mu + eps); // beta = 1
}

function makeDeriv(p) {
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

function computeG(s, p) {
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

function simulate(p, ic, sigmaMax = 15, dt = 0.005) {
  const deriv = makeDeriv(p);
  const steps = Math.max(1, Math.ceil(sigmaMax / dt));
  let state = [...ic];
  const traj = [{ sigma: 0, Q: state[0], m: state[1], e: state[2], eta: state[3], O: computeO(state[1], state[2]) }];
  let escaped = false, escapeReason = null;
  let peakQ = state[0];
  for (let i = 1; i <= steps; i++) {
    const Gnow = computeG(state, p);
    if (Gnow <= 1e-6) { escaped = true; escapeReason = 'G<=0'; break; }
    state = rk4Step(state, dt, deriv);
    state[0] = Math.min(Math.max(state[0], 0), 1);
    state[1] = Math.max(state[1], 0);
    state[2] = Math.max(state[2], 0);
    state[3] = Math.max(state[3], 0);
    if (state.some(v => !Number.isFinite(v))) { escaped = true; escapeReason = 'numerical'; break; }
    peakQ = Math.max(peakQ, state[0]);
    if (state[0] > 0.999) escaped = true;
    const sigma = i * dt;
    traj.push({ sigma, Q: state[0], m: state[1], e: state[2], eta: state[3], O: computeO(state[1], state[2]) });
    if (escaped) break;
  }
  return { traj, escaped, escapeReason, peakQ, final: traj[traj.length - 1] };
}

function findSteadyStatesR(p) {
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

const rToQ = (r) => r / (1 + r);

function classifyBasin(p) {
  const qRoots = findSteadyStatesR(p).map(rToQ);
  const alpha = 1 + p.l - p.k_uu;
  if (qRoots.length === 0) return { kind: 'escape', qRoots, qStable: null, qSaddle: null, alpha };
  if (qRoots.length === 1) return { kind: 'monostable', qRoots, qStable: qRoots[0], qSaddle: null, alpha };
  return { kind: 'bistable', qRoots, qStable: qRoots[0], qSaddle: qRoots[1], alpha };
}

function calibratedRates(p) {
  const Q = p.q0, eta = p.eta0;
  const R0 = Math.max(p.R0, 1e-9);
  const O0 = 1 / (1 + R0);
  const m0 = 1, e0 = R0;
  const G0 = Math.max((1 - Q) + eta + (p.k_uu - p.l * O0) * Q, 0.05);
  const monitoringSource0 = Math.max((1 - Q) + eta / p.a_ai_h, 1e-9);
  const c_M = (G0 * m0 / monitoringSource0) * (p.T_auto / p.T_M_half);
  const c0Raw = G0 * e0 * (p.T_auto / p.T_E_double) - p.a_e_m * c_M * Q;
  return { c_M, c_0: Math.max(0, c0Raw), c0Raw };
}

function odeParams(p) {
  const r = calibratedRates(p);
  return { k_uu: p.k_uu, k_cu: p.k_cu, k_hu: p.k_hu, l: p.l, a_ai_h: p.a_ai_h, a_e_m: p.a_e_m, c_M: r.c_M, c_0: r.c_0 };
}
function buildIC(p) { return [p.q0, 1, Math.max(p.R0, 1e-9), p.eta0]; }

const DEFAULTS = {
  T_auto: 3, T_M_half: 3, T_E_double: 3,
  k_uu: 0.5, k_cu: 0.05, k_hu: 0.1, l: 2, a_e_m: 1, a_ai_h: 1,
  R0: 0.0101, q0: 0.01, eta0: 5,
};

function run(over = {}) {
  const p = { ...DEFAULTS, ...over };
  const ode = odeParams(p);
  const basin = classifyBasin(ode);
  const sim = simulate(ode, buildIC(p), 15);
  return { p, ode, basin, sim };
}

function fmt(x, d = 3) { return Number.isFinite(x) ? x.toFixed(d) : String(x); }

function report(name, over = {}) {
  const { p, ode, basin, sim } = run(over);
  const f = sim.final;
  // Absolute-stock proxies: A = e^σ, A_u ∝ Q·e^σ, caught ∝ O·Q·e^σ.
  const sig = f.sigma;
  const Au = f.Q * Math.exp(sig);
  const Au0 = p.q0 * 1;
  const caught = f.O * Au, caught0 = computeO(1, p.R0) * Au0;
  console.log(`\n=== ${name} ===`);
  const keys = Object.keys(over);
  console.log(`  overrides: ${keys.length ? keys.map(k => `${k}=${over[k]}`).join(', ') : '(defaults)'}`);
  console.log(`  c_M=${fmt(ode.c_M)} c_0=${fmt(ode.c_0)}  alpha(1+l-k_uu)=${fmt(basin.alpha)}`);
  console.log(`  basin: ${basin.kind}  qStable=${basin.qStable === null ? '-' : fmt(basin.qStable)}  qSaddle=${basin.qSaddle === null ? '-' : fmt(basin.qSaddle)}`);
  console.log(`  sim: Q0=${fmt(p.q0)} -> Qf=${fmt(f.Q)} (peak ${fmt(sim.peakQ)})  O0=${fmt(computeO(1, p.R0))} -> Of=${fmt(f.O)}  ${sim.escaped ? 'ESCAPED(' + sim.escapeReason + ')' : 'no escape'}`);
  console.log(`  abs proxy @σ=${fmt(sig,1)}: A_u ${fmt(Au0,4)}->${fmt(Au,2)} (x${fmt(Au / Au0,1)})   caught ${fmt(caught0,4)}->${fmt(caught,3)} (x${fmt(caught / caught0,1)})`);
  return { p, ode, basin, sim };
}

// ============================================================================
console.log('################ SCENARIO INVESTIGATION ################');

report('Baseline (defaults)');

// ---- High leakage: misalignment grows with capability ----
console.log('\n\n########## HIGH LEAKAGE (k_cu>0.9, l<<1) ##########');
report('High coop->uncoop leakage, weak suppression', { k_cu: 0.95, l: 0.1 });
report('High leakage, but strong suppression', { k_cu: 0.95, l: 6 });
report('High leakage, moderate suppression', { k_cu: 0.95, l: 2 });
// Search: any good outcome with k_cu>0.9?
console.log('\n  -- search for cooperative outcomes with k_cu=0.95 --');
let foundGood = 0;
for (const l of [0.5, 1, 2, 4, 8, 16, 32]) {
  for (const a_ai of [0.2, 1, 5]) {
    for (const TM of [0.25, 1, 3]) {
      const { sim } = run({ k_cu: 0.95, l, a_ai_h: a_ai, T_M_half: TM });
      const good = !sim.escaped && sim.final.Q < 0.5;
      if (good) { foundGood++; console.log(`     GOOD: l=${l} a_ai=${a_ai} T_M=${TM} -> Qf=${fmt(sim.final.Q)}`); }
    }
  }
}
console.log(`  cooperative configs found at k_cu=0.95: ${foundGood}`);
// What k_cu can suppression still hold?
console.log('\n  -- max k_cu that stays cooperative, varying l (a_ai=1,T_M=3) --');
for (const l of [1, 2, 4, 8, 16, 32]) {
  let maxGood = 0;
  for (let kcu = 0; kcu <= 1; kcu += 0.01) {
    const { sim } = run({ k_cu: kcu, l });
    if (!sim.escaped && sim.final.Q < 0.5) maxGood = kcu;
  }
  console.log(`     l=${l}: cooperative up to k_cu≈${fmt(maxGood, 2)}`);
}

// ---- Growth advantage to A_u: test O*l > k_uu condition ----
console.log('\n\n########## GROWTH ADVANTAGE TO A_u (test O·l vs k_uu) ##########');
report('High self-productivity k_uu=1.2', { k_uu: 1.2 });
report('High k_uu + high evasion advantage', { k_uu: 1.2, a_e_m: 5 });
report('Strong attacker tilt (k_uu=1.2, a_e_m=8, l=1)', { k_uu: 1.2, a_e_m: 8, l: 1 });
console.log('\n  -- vary k_uu at fixed l=2; report O at small-Q and outcome --');
for (const kuu of [0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.2]) {
  const { ode, basin, sim } = run({ k_uu: kuu });
  // O at the (near-zero Q) cooperative QSS:
  const mS = ode.c_M, eS = ode.c_0; // Q->0
  const Osmall = mS / (mS + eS);
  console.log(`     k_uu=${kuu}: O_lowQ=${fmt(Osmall)}  l*O=${fmt(2 * Osmall)}  ${2 * Osmall > kuu ? '(l*O>k_uu)' : '(l*O<k_uu)'}  basin=${basin.kind}  Qf=${fmt(sim.final.Q)} ${sim.escaped ? 'ESC' : ''}`);
}

// ---- Safety work special challenges ----
console.log('\n\n########## SAFETY WORK IS HARD (high leak, low AI obs productivity) ##########');
report('Low AI monitoring productivity a_ai_h=0.2', { a_ai_h: 0.2 });
report('Hard safety: a_ai_h=0.2, k_cu=0.4, a_e_m=3', { a_ai_h: 0.2, k_cu: 0.4, a_e_m: 3 });
report('Hard safety severe: a_ai_h=0.1, k_cu=0.6, a_e_m=5, l=1', { a_ai_h: 0.1, k_cu: 0.6, a_e_m: 5, l: 1 });

// ---- We observe growing A_u: is it self-stabilizing? ----
console.log('\n\n########## OBSERVING GROWING A_u ##########');
report('Growing-Au regime (k_cu=0.3, a_e_m=3, l=1)', { k_cu: 0.3, a_e_m: 3, l: 1, q0: 0.02 });

// ---- Decoupling: caught SHARE (O*Q) vs true SHARE (Q). e^σ growth is common
//      to both and factored out, so we compare shares of total automation. ----
console.log('\n\n########## DECOUPLING: true share Q vs caught share O*Q ##########');
function showShares(label, sim) {
  console.log(label);
  console.log('   σ      Q(true share)   O(catch rate)   caughtShare=O*Q   missedShare=(1-O)*Q');
  const pick = sim.traj.filter((_, i) => i % Math.ceil(sim.traj.length / 8) === 0).concat([sim.final]);
  for (const t of pick) {
    console.log(`   ${fmt(t.sigma, 1).padStart(4)}   ${fmt(t.Q).padStart(8)}        ${fmt(t.O).padStart(6)}        ${fmt(t.O * t.Q, 4).padStart(8)}          ${fmt((1 - t.O) * t.Q, 4).padStart(8)}`);
  }
}
{
  // Case A: true share Q rising while caught share O*Q ~flat (monitoring collapses as fast as Q grows).
  const { sim } = run({ k_cu: 0.18, a_e_m: 8, l: 0.5, T_E_double: 1.2, q0: 0.015 });
  showShares('\nCase A — TRUE uncoop share rising, CAUGHT share ~flat/declining:', sim);
}
{
  // Case B: true share Q ~flat/falling while caught share O*Q rising (monitoring improving).
  const { sim } = run({ k_cu: 0.02, a_e_m: 0.3, l: 4, T_M_half: 0.5, R0: 5, q0: 0.05 });
  showShares('\nCase B — TRUE uncoop share flat/falling, CAUGHT share rising (better evals):', sim);
}
