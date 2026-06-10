import {
  odeParams, classifyBasinAnalytic,
  simulate, DEFAULTS, fmt,
} from './model.mjs';

// Outcome of the FULL 4D sim from an explicit initial-condition vector,
// holding the long-run ODE coefficients fixed (so we isolate transients).
function outcome(ode, ic, sigmaMax = 60) {
  const sim = simulate(ode, ic, sigmaMax);
  const esc = sim.escaped || sim.final.Q > 0.99;
  return { esc, finalQ: sim.final.Q, finalO: sim.final.O, peakQ: sim.peakQ };
}

// Bisect the critical value of IC component `idx` between lo and hi where the
// outcome flips escape<->contain. Assumes monotone in that component.
function criticalIC(ode, baseIC, idx, lo, hi, sigmaMax = 60) {
  const oc = (v) => { const ic = [...baseIC]; ic[idx] = v; return outcome(ode, ic, sigmaMax).esc; };
  const escLo = oc(lo), escHi = oc(hi);
  if (escLo === escHi) return { flips: false, escLo, escHi };
  for (let j = 0; j < 50; j++) {
    const mid = 0.5 * (lo + hi);
    if (oc(mid) === escLo) lo = mid; else hi = mid;
  }
  return { flips: true, crit: 0.5 * (lo + hi), escLo, escHi };
}

console.log('################ TRANSIENT SENSITIVITY ################');
console.log('All experiments hold long-run ODE coefficients FIXED and vary only');
console.log('the initial state [q0, m0=1, e0=R(0), eta0], so calibration is not');
console.log('confounded with the transient. sigmaMax=60.');

// ---------------------------------------------------------------------------
// 0. Three reference regimes (monostable / bistable / escape).
// ---------------------------------------------------------------------------
const REGIMES = {
  monostable: { k_uu: 0.5, k_cu: 0.05, l: 2 },
  bistable:   { k_uu: 1.2, k_cu: 0.10, l: 4 },
  escape:     { k_uu: 1.2, k_cu: 0.40, l: 1 },
};
console.log('\n## 0. Reference regimes (analytic kind, stable Q*, saddle Q_s) ##');
for (const [name, over] of Object.entries(REGIMES)) {
  const ode = odeParams({ ...DEFAULTS, ...over });
  const ana = classifyBasinAnalytic(ode);
  console.log(`   ${name.padEnd(10)} kind=${ana.kind.padEnd(10)} Q*=${ana.qStable===null?'  -  ':fmt(ana.qStable)}  Q_saddle=${ana.qSaddle===null?'  -  ':fmt(ana.qSaddle)}`);
}

// ---------------------------------------------------------------------------
// 1. Initial A_u (q0): sensitivity is confined to the BISTABLE regime.
//    The separatrix is the saddle. We measure the critical q0 from the full
//    4D sim and compare to the analytic slow-manifold saddle.
// ---------------------------------------------------------------------------
console.log('\n## 1. Initial A_u  (q0)  — sweep, full-4D outcome ##');
for (const [name, over] of Object.entries(REGIMES)) {
  const ode = odeParams({ ...DEFAULTS, ...over });
  const ana = classifyBasinAnalytic(ode);
  const baseIC = [0, 1, Math.max(DEFAULTS.R0, 1e-9), DEFAULTS.eta0];
  let row = `   ${name.padEnd(10)} q0: `;
  for (const q0 of [0.001, 0.01, 0.05, 0.2, 0.4, 0.6, 0.8]) {
    const oc = outcome(ode, [q0, baseIC[1], baseIC[2], baseIC[3]]);
    row += `${q0}:${oc.esc ? 'E' : 'C'}  `;
  }
  // critical q0 from sim vs analytic saddle
  const crit = criticalIC(ode, baseIC, 0, 0.0005, 0.999);
  const critStr = crit.flips ? fmt(crit.crit) : '(no flip)';
  const sadStr = ana.qSaddle === null ? '   -  ' : fmt(ana.qSaddle);
  console.log(row);
  console.log(`              q0* (4D sim) = ${critStr}   vs analytic saddle Q_s = ${sadStr}`);
}

// ---------------------------------------------------------------------------
// 2. Initial observability R(0)=e0 (m0=1) in the bistable regime, with the
//    human-era help switched OFF (eta0=0) and q0 parked just under the
//    eta0=0 separatrix (q0*~0.266), so the m/e transient can tip the outcome.
// ---------------------------------------------------------------------------
console.log('\n## 2. Initial observability  R(0)=e0  (bistable, eta0=0, q0 near separatrix) ##');
{
  const ode = odeParams({ ...DEFAULTS, ...REGIMES.bistable });
  for (const q0 of [0.20, 0.25, 0.26]) {
    let row = `   q0=${fmt(q0)}  e0: `;
    for (const e0 of [0.001, 0.01, 0.1, 1, 10, 100]) {
      const oc = outcome(ode, [q0, 1, e0, 0]);
      row += `${e0}:${oc.esc ? 'E' : 'C'} `;
    }
    const crit = criticalIC(ode, [q0, 1, 0, 0], 2, 1e-4, 1e4);
    row += crit.flips ? `  | e0* @ ${fmt(crit.crit)} (O0=${fmt(1/(1+crit.crit))})` : '  | no flip';
    console.log(row);
  }
  console.log('   does e0 move the q0 separatrix at all? (eta0=0)');
  for (const e0 of [0.001, 0.1, 10, 1000]) {
    const crit = criticalIC(ode, [0, 1, e0, 0], 0, 0.0005, 0.999);
    console.log(`     e0=${String(e0).padStart(5)} (O0=${fmt(1/(1+e0))})  q0* = ${crit.flips ? fmt(crit.crit) : '(no flip)'}`);
  }
}

// ---------------------------------------------------------------------------
// 3. Human-era leakage eta0 and AI monitoring advantage a_ai_h.
//    eta decays as e^{-sigma}: pure transient. With k_hu small most of eta
//    feeds the cooperative pool AND the monitoring source (FM has eta/a_ai_h),
//    so the human era is STABILIZING. a_ai_h sets how much eta helps monitoring.
// ---------------------------------------------------------------------------
console.log('\n## 3. Human-era transient: eta0 and a_ai_h (bistable, q0=0.9·Q_s) ##');
{
  const base = { ...DEFAULTS, ...REGIMES.bistable };
  const ode = odeParams(base);
  const ana = classifyBasinAnalytic(ode);
  const q0 = ana.qSaddle * 0.9;
  console.log(`   fixed q0=${fmt(q0)} (=0.9·Q_s=${fmt(ana.qSaddle)}), e0=R(0) default`);
  let row = '   eta0: ';
  for (const eta0 of [0, 1, 2, 3, 5, 20, 100]) {
    const oc = outcome(ode, [q0, 1, Math.max(base.R0, 1e-9), eta0]);
    row += `${String(eta0).padStart(3)}:${oc.esc ? 'E' : 'C'}  `;
  }
  console.log(row);
  const cEta = criticalIC(ode, [q0, 1, Math.max(base.R0, 1e-9), 0], 3, 0, 500);
  console.log(`        eta0* flip = ${cEta.flips ? fmt(cEta.crit) : '(no flip)'} (eta0 above this contains)`);

  // a_ai_h enters calibration AND the eta-era monitoring source. Hold eta0 at
  // the level where it is marginal (just below the eta0* found above) so a_ai_h
  // can decide the outcome; recompute ode per value (it recalibrates).
  const etaMarg = cEta.flips ? cEta.crit * 0.9 : 1;
  console.log(`   a_ai_h sweep at marginal eta0=${fmt(etaMarg)} (recalibrated each):`);
  let row2 = '        ';
  for (const a_ai_h of [0.2, 0.5, 1, 2, 5, 8]) {
    const odeA = odeParams({ ...base, a_ai_h });
    const oc = outcome(odeA, [q0, 1, Math.max(base.R0, 1e-9), etaMarg]);
    row2 += `${a_ai_h}:${oc.esc ? 'E' : 'C'}  `;
  }
  console.log(row2);
  console.log('   (small a_ai_h => eta/a_ai_h large => human era boosts monitoring => contain)');
}

// ---------------------------------------------------------------------------
// 4. How wide is the transient-tilted basin boundary? For the bistable
//    regime, trace critical q0 as a function of eta0 to show the separatrix
//    is tilted by the human-era transient (vs the flat analytic saddle).
// ---------------------------------------------------------------------------
console.log('\n## 4. Separatrix tilt: critical q0 vs eta0 (bistable) ##');
{
  const ode = odeParams({ ...DEFAULTS, ...REGIMES.bistable });
  const ana = classifyBasinAnalytic(ode);
  console.log(`   analytic (eta->0) saddle Q_s = ${fmt(ana.qSaddle)}`);
  for (const eta0 of [0, 1, 5, 20, 50]) {
    const crit = criticalIC(ode, [0, 1, Math.max(DEFAULTS.R0, 1e-9), eta0], 0, 0.0005, 0.999);
    console.log(`     eta0=${String(eta0).padStart(3)}  q0* = ${crit.flips ? fmt(crit.crit) : '(no flip)'}`);
  }
}

console.log('\n################ END ################');
