import {
  odeParams, classifyBasinNumeric, classifyBasinAnalytic,
  oStar, simulate, buildIC, DEFAULTS, fmt,
} from './model.mjs';

console.log('################ ANALYTIC ANALYSIS ################');

// ---------------------------------------------------------------------------
// 1. Verify the closed-form basin quadratic against the app's numeric solver.
// ---------------------------------------------------------------------------
console.log('\n## 1. Analytic quadratic vs numeric root-finder (random params) ##');
function randParams() {
  const r = (lo, hi) => lo + Math.random() * (hi - lo);
  return {
    k_uu: r(0, 2.2), k_cu: r(0, 1), l: r(0, 10),
    a_e_m: r(0.2, 8), a_ai_h: r(0.2, 8),
    T_E_double: r(0.5, 10), T_M_half: r(0.5, 10),
    R0: Math.pow(10, r(-2, 1)), q0: r(0.005, 0.1),
  };
}
let maxErr = 0, mismatches = 0, n = 4000;
for (let i = 0; i < n; i++) {
  const ode = odeParams({ ...DEFAULTS, ...randParams() });
  const num = classifyBasinNumeric(ode);
  const ana = classifyBasinAnalytic(ode);
  if (num.kind !== ana.kind) mismatches++;
  if (num.qStable !== null && ana.qStable !== null) maxErr = Math.max(maxErr, Math.abs(num.qStable - ana.qStable));
}
console.log(`   ${n} random configs:  kind mismatches=${mismatches}   max |Q*_num - Q*_ana|=${maxErr.toExponential(2)}`);

// ---------------------------------------------------------------------------
// 2. The qualitative-regime boundary is exactly k_uu + k_cu = 1.
//    P(0) = C = -k_cu(1+β0) < 0 always;  P(1) = (a+β0)(1 - k_uu - k_cu).
//    So sign(P(1)) = sign(1 - k_uu - k_cu):
//      k_uu+k_cu < 1  -> P(0)<0,P(1)>0 -> exactly 1 interior root -> MONOSTABLE
//      k_uu+k_cu > 1  -> P(0)<0,P(1)<0 -> 0 or 2 roots -> ESCAPE or BISTABLE
// ---------------------------------------------------------------------------
console.log('\n## 2. Regime boundary at k_uu + k_cu = 1 (independent of ℓ, O, monitoring) ##');
console.log('   sweeping (k_uu,k_cu) across the line; ℓ and a varied to show independence:');
for (const l of [0.5, 4, 20]) {
  for (const sum of [0.8, 0.95, 1.0, 1.05, 1.3]) {
    const k_uu = sum * 0.6, k_cu = sum * 0.4; // keep ratio fixed, vary the sum
    const ode = odeParams({ ...DEFAULTS, k_uu, k_cu, l });
    const ana = classifyBasinAnalytic(ode);
    console.log(`     ℓ=${String(l).padStart(2)}  k_uu+k_cu=${fmt(sum, 2)}  ->  ${ana.kind.padEnd(10)} (P(1) sign ${1 - sum > 0 ? '+' : (1 - sum < 0 ? '-' : '0')})`);
  }
}
console.log('   => Crossing 1 flips monostable <-> (bistable/escape) regardless of ℓ.');

// ---------------------------------------------------------------------------
// 3. Within k_uu+k_cu>1: the bistable<->escape boundary is the saddle-node,
//    disc = B² - 4AC = 0. Solve it analytically for critical ℓ given (k_uu,k_cu).
// ---------------------------------------------------------------------------
console.log('\n## 3. Saddle-node (bistable<->escape) boundary: critical ℓ vs k_cu ##');
// For fixed k_uu, a, β0, disc(ℓ)=0 is quadratic in ℓ (A and B are linear in ℓ).
// A = (1-k_uu)(a-1) - ℓ ;  B = (1-k_uu)(1+β0) + ℓ - k_cu(a-1) ;  C const.
// disc = B² - 4AC. Solve for the smallest ℓ>0 giving disc=0 (saddle-node birth).
function criticalL(k_uu, k_cu, a, beta0) {
  const C = -k_cu * (1 + beta0);
  const A0 = (1 - k_uu) * (a - 1);     // A = A0 - ℓ
  const B0 = (1 - k_uu) * (1 + beta0) - k_cu * (a - 1); // B = B0 + ℓ
  // disc(ℓ) = (B0+ℓ)² - 4(A0-ℓ)C = ℓ² + (2B0+4C)ℓ + (B0² - 4A0 C)
  const aa = 1, bb = 2 * B0 + 4 * C, cc = B0 * B0 - 4 * A0 * C;
  const d = bb * bb - 4 * aa * cc;
  if (d < 0) return null;
  const sq = Math.sqrt(d);
  const r1 = (-bb - sq) / 2, r2 = (-bb + sq) / 2;
  return [r1, r2].filter(x => x > 0).sort((x, y) => x - y);
}
const aFix = DEFAULTS.a_e_m, beta0Fix = (() => { const o = odeParams(DEFAULTS); return o.c_0 / o.c_M; })();
console.log(`   (a=${aFix}, β0≈${fmt(beta0Fix, 4)}, k_uu=1.2 so k_uu+k_cu>1 for all k_cu>0)`);
for (const k_cu of [0.0, 0.1, 0.3, 0.6, 0.95]) {
  const k_uu = 1.2;
  const crit = criticalL(k_uu, k_cu, aFix, beta0Fix);
  const critStr = crit && crit.length ? crit.map(x => fmt(x, 2)).join(' or ') : 'none';
  // verify: just below vs just above predicted ℓ*
  let verify = '';
  if (crit && crit.length) {
    const Lc = crit[crit.length - 1];
    const below = classifyBasinAnalytic(odeParams({ ...DEFAULTS, k_uu, k_cu, l: Lc * 0.9 })).kind;
    const above = classifyBasinAnalytic(odeParams({ ...DEFAULTS, k_uu, k_cu, l: Lc * 1.1 })).kind;
    const belowN = classifyBasinNumeric(odeParams({ ...DEFAULTS, k_uu, k_cu, l: Lc * 0.9 })).kind;
    const aboveN = classifyBasinNumeric(odeParams({ ...DEFAULTS, k_uu, k_cu, l: Lc * 1.1 })).kind;
    verify = `  | ℓ=0.9ℓ*:${below}/${belowN}  ℓ=1.1ℓ*:${above}/${aboveN}`;
  }
  console.log(`     k_cu=${fmt(k_cu, 2)}:  ℓ* = ${critStr}${verify}`);
}
console.log('   (format above/below shows analytic/numeric agree across the bifurcation)');

// ---------------------------------------------------------------------------
// 4. Closed-form contained equilibrium and its observability, vs full 4D sim.
//    Q* = k_cu / (1 - k_uu + ℓ·O*(Q*)),  O* = (1-Q*)/(1+β0+(a-1)Q*).
// ---------------------------------------------------------------------------
console.log('\n## 4. Closed-form Q*, O* vs full 4D simulation (σ=40, η decayed out) ##');
const cases = [
  {}, { k_cu: 0.3 }, { k_uu: 1.2 }, { a_e_m: 5, k_cu: 0.2 },
  { k_cu: 0.95, l: 6 }, { a_ai_h: 0.2, k_cu: 0.4, a_e_m: 3 },
];
console.log('   config                         Q*_ana   Q*_sim    O*_ana   O*_sim');
for (const over of cases) {
  const p = { ...DEFAULTS, ...over };
  const ode = odeParams(p);
  const ana = classifyBasinAnalytic(ode);
  const sim = simulate(ode, buildIC(p), 40);
  const Qa = ana.qStable, Oa = Qa !== null ? oStar(Qa, ode.c_0 / ode.c_M, ode.a_e_m) : null;
  const label = (Object.keys(over).length ? Object.entries(over).map(([k, v]) => `${k}=${v}`).join(',') : 'defaults').padEnd(28);
  const escaped = sim.escaped || sim.final.Q > 0.99;
  console.log(`   ${label}  ${Qa === null ? 'escape ' : fmt(Qa).padStart(6)}  ${escaped ? 'escape' : fmt(sim.final.Q).padStart(6)}    ${Oa === null ? '  -   ' : fmt(Oa).padStart(6)}  ${escaped ? '  -   ' : fmt(sim.final.O).padStart(6)}`);
}

// ---------------------------------------------------------------------------
// 5. Analytic phase diagram in (k_cu, ℓ) for k_uu=1.2, checked against sims.
// ---------------------------------------------------------------------------
console.log('\n## 5. Phase diagram (k_cu, ℓ) at k_uu=1.2  [M=mono E=escape B=bistable] ##');
console.log('        k_cu:  0.0  0.1  0.2  0.4  0.6  0.8  0.95');
for (const l of [0.5, 1, 2, 4, 8, 16]) {
  let row = `   ℓ=${String(l).padStart(4)}   `;
  for (const k_cu of [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95]) {
    const k = classifyBasinAnalytic(odeParams({ ...DEFAULTS, k_uu: 1.2, k_cu, l })).kind;
    row += '  ' + (k === 'monostable' ? 'M' : k === 'escape' ? 'E' : 'B') + '  ';
  }
  console.log(row);
}
console.log('   (k_uu+k_cu>1 throughout, so no M expected; B<->E set by ℓ vs ℓ*(k_cu))');
