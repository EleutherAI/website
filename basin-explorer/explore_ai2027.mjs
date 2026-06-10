// Grounding the AI-2027 appendix figures against the canonical model core.
import {
  DEFAULTS, odeParams, classifyBasinAnalytic, simulate, buildIC, oStar, fmt,
} from './model.mjs';

const P = (over = {}) => ({ ...DEFAULTS, ...over });

console.log('DEFAULTS =', JSON.stringify(DEFAULTS));
console.log('\n## A. Basin at central defaults and rising k_cu (k_uu=1, l=0.2, Ostar=0.5) ##');
for (const k_cu of [0, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.95]) {
  const ode = odeParams(P({ k_cu }));
  const b = classifyBasinAnalytic(ode);
  console.log(`  k_cu=${fmt(k_cu,3).padStart(5)}  ->  ${b.kind.padEnd(11)} qStable=${b.qStable===null?'  -  ':fmt(b.qStable)} qSaddle=${b.qSaddle===null?'  -  ':fmt(b.qSaddle)}`);
}

console.log('\n## B. AI-2027 "high-leakage" trajectories: sigma to reach Q=0.5 and Q=0.9 ##');
for (const k_cu of [0.1, 0.2, 0.3, 0.5, 0.95]) {
  const p = P({ k_cu });
  const sim = simulate(odeParams(p), buildIC(p), 20);
  const hit = (thr) => { const t = sim.traj.find(s => s.Q >= thr); return t ? t.sigma : null; };
  const s50 = hit(0.5), s90 = hit(0.9);
  const f = sim.final;
  console.log(`  k_cu=${fmt(k_cu,3).padStart(5)}  sigma@Q=0.5: ${s50===null?'  -  ':fmt(s50,2).padStart(5)}  sigma@Q=0.9: ${s90===null?'  -  ':fmt(s90,2).padStart(5)}  final Q=${fmt(f.Q)} O=${fmt(f.O)} ${sim.escaped?'ESC':''}`);
}

console.log('\n## C. LEAD CLAIM: does better observability (Ostar) rescue high leakage? ##');
console.log('   max k_cu that still admits a cooperative basin (not escape), per (l, Ostar):');
const maxCoopKcu = (l, Ostar) => {
  let best = 0;
  for (let k = 0; k <= 1.0001; k += 0.005) {
    const b = classifyBasinAnalytic(odeParams(P({ k_cu: k, l, Ostar })));
    if (b.kind !== 'escape') best = k;
  }
  return best;
};
const Ostars = [0.5, 0.7, 0.9, 0.99];
process.stdout.write('     l \\ Ostar ');
for (const o of Ostars) process.stdout.write(`  ${fmt(o,2).padStart(5)}`);
console.log();
for (const l of [0.2, 0.5, 1, 2]) {
  let row = `     l=${fmt(l,2).padStart(4)}    `;
  for (const o of Ostars) row += `  ${fmt(maxCoopKcu(l, o),3).padStart(5)}`;
  console.log(row);
}
console.log('   (analytic boundary for k_uu=1,a=1: escape unless k_cu < l*Ostar/4)');

console.log('\n## D. Even with PERFECT observability (Ostar=0.999), how big must l be vs k_cu? ##');
for (const k_cu of [0.05, 0.1, 0.2, 0.3, 0.5, 0.95]) {
  // smallest l (Ostar=0.999) that escapes -> not escape
  let lstar = null;
  for (let l = 0; l <= 20.0001; l += 0.05) {
    const b = classifyBasinAnalytic(odeParams(P({ k_cu, l, Ostar: 0.999 })));
    if (b.kind !== 'escape') { lstar = l; break; }
  }
  console.log(`  k_cu=${fmt(k_cu,3).padStart(5)}  needs l >= ${lstar===null?'>20':fmt(lstar,2)}  (predicted 4*k_cu=${fmt(4*k_cu,2)})`);
}

console.log('\n## E. Phase grid (k_cu x Ostar) at central l=0.2  [M/B/E] ##');
const Ogrid = [0.5, 0.6, 0.7, 0.8, 0.9, 0.99];
process.stdout.write('     Ostar\\k_cu');
for (const k of [0.0,0.025,0.05,0.1,0.2,0.4,0.95]) process.stdout.write(`${fmt(k,3).padStart(6)}`);
console.log();
for (const o of Ogrid) {
  let row = `     ${fmt(o,2).padStart(5)}     `;
  for (const k of [0.0,0.025,0.05,0.1,0.2,0.4,0.95]) {
    const b = classifyBasinAnalytic(odeParams(P({ k_cu: k, l: 0.2, Ostar: o })));
    row += `     ${b.kind==='monostable'?'M':b.kind==='escape'?'E':'B'}`;
  }
  console.log(row);
}
