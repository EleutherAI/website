# basin-explorer

Interactive explorer for the dynamical model in
`content-blog/dynamical-models-of-ai-governability.md`. Built with Vite +
React; the deployed static build lives at `static/basin-explorer/` and is
served at `/basin-explorer/`.

```
npm install
npm run dev      # local dev server
npm run build    # writes dist/; copy to static/basin-explorer/ to deploy
```

**The canonical math engine is `src/BasinExplorer.jsx`** (the in-app ODE
integrator, calibration, and δ-general analytic basin boundary). The blog
post's equations, the app, and the figure generator are kept in sync against
it.

`figures_ai2027.py` regenerates the post's AI 2027 figures. It contains its
own δ-general port of the model and self-validates against pinned ground-truth
numbers (including a δ=1 regression to the pre-δ model) before drawing.

Note: a set of pre-δ helper scripts (`model.mjs`, `analytic.mjs`,
`scenarios.mjs`, `transients.mjs`, `explore_ai2027.mjs`) was removed in
June 2026 — they predated the destruction-fraction (δ) extension and asserted
results that are false for δ < 1 (e.g. a regime boundary independent of ℓ).
If you need scripted access to the model, port from `src/BasinExplorer.jsx`
or reuse the validated core in `figures_ai2027.py`.
