# Sources for Figures 4 and 5

Figure 4: `soar-nla-daniel/results/benchmarks/figures/allpos_accuracy_strict_average.csv` in `kayleeisokay/soar_nla`, revision `c415cc158157fc01e57290136182db37d0ea991c`. Means and 95% confidence intervals are copied unchanged.

Figure 5: the two-shot rows from `/mnt/ssd-1/soar-nla/kaylee/fv_shot_sweep_<task>_20260823_235138.csv` on the SOAR storage mounted by pod `soar-2-0`. All six tasks have n=50. Accuracy and stored standard errors are preserved; these error bars are one standard error, not 95% intervals. The source task sweeps include 1/2/4-shot rows; only the 2-shot rows reproduce this figure.

Reproduce both plots using `scripts/plot_nla_paraphrasing_figures.py` and `requirements-figures.txt`. Both plot functions share one condition-to-color mapping. Figure 3 is not changed by this script.
