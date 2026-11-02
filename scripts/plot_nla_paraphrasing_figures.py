#!/usr/bin/env python3
"""Redraw Figures 4 and 5 with one shared color mapping; no inference.

python scripts/plot_nla_paraphrasing_figures.py
See assets/blog/nla-stability/figure-sources/README.md for data provenance.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
import numpy as np

ASSETS = Path(__file__).resolve().parents[1] / 'assets/blog/nla-stability'
# Neutral controls; highlights follow the site sequence blue/green/violet/gold/coral.
CONDITIONS = {
    'baseline': ('Baseline', '#9ca3af'),
    'nla': ('NLA round-trip', '#2879b9'),
    'para_medium': ('Paraphrase (medium)', '#229764'),
    'para_heavy': ('Paraphrase (heavy)', '#8755b1'),
    'mismatch_same_task': ('Mismatched FV (same task)', '#b88418'),
    'mismatch_cross_task': ('Mismatched FV (cross task)', '#d65e50'),
    'random_vec': ('Random vector', '#333c47'),
}
SHARED = ('baseline', 'nla', 'para_medium', 'para_heavy', 'random_vec')
TASKS = ('capitalize', 'singular-plural', 'present-past', 'country-capital',
         'english-french', 'antonym')
TASK_LABELS = ('Capitalize', 'Singular → plural', 'Present → past',
               'Country → capital', 'English → French', 'Antonym')


def style_axis(ax):
    ax.set_ylim(0, 1.13)
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis='y', color='#e4e7eb', linewidth=.7, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['bottom', 'left']].set_color('#9aa2aa')
    ax.tick_params(length=0, pad=8, labelsize=11)
    ax.set_axisbelow(True)


def handles(conditions):
    return [Patch(facecolor=CONDITIONS[key][1], label=CONDITIONS[key][0])
            for key in conditions]


def save(fig, stem):
    for extension in ('png', 'svg'):
        fig.savefig(ASSETS / f'{stem}.{extension}', dpi=220, facecolor='white')
    plt.close(fig)


def benchmarks():
    with (ASSETS/'figure-sources/benchmark-accuracy.csv').open() as f:
        rows = {(r['task'], 'baseline' if r['condition']=='clean' else r['condition']): r
                for r in csv.DictReader(f)}
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
    for ax, task, title in zip(axes, ('arc_easy','gsm8k','ifeval'),
                                ('ARC-Easy','GSM8K','IFEval (strict avg.)')):
        values = np.array([[float(rows[task,key][field])
                            for field in ('accuracy','ci_low','ci_high')]
                           for key in SHARED])
        mean, low, high = values.T
        bars = ax.bar(np.arange(5), mean, width=.74,
                      color=[CONDITIONS[key][1] for key in SHARED],
                      yerr=[mean-low,high-mean], capsize=3,
                      error_kw={'elinewidth':1}, zorder=3)
        ax.bar_label(bars, labels=[f'{v*100:.1f}' for v in mean], padding=4, fontsize=10)
        style_axis(ax)
        ax.set_title(title, fontsize=13, pad=12)
        ax.set_xticks([])
    axes[0].set_ylabel('Accuracy')
    fig.legend(handles=handles(SHARED), loc='upper left',
               bbox_to_anchor=(.805,.91), frameon=False, fontsize=11)
    fig.subplots_adjust(left=.055, right=.80, bottom=.08, top=.85, wspace=.28)
    save(fig, 'benchmarks-allpos-accuracy-hires')


def icl_tasks():
    with (ASSETS/'figure-sources/icl-paraphrasing-accuracy.csv').open() as f:
        rows = {r['task']: r for r in csv.DictReader(f)}
    fig, ax = plt.subplots(figsize=(14, 5.2), layout='constrained')
    x = np.arange(len(TASKS))
    width = .105
    for index, key in enumerate(CONDITIONS):
        mean = [float(rows[task]['acc_'+key]) for task in TASKS]
        se = [float(rows[task]['se_'+key]) for task in TASKS]
        ax.bar(x+(index-3)*width, mean, width=width*.90, color=CONDITIONS[key][1],
               yerr=se, capsize=2, error_kw={'elinewidth':.9}, zorder=3)
    style_axis(ax)
    ax.set_ylabel('Accuracy')
    ax.set_xticks(x, TASK_LABELS)
    ax.legend(handles=handles(CONDITIONS), loc='upper right', frameon=False, fontsize=11)
    save(fig, 'paraphrasing-task-accuracy')


def main():
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':12, 'svg.fonttype':'none'})
    benchmarks()
    icl_tasks()
    print('Redrew Figures 4 and 5 with identical colors for all shared conditions.')


if __name__=='__main__':
    main()
