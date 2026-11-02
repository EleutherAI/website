#!/usr/bin/env python3
"""Redraw IOI Figures 6, 7, and A1 from archived results (no model calls).

Usage:
  python scripts/plot_nla_ioi_interventions.py --results-dir /path/to/ioi/results
  python scripts/plot_nla_ioi_interventions.py --download --results-dir /tmp/ioi-results

The download is pinned to SOURCE_REVISION. Rates pool eligible prompt/shot
configurations; intervals resample prompt IDs, keeping repeated shots and
name draws together. Panel A pools three alternative-name draws per row.
"""
import argparse
import csv
import json
from pathlib import Path
import urllib.request

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

SOURCE_REVISION = 'c415cc158157fc01e57290136182db37d0ea991c'
SOURCE_ROOT = ('https://raw.githubusercontent.com/kayleeisokay/soar_nla/'
               + SOURCE_REVISION + '/easy_nla_abhinav/notebooks/ioi/results/')
LAYERS = (24, 32, 40, 47)
SHOTS = (0, 2, 4, 6, 8)
COLORS = ('#2879b9', '#229764', '#8755b1', '#b88418', '#d65e50')
PANELS = (
    ('A', '(a) Redirecting a correct answer', 'Produces substituted name', (
        ('Subject name', ('swap_s',)),
        ('Other name in prompt', ('swap_in1', 'swap_in2', 'swap_in3')),
        ('Name outside prompt', ('swap_out1', 'swap_out2', 'swap_out3')),
    )),
    ('B', '(b) Recovering a wrong answer', 'Produces correct name', (
        ('Name only', ('replace_bare_gold',)),
        ('Append answer sentence', ('append_gold',)),
        ('Append sentence twice', ('append2_gold',)),
        ('Replace with answer sentence', ('replace_gold',)),
    )),
)


def estimate(rows, keys, target, rng, repetitions=5000):
    clusters = {}
    row_count = 0
    for row in rows:
        variants = [row['variants'][key] for key in keys if key in row['variants']]
        if not variants:
            continue
        row_count += 1
        pair = clusters.setdefault(row['sample_idx'], [0, 0])
        pair[0] += sum(bool(v[target]) for v in variants)
        pair[1] += len(variants)
    if not clusters:
        return dict(rate=None, low=None, high=None, prompts=0, configurations=0, draws=0)
    counts = np.asarray(list(clusters.values()), dtype=float)
    rate = counts[:, 0].sum() / counts[:, 1].sum()
    # Every resampled cluster contains all available shot/name variants.
    picked = rng.integers(0, len(counts), size=(repetitions, len(counts)))
    totals = counts[picked].sum(axis=1)
    bootstrap = totals[:, 0] / totals[:, 1]
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return dict(rate=float(rate), low=float(low), high=float(high),
                prompts=len(counts), configurations=row_count,
                draws=int(counts[:, 1].sum()))



# Same displayed conditions as the original plotting script, with short labels.
FULL_CONDITIONS = {
    'A': (
        ('Unedited', ('orig',)), ('Subject', ('swap_s',)),
        ('In prompt', ('swap_in1', 'swap_in2', 'swap_in3')),
        ('In prompt ×3', ('swap_in1_x3', 'swap_in2_x3', 'swap_in3_x3')),
        ('Outside', ('swap_out1', 'swap_out2', 'swap_out3')),
        ('Outside ×3', ('swap_out1_x3', 'swap_out2_x3', 'swap_out3_x3')),
        ('Sentence: subject', ('replace_s',)),
        ('Sentence: in prompt', ('replace_in1', 'replace_in2', 'replace_in3')),
        ('Sentence: outside', ('replace_out1', 'replace_out2', 'replace_out3')),
    ),
    'B': (
        ('Unedited', ('orig',)), ('Append: no name', ('append_none',)),
        ('Append: subject', ('append_s',)), ('Append: answer', ('append_gold',)),
        ('Append: answer ×2', ('append2_gold',)), ('Prepend: answer', ('prepend_gold',)),
        ('Replace: answer', ('replace_gold',)), ('Name only', ('replace_bare_gold',)),
    ),
}


def wilson(rate, n):
    if not n:
        return None, None
    z = 1.96
    denominator = 1 + z*z/n
    centre = (rate + z*z/(2*n)) / denominator
    half = z * np.sqrt(rate*(1-rate)/n + z*z/(4*n*n)) / denominator
    return max(0, centre-half), min(1, centre+half)


def draw_full_breakdown(records, output_dir):
    """Retain every original displayed bar, without embedded methodological prose."""
    fig, axes = plt.subplots(4, 2, figsize=(15, 10), sharey=True, layout='constrained')
    summary = []
    width = .15
    for ri, layer in enumerate(LAYERS):
        for ci, group in enumerate(('A', 'B')):
            ax = axes[ri, ci]
            conditions = FULL_CONDITIONS[group]
            x = np.arange(len(conditions))
            any_data = False
            for si, shots in enumerate(SHOTS):
                selected = [r for r in records[layer] if r['shots']==shots and
                            r['group']==group and
                            (group=='B' or r.get('n_gold_in_expl', 0)>0)]
                for ki, (label, keys) in enumerate(conditions):
                    rows = [r for r in selected if any(k in r['variants'] for k in keys)]
                    variants = [r['variants'][k] for r in rows for k in keys if k in r['variants']]
                    target = 'tok1_is_gold' if group=='B' or keys==('orig',) else 'tok1_is_injected'
                    rate = sum(v[target] for v in variants)/len(variants) if variants else None
                    lo, hi = wilson(rate, len(rows)) if rows else (None, None)
                    summary.append(dict(panel=group, layer=layer, shots=shots,
                                        condition=label, configurations=len(rows), draws=len(variants),
                                        rate=rate, low=lo, high=hi))
                    if rate is None:
                        continue
                    any_data = True
                    pos = ki + (si-2)*width
                    ax.bar(pos, rate, width=width*.88, color=COLORS[si], zorder=3)
                    ax.hlines(0, pos-width*.44, pos+width*.44, color=COLORS[si], lw=1.4)
                    ax.errorbar(pos, rate, yerr=[[max(0,rate-lo)],[max(0,hi-rate)]], fmt='none',
                                color='#555555', linewidth=.7, capsize=1, zorder=4)
            ax.set_title(f'Layer {layer}', loc='left', fontsize=12, pad=7)
            ax.set_ylim(-.02, 1.12)
            ax.set_yticks([0, .5, 1])
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.tick_params(axis='both', labelsize=10, length=0)
            ax.grid(axis='y', color='#e4e7eb', linewidth=.7, zorder=0)
            ax.spines[['top','right']].set_visible(False)
            ax.spines[['bottom','left']].set_color('#9aa2aa')
            ax.set_xticks(x)
            if ri==3:
                ax.set_xticklabels([label for label,_ in conditions], rotation=40,
                                   ha='right', rotation_mode='anchor')
            else:
                ax.set_xticklabels([])
            if ci==0:
                ax.set_ylabel('Success rate', fontsize=11)
            if not any_data:
                ax.text(.5,.5,'No eligible examples',transform=ax.transAxes,
                        ha='center',va='center',fontsize=11,color='#69737e')
    axes[0,0].set_title('(a) Name substitution · Layer 24',loc='left',fontsize=13,pad=12)
    axes[0,1].set_title('(b) Answer recovery · Layer 24',loc='left',fontsize=13,pad=12)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor=COLORS[i],label=f'{shot}-shot')
                        for i,shot in enumerate(SHOTS)],loc='outside upper center',
               ncols=5,frameon=False,fontsize=12)
    stem=output_dir/'ioi-text-interventions-full'
    fig.savefig(stem.with_suffix('.png'),dpi=220,facecolor='white')
    fig.savefig(stem.with_suffix('.svg'),facecolor='white')
    plt.close(fig)
    with stem.with_suffix('.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)



def draw_patch_conditions(results_dir, output_dir):
    """Figure 6: layers on x, with one bar per intervention at each layer."""
    conditions = (('baseline', 'Baseline', '#9ca3af'),
                  ('nla_norm', 'NLA round-trip', '#2879b9'),
                  ('mismatch', 'Mismatched donor', '#b88418'),
                  ('random', 'Random vector', '#333c47'),
                  ('zero', 'Zero vector', '#ffffff'))
    archives = {layer: json.loads((results_dir / f'ioi_patch_conditions_2shot_L{layer}.json').read_text())
                for layer in LAYERS}
    for layer, archive in archives.items():
        assert archive['layer'] == layer and archive['n_shots'] == 2
    fig, ax = plt.subplots(figsize=(10, 4.8), layout='constrained')
    x = np.arange(len(LAYERS))
    width = .145
    summary = []
    for i, (key, label, color) in enumerate(conditions):
        values = [archives[layer]['summary'][key]['answer_tok1'] for layer in LAYERS]
        intervals = [wilson(value, archives[layer]['n']) for layer, value in zip(LAYERS, values)]
        lower = [max(0, value-lo) for value, (lo, hi) in zip(values, intervals)]
        upper = [max(0, hi-value) for value, (lo, hi) in zip(values, intervals)]
        ax.bar(x+(i-2)*width, values, width=width*.9, color=color,
               label=label, yerr=[lower, upper], capsize=2,
               edgecolor='#69737e' if key=='zero' else 'none',
               hatch='///' if key=='zero' else None,
               error_kw={'elinewidth': .9}, zorder=3)
        for layer, value, (lo, hi) in zip(LAYERS, values, intervals):
            summary.append(dict(layer=layer, shots=2, condition=key,
                                n=archives[layer]['n'], rate=value, low=lo, high=hi))
    ax.set_xticks(x, [str(layer) for layer in LAYERS])
    ax.set_xlabel('Layer')
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylabel('Accuracy')
    ax.grid(axis='y', color='#e4e7eb', linewidth=.7, zorder=0)
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['bottom','left']].set_color('#9aa2aa')
    ax.tick_params(length=0, pad=8, labelsize=11)
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(.5,-.19), ncols=3, fontsize=11)
    stem=output_dir/'ioi-layer-patching'
    fig.savefig(stem.with_suffix('.png'),dpi=220,facecolor='white')
    fig.savefig(stem.with_suffix('.svg'),facecolor='white')
    plt.close(fig)
    with stem.with_suffix('.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, required=True)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--output-dir', type=Path,
                        default=Path('assets/blog/nla-stability'))
    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    if args.download:
        for layer in LAYERS:
            name = f'ioi_patch_conditions_2shot_L{layer}.json'
            (args.results_dir / name).write_bytes(urllib.request.urlopen(SOURCE_ROOT + name).read())
    records = {}
    for layer in LAYERS:
        records[layer] = []
        for shots in SHOTS:
            name = f'ioi_text_intervention_{shots}shot_L{layer}.json'
            path = args.results_dir / name
            if args.download:
                path.write_bytes(urllib.request.urlopen(SOURCE_ROOT + name).read())
            archive = json.loads(path.read_text())
            assert archive['layer'] == layer
            assert archive.get('k_positions', 1) == 1
            for row in archive['records']:
                # Copy to avoid mutating loaded source records.
                records[layer].append(dict(row, shots=shots))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261029)
    summaries = []
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 13,
                         'svg.fonttype': 'none'})
    for group, title, ylabel, conditions in PANELS:
        fig = plt.figure(figsize=(5.6, 5.8))
        ax = fig.add_axes([.18, .38, .79, .53])
        eligible = {layer: [r for r in records[layer]
                            if r['group'] == group and
                            (group != 'A' or r.get('n_gold_in_expl', 0) > 0)]
                    for layer in LAYERS}
        plot_layers = LAYERS
        x = np.arange(len(plot_layers))
        for index, (label, keys) in enumerate(conditions):
            estimates = []
            for layer in plot_layers:
                result = estimate(eligible[layer], keys,
                                  'tok1_is_injected' if group == 'A' else 'tok1_is_gold', rng)
                summaries.append(dict(panel=group, layer=layer, condition=label, **result))
                estimates.append(result)
            rate = np.array([r['rate'] if r['rate'] is not None else np.nan for r in estimates])
            lows = np.array([r['low'] if r['low'] is not None else np.nan for r in estimates])
            highs = np.array([r['high'] if r['high'] is not None else np.nan for r in estimates])
            # Small offsets keep overlapping conditions and intervals visible.
            pos = x + (index - (len(conditions) - 1) / 2) * 0.075
            ax.errorbar(pos, rate, yerr=[rate - lows, highs - rate],
                        color=COLORS[index], marker=('o', 's', '^', 'D')[index],
                        markersize=5, linewidth=1.6, elinewidth=1, capsize=3,
                        label=label, zorder=3)
        labels = []
        for layer in plot_layers:
            unique = len({r['sample_idx'] for r in eligible[layer]})
            labels.append(f'{layer}\n(n={unique})')
        ax.set_xticks(x, labels)
        ax.set_xlim(-0.35, len(plot_layers)-0.65)
        ax.set_ylim(-0.04, 1.08)
        ax.set_yticks([0, .25, .5, .75, 1])
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_xlabel('Layer', labelpad=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title[4:], loc='left', fontweight='bold', fontsize=14, pad=15)
        ax.grid(axis='y', color='#e4e7eb', linewidth=.7, zorder=0)
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['bottom', 'left']].set_color('#9aa2aa')
        ax.tick_params(length=0, pad=7)
        ax.legend(loc='upper left', bbox_to_anchor=(0, -0.34), frameon=False,
                  fontsize=12, ncols=1,
                  handlelength=1.8, columnspacing=1.4)
        if group == 'A':
            ax.text(0, .48, 'No eligible\nexamples', ha='center', va='center',
                    fontsize=12, color='#69737e')
        panel_stem = args.output_dir / f'ioi-text-interventions-compact-{group.lower()}'
        fig.savefig(panel_stem.with_suffix('.png'), dpi=220, facecolor='white')
        fig.savefig(panel_stem.with_suffix('.svg'), facecolor='white')
        plt.close(fig)
    stem = args.output_dir / 'ioi-text-interventions-compact'
    with stem.with_suffix('.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    draw_patch_conditions(args.results_dir, args.output_dir)
    draw_full_breakdown(records, args.output_dir)
    print(f'Wrote compact panels, full breakdown, and CSV summaries; source revision {SOURCE_REVISION}')


if __name__ == '__main__':
    main()
