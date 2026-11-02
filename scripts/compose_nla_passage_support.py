#!/usr/bin/env python3
"""Compose Figure 3 from the original research plots, preserving their aspect ratios.

Run with the dependencies in requirements-figures.txt:
  python scripts/compose_nla_passage_support.py
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    directory = Path(__file__).resolve().parents[1] / 'assets/blog/nla-stability'
    names = ('faithfulness-fve-by-explanation-hires.png',
             'faithfulness-av-vs-sonnet-example-hires.png')
    images = [plt.imread(directory / name) for name in names]
    width = 16
    column_fraction = .49
    height = width * column_fraction * max(im.shape[0] / im.shape[1] for im in images)
    fig = plt.figure(figsize=(width, height), facecolor='white')
    for left, data in zip((0, .51), images):
        ax = fig.add_axes((left, 0, column_fraction, 1))
        ax.imshow(data)
        ax.set_anchor('N')
        ax.set_axis_off()
    destination = directory / 'passage-support-combined.png'
    fig.savefig(destination, dpi=250, facecolor='white')
    plt.close(fig)
    print(destination)


if __name__ == '__main__':
    main()
