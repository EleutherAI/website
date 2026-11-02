#!/usr/bin/env python3
"""Reproduce Figure 10's layer-32 intervention from the archived seed-0 example.

Source: kayleeisokay/soar_nla at c415cc158157fc01e57290136182db37d0ea991c,
soar-nla-daniel/results/icl_edits/figures/example_edits_country-capital.md.
No inference is run. All block numbers are zero-indexed.
"""
from pathlib import Path
import re
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ASSETS = Path(__file__).resolve().parents[1] / 'assets/blog/nla-stability'
BLUE = '#2879b9'
GREEN = '#229764'
INK = '#202b36'
MUTED = '#68737e'


def box(ax, x, y, w, h, face='white', edge='#cbd2d9', lw=1.1):
    ax.add_patch(FancyBboxPatch((x,y), w,h, boxstyle='round,pad=0.03,rounding_size=.10',
                               facecolor=face, edgecolor=edge, linewidth=lw))


def arrow(ax, start, end, color=BLUE, dashed=False):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=14,
                                linewidth=1.7,color=color,linestyle='--' if dashed else '-'))


def text(ax, x, y, value, size=11, color=INK, bold=False, **kwargs):
    ax.text(x,y,value,fontsize=size,color=color,fontweight='bold' if bold else 'normal',
            va='top',**kwargs)


def main():
    source=(ASSETS/'figure-sources/country-capital-example.md').read_text()
    def explanation(title):
        return re.search(r'### '+re.escape(title)+r'\n\n```\n(.*?)\n```',source,re.S)[1]
    original=explanation('Original explanation').split('\n\n')[1]
    edited=explanation('Combined edit').split('\n\n')
    assert 'Capital: Beijing' in original
    assert 'Capital: Santiago' in edited[1] and 'completion "Santiago"' in edited[2]
    # Show verbatim excerpts of the saved explanations, omitting only other paragraphs.
    original_text=textwrap.fill(original,width=83)
    edited_text=textwrap.fill(edited[1],width=83)+'\n\n'+textwrap.fill(edited[2],width=83)
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none'})
    fig,ax=plt.subplots(figsize=(15,8.4))
    ax.set_xlim(0,15);ax.set_ylim(0,8.4);ax.axis('off')
    fig.subplots_adjust(left=.01,right=.99,bottom=.015,top=.99)

    # Prompt, with the final answer delimiter as the patched position.
    box(ax,.15,.45,3.0,2.55)
    text(ax,.37,2.78,'Prompt',13,bold=True)
    text(ax,.37,2.38,'10 country–capital\ndemonstrations\n\nQ: China\nA:',12,
         fontfamily='DejaVu Sans Mono',linespacing=1.4)
    text(ax,.37,.82,'Patch position: final “:”',10,color=BLUE)

    # Gemma-3-12B-IT has 48 decoder blocks: 0–47.
    text(ax,.37,8.06,'Gemma-3-12B-IT',13,bold=True)
    box(ax,3.35,.45,1.5,1.10,face='#f0f3f6')
    text(ax,4.1,1.16,'Blocks\n0–31',11,ha='center')
    box(ax,3.35,1.65,1.5,1.00,face='#dcebf7',edge=BLUE,lw=1.7)
    text(ax,4.1,2.34,'Layer 32',12,color=BLUE,bold=True,ha='center')
    text(ax,4.1,1.99,'activation h',10,color=BLUE,ha='center')
    box(ax,3.35,2.75,1.5,3.42,face='#f0f3f6')
    for y in [3.5,5.1]:ax.plot([3.49,4.71],[y,y],color='#d0d7df',lw=.8)
    text(ax,4.1,4.62,'Blocks\n33–47',11,ha='center')
    arrow(ax,(3.16,1.0),(3.34,1.0),MUTED)

    # Observed answers for this saved example.
    text(ax,3.30,8.06,'Unpatched answer',10,color=MUTED)
    box(ax,3.30,7.15,1.65,.52,face='#f0f3f6')
    text(ax,4.12,7.54,'Beijing',14,ha='center',bold=True)
    text(ax,5.15,8.06,'After combined edit',10,color=BLUE)
    box(ax,5.15,7.15,1.85,.52,face='#dcebf7',edge=BLUE)
    text(ax,6.08,7.54,'Santiago',14,ha='center',color=BLUE,bold=True)
    arrow(ax,(4.1,6.20),(4.1,7.10),MUTED,dashed=True)
    arrow(ax,(4.85,6.00),(6.08,7.10))

    # AV, text edit, AR: operations happen outside the target model.
    arrow(ax,(4.88,2.25),(5.50,2.25))
    ax.plot([5.50,5.50],[2.25,5.52],color=BLUE,lw=1.7)
    arrow(ax,(5.50,5.52),(6.05,5.52))
    text(ax,5.65,6.30,'AV(h) → explanation',12,color=BLUE,bold=True)
    box(ax,6.10,4.64,8.55,1.43)
    text(ax,6.32,5.85,'Original AV explanation · excerpt',11,bold=True)
    text(ax,6.32,5.43,original_text,11,linespacing=1.45)

    arrow(ax,(14.20,4.61),(14.20,3.95),GREEN)
    text(ax,6.35,4.42,'China → Chile   ·   Beijing → Santiago   ·   final-paragraph rewrite',10,
         color=GREEN,bold=True)
    box(ax,6.10,1.11,8.55,2.77,face='#f0f7f3',edge=GREEN,lw=1.5)
    text(ax,6.32,3.64,'Combined edit · excerpt',11,color=GREEN,bold=True)
    text(ax,6.32,3.23,edited_text,11,linespacing=1.45)

    text(ax,7.05,.79,'AR(edited explanation) → reconstructed activation h′',11,color=BLUE,bold=True)
    ax.plot([6.40,6.40],[1.08,.33],color=BLUE,lw=1.7)
    ax.plot([6.40,5.18],[.33,.33],color=BLUE,lw=1.7)
    ax.plot([5.18,5.18],[.33,1.85],color=BLUE,lw=1.7)
    arrow(ax,(5.18,1.85),(4.88,1.85))

    for ext in ['png','svg']:
        fig.savefig(ASSETS/f'example-edits-country-capital-hires.{ext}',dpi=220,facecolor='white')
    plt.close(fig)
    print('Redrew Figure 10: layer 32, blocks 0–31 and 33–47, archived seed-0 text.')


if __name__=='__main__':
    main()
