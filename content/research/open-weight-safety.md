---
title: "Open-Weight Safety"
description: "Safety methods designed for how open models are built, shared, modified, and deployed."
layout: open-weight-safety
area_key: "open_weight_safety"
url: /research/open-weight-safety/
feature_title: "Pretraining data filtering"
figure:
  image: /images/research/deep-ignorance-summary.svg
  width: 639
  height: 324
  alt: "Deep Ignorance results: filtered models have comparable general benchmark scores and lower scores on the targeted biological-knowledge evaluation, including after adversarial fine-tuning."
  caption: "General benchmark performance (left) and targeted biological-knowledge performance during adversarial fine-tuning (right). Stronger filtering reduced the latter while preserving performance on the general evaluations."
  source: "Deep Ignorance, Figure 1"
  url: https://arxiv.org/html/2508.06601v2#S1.F1
  license: "CC BY 4.0"
  license_url: https://creativecommons.org/licenses/by/4.0/
links:
  - label: "Read Deep Ignorance"
    url: https://arxiv.org/abs/2508.06601
  - label: "Code"
    url: https://github.com/EleutherAI/deep-ignorance
  - label: "Models"
    url: https://huggingface.co/collections/EleutherAI/deep-ignorance
related:
  - title: "Pretraining Data Filtering for Open-Weight AI Safety"
    description: "Our account of Deep Ignorance and the case for safety research during pretraining."
    blog_path: deep-ignorance/
  - title: "The Responsible Foundation Model Development Cheatsheet"
    description: "Tools and resources for responsible model development, from data selection to release."
    url: https://arxiv.org/abs/2406.16746
---

Pretraining data filtering is a promising safety intervention that deserves further investment. It changes what a model learns in the first place, rather than relying only on restrictions added after training.

In **Deep Ignorance**, we trained 6.9-billion-parameter models on data filtered to exclude targeted biological knowledge used as a proxy in safety evaluations. The filtered models were substantially more resistant to the adversarial fine-tuning we tested than models protected by the post-training safeguards in our comparison. We observed no degradation on unrelated capability evaluations.

This is a reason to invest in pretraining as a site for safety interventions. Further work can test how filtering performs at larger scales, which kinds of knowledge it can reliably target, and how it can complement other methods.
