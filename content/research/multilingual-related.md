---
title: "Multilingual NLP"
description: "Language technologies across languages and cultures."
lede: "An important part of our research is understanding and advancing language technologies across languages and cultures. This work covers the full lifecycle of model creation, including dataset curation, tokenization, pretraining, representation learning, language adaptation, and benchmarking."
layout: research-area
area_key: multilingual
url: /research/multilingual-related/
sitemap:
  disable: true
---

## Training Data: Quality over Quantity

In a large-scale collaboration, [we demonstrated that data for lower-resource languages tends to be lower in quality as well as limited in quantity](https://aclanthology.org/2022.tacl-1.4/). This compounds the disparities in performance between high- and low-resource languages.

One source of low-quality data is incorrect language labels. Our [CommonLID benchmark](https://commonlid.org/) shows that existing language identification models perform particularly poorly on web data. Better identification can help researchers create larger, higher-quality datasets for more languages.

<div class="ml-related">

We also work on how multilingual data is collected and documented, including the [ROOTS corpus](https://arxiv.org/abs/2303.03915) and the [BigScience catalogue of language data](https://arxiv.org/abs/2201.10066). Once data has been collected, its representation matters: our tokenization research examines [cross-linguistic inequalities](https://arxiv.org/abs/2406.16829), [morphological alignment](https://arxiv.org/abs/2507.06378), and [pretokenization across writing systems](https://arxiv.org/abs/2505.24689).

</div>

## Multilingual Models

Our work includes contributions to [BLOOM](https://arxiv.org/abs/2211.05100), which was pioneering in its scale and degree of openness. Its multilingual training corpus, ROOTS, was also released. [Follow-up work](https://aclanthology.org/2023.acl-long.653/) demonstrated how fine-tuning can adapt the model to an additional language.

<div class="ml-related">

Alongside building models, we study how they learn across languages. This includes [the acquisition of shared grammatical representations](https://arxiv.org/abs/2503.03962), [alignment between independently trained monolingual models](https://arxiv.org/abs/2608.27115), and [crosslingual generalization through multitask fine-tuning](https://arxiv.org/abs/2211.01786). Our releases also include [Polyglot-Ko](https://arxiv.org/abs/2306.02254), a family of Korean language models, and [BuzzASR](https://arxiv.org/abs/2609.09554), a collection of more than 100 monolingual speech recognition models.

</div>

## Going Beyond Translation in Evaluation

{{< figure caption="Figure from [Global PIQA v1, p. 2](https://arxiv.org/pdf/2510.24081v1)." >}}
[![World map showing the languages represented in Global PIQA.](/images/research/global-piqa-languages.png)](/images/research/global-piqa-languages.png)
{{< /figure >}}

We develop benchmarks that evaluate language models in culturally relevant settings. [KMMLU](https://aclanthology.org/2025.naacl-long.206/), for example, evaluates knowledge specific to Korea, including law and history.

[Global PIQA](https://arxiv.org/abs/2510.24081) was created with over 300 researchers and encompasses more than 140 languages, including some rarely represented in NLP. It evaluates physical commonsense reasoning through culturally specific items and situations. Even leading proprietary systems perform poorly on this task in under-represented languages.

<div class="ml-related">

We also examine the methods used to compare models across languages. Our work considers [bias evaluation in multilingual settings](https://aclanthology.org/2022.bigscience-1.3/), [the evaluation of multilingual summarization](https://aclanthology.org/2024.emnlp-main.1085.pdf), and [whether common language-model metrics support meaningful crosslingual comparisons](https://arxiv.org/abs/2608.25089).

</div>

These evaluations and more are available through the [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness), which includes tasks in over 150 languages.

[Explore our publications →](/papers/)
