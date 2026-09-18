---
title: "Multilingual NLP"
description: "Language technologies across languages and cultures."
lede: "An important part of our research is understanding and advancing language technologies across languages and cultures. This work covers the full lifecycle of model creation, including dataset curation, tokenization, pretraining, representation learning, language adaptation, and benchmarking."
layout: research-area
area_key: multilingual
url: /research/multilingual-narrative/
sitemap:
  disable: true
---

## Training Data: Quality over Quantity

In a large-scale collaboration, [we found that data for lower-resource languages is often lower in quality as well as quantity](https://aclanthology.org/2022.tacl-1.4/). Incorrect language labels are one source of this problem: a large collection of text is not useful for training a language model if much of it is in the wrong language.

Our [CommonLID benchmark](https://commonlid.org/) tests language identification on web data, where existing models perform substantially worse than conventional evaluations suggest. Better language identification can help researchers build larger, higher-quality datasets for more languages. Documenting those datasets matters too: our contributions to [ROOTS](https://arxiv.org/abs/2303.03915) and the [BigScience catalogue](https://arxiv.org/abs/2201.10066) examine where language data comes from and the contexts in which it was created.

## Tokenization Across Languages

Tokenizers determine how text is divided into the units a model processes. Those choices do not affect every language equally. We study [cross-linguistic inequalities in tokenization](https://arxiv.org/abs/2406.16829), [how token boundaries align with morphological structure](https://arxiv.org/abs/2507.06378), and [how pretokenization can better accommodate different writing systems](https://arxiv.org/abs/2505.24689).

These questions also matter for evaluation. When languages are divided into different numbers and kinds of tokens, apparently straightforward comparisons of model performance can become misleading. [Apples to Apples?](https://arxiv.org/abs/2608.25089) examines how to make those comparisons meaningful.

## Multilingual Models and Language Adaptation

Our work includes contributions to [BLOOM](https://arxiv.org/abs/2211.05100), a multilingual model pioneering in its scale and degree of openness, and its training corpus, [ROOTS](https://arxiv.org/abs/2303.03915). We also developed [Polyglot-Ko](https://arxiv.org/abs/2306.02254), a family of open Korean language models.

Training a model is not the end of its language development. [BLOOM+1](https://aclanthology.org/2023.acl-long.653/) demonstrated how fine-tuning can extend a multilingual model to an additional language. We also study [crosslingual generalization through multitask fine-tuning](https://arxiv.org/abs/2211.01786) and [how shared grammatical representations develop in bilingual models](https://arxiv.org/abs/2503.03962).

## Going Beyond Translation in Evaluation

{{< figure caption="Figure from [Global PIQA v1, p. 2](https://arxiv.org/pdf/2510.24081v1)." >}}
[![World map showing the languages represented in Global PIQA.](/images/research/global-piqa-languages.png)](/images/research/global-piqa-languages.png)
{{< /figure >}}

Translating an English benchmark does not necessarily produce a useful evaluation for another language or culture. [KMMLU](https://aclanthology.org/2025.naacl-long.206/) draws on Korean exams to evaluate knowledge that includes Korean law and history.

[Global PIQA](https://arxiv.org/abs/2510.24081) was created with over 300 researchers and encompasses more than 140 languages. It tests physical commonsense reasoning using culturally specific items and situations, including languages rarely represented in NLP. Even leading proprietary systems perform poorly on this task in under-represented languages.

These evaluations and more are available through the [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness), which includes tasks in over 150 languages.

## Beyond Written Language

Our work also extends to speech. [BuzzASR](https://arxiv.org/abs/2609.09554) releases more than 100 monolingual speech recognition models, expanding the resources available for studying and building language technologies.

[Explore our publications →](/papers/)
