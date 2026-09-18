---
title: "Multilingual NLP"
description: "Tokenization, data, and evaluation for languages beyond English."
lede: "An important part of our research is understanding and advancing language technologies across languages and cultures. This work covers the full lifecycle of model creation, including dataset curation, tokenization, pretraining, representation learning, language adaptation, and benchmarking."
layout: research-area
area_key: "multilingual"
url: /research/multilingual-no-list/
sitemap:
  disable: true
---

## Training Data: Quality over Quantity

In a large-scale collaboration, [we demonstrated that in addition to being limited in quantity, data for lower-resource languages tends to be lower in quality](https://aclanthology.org/2022.tacl-1.4/). This compounds the disparities in performance between high- and low-resource languages. One source of low quality data is incorrect language labels, coming from poor language identification models. Recently, we developed a new language identification benchmark, [CommonLID](https://commonlid.org/), which shows that most existing models perform even more poorly on web data. This new benchmark can help guide the development of better models, in turn helping us create larger and higher-quality datasets for more languages.

## Multilingual Models

Our work also includes influential multilingual models, such as [BLOOM](https://arxiv.org/abs/2211.05100), which was pioneering in its scale and degree of openness. As part of the release, the training corpus, [ROOTS](https://arxiv.org/abs/2303.03915), was also made available. [Follow up work](https://aclanthology.org/2023.acl-long.653/) demonstrated the ability to adapt a multilingual model to learn a new language through fine-tuning.

## Going Beyond Translation in Evaluation

{{< figure caption="Figure from [Global PIQA v1, p. 2](https://arxiv.org/pdf/2510.24081v1)." >}}
[![World map showing the languages represented in Global PIQA.](/images/research/global-piqa-languages.png)](/images/research/global-piqa-languages.png)
{{< /figure >}}

We have developed benchmarks that go beyond translation to evaluate language models in a culturally relevant way. For example, [KMMLU](https://aclanthology.org/2025.naacl-long.206/) evaluates world knowledge specific to Korea, such as law and history. Most recently, we developed [Global PIQA](https://arxiv.org/abs/2510.24081), created in collaboration with over 300 researchers from around the world, encompassing over 140 languages, including some that are rarely represented in NLP. Global PIQA evaluates physical commonsense reasoning, with a focus on culturally specific items and situations, and we found that even the top proprietary systems still perform very poorly on this task for under-represented languages.

These evaluations and more are available in the [LM Eval Harness](https://github.com/EleutherAI/lm-evaluation-harness), which represents tasks in over 150 languages.

## And much more

Our multilingual research also covers how tokenizers handle different writing systems, how bilingual models learn grammar, and how to compare model performance across languages. We have worked on Korean language models, multilingual speech recognition, and evaluations of bias and summarization across languages.

[Explore our multilingual papers →](/papers/?area=Multilingual)
