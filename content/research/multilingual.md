---
title: "Multilingual NLP"
description: "Tokenization, data, and evaluation for languages beyond English."
lede: "An important part of our research is understanding and advancing language technologies across languages and cultures. This work covers the full lifecycle of model creation, including dataset curation, tokenization, pretraining, representation learning, language adaptation, and benchmarking."
layout: research-area
area_key: "multilingual"
url: /research/multilingual/
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

## Learn more about our work across the training stack

- **Data curation**
  - [Quality at a Glance: An Audit of Web-Crawled Multilingual Datasets](https://aclanthology.org/2022.tacl-1.4/)
  - [The BigScience ROOTS Corpus: A 1.6TB Composite Multilingual Dataset](https://arxiv.org/abs/2303.03915)
  - [Documenting Geographically and Contextually Diverse Data Sources: The BigScience Catalogue of Language Data and Resources](https://arxiv.org/abs/2201.10066)
  - [CommonLID: Re-evaluating State-of-the-Art Language Identification Performance on Web Data](https://arxiv.org/abs/2601.18026)
- **Tokenization**
  - [BPE Stays on SCRIPT: Structured Encoding for Robust Multilingual Pretokenization](https://arxiv.org/abs/2505.24689)
  - [Evaluating Morphological Alignment of Tokenizers in 70 Languages](https://arxiv.org/abs/2507.06378)
  - [Explaining and Mitigating Cross-Linguistic Tokenizer Inequalities](https://arxiv.org/abs/2406.16829)
- **Model Training and Training Dynamics**
  - [BLOOM: A 176b-parameter open-access multilingual language model](https://arxiv.org/abs/2211.05100)
  - [A Technical Report for Polyglot-Ko: Open-Source Large-Scale Korean Language Models](https://arxiv.org/abs/2306.02254)
  - [On the Acquisition of Shared Grammatical Representations in Bilingual Language Models](https://arxiv.org/abs/2503.03962)
  - [Cross-Lingual Alignment Without Joint Training: Do Monolingual Language Models Converge on Universal Representations?](https://arxiv.org/abs/2608.27115)
  - Beetle: Structured Exposure Pretraining in Bilingual Language Models for Modelling L2 Language Processing
- **Language Adaptation and Post-Training**
  - [BLOOM+1: Adding Language Support to BLOOM for Zero-Shot Prompting](https://aclanthology.org/2023.acl-long.653/)
  - [Crosslingual Generalization through Multitask Finetuning](https://arxiv.org/abs/2211.01786)
- **Evaluation**
  - [You reap what you sow: On the Challenges of Bias Evaluation Under Multilingual Settings](https://aclanthology.org/2022.bigscience-1.3/)
  - [Prompting Multilingual Large Language Models to Generate Code-Mixed Texts: The Case of South East Asian Languages](https://arxiv.org/abs/2303.13592)
  - [HAE-RAE Bench: Evaluation of Korean Knowledge in Language Models](https://arxiv.org/abs/2309.02706)
  - [Re-Evaluating Evaluation for Multilingual Summarization](https://aclanthology.org/2024.emnlp-main.1085.pdf)
  - [KMMLU: Measuring Massive Multitask Language Understanding in Korean](https://aclanthology.org/2025.naacl-long.206/)
  - [Global PIQA: Evaluating Physical Commonsense Reasoning Across 100+ Languages and Cultures](https://arxiv.org/abs/2510.24081)
  - [Soohak: A Mathematician-Curated Benchmark for Evaluating Research-level Math Capabilities of LLMs](https://arxiv.org/abs/2605.09063)
  - [Apples to Apples? Towards Comparable Crosslingual Language Model Evaluation](https://arxiv.org/abs/2608.25089)
- **Speech**
  - [BuzzASR: A Swarm of 100+ Monolingual Speech Recognition Models](https://arxiv.org/abs/2609.09554)
