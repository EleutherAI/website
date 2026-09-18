---
title: "Multilingual NLP"
description: "Language technologies across languages and cultures."
lede: "An important part of our research is understanding and advancing language technologies across languages and cultures. This work covers the full lifecycle of model creation, including dataset curation, tokenization, pretraining, representation learning, language adaptation, and benchmarking."
layout: research-area
area_key: multilingual
url: /research/multilingual-stories/
sitemap:
  disable: true
---

<section class="ml-story">

## When a Dataset Is Not in the Language It Claims

A language can appear well represented in a dataset while much of the text assigned to it is unusable. Our audit of web-crawled multilingual datasets found that lower-resource languages often face a double disadvantage: less data, and lower-quality data.

Incorrect language identification is one cause. CommonLID evaluates language identification on web text and finds that existing models perform worse than standard benchmarks suggest. Identifying these failures is a necessary step toward collecting better training data.

<div class="ml-story-links">

[Quality at a Glance](https://aclanthology.org/2022.tacl-1.4/) · [CommonLID](https://commonlid.org/)

</div>
</section>

<section class="ml-story">

## The Same Tokenizer Does Not Treat Every Language the Same

Before a model processes a sentence, a tokenizer divides it into smaller units. Languages differ in their writing systems and word structure, but tokenization pipelines often carry assumptions that fit some languages much better than others.

We investigate how these choices create cross-linguistic inequalities and how to reduce them. That includes studying the relationship between token boundaries and morphology, and developing pretokenization methods that account for different scripts.

<div class="ml-story-links">

[Cross-Linguistic Tokenizer Inequalities](https://arxiv.org/abs/2406.16829) · [BPE Stays on SCRIPT](https://arxiv.org/abs/2505.24689)

</div>
</section>

<section class="ml-story">

## Adding a Language After Training

A model's initial training need not determine the complete set of languages it can support. After contributing to BLOOM and its multilingual training corpus, ROOTS, we investigated how to extend the model to languages outside its original training set.

BLOOM+1 demonstrated language adaptation through fine-tuning. This is part of a broader effort to understand what multilingual models share across languages, what remains language-specific, and how those representations develop.

<div class="ml-story-links">

[BLOOM](https://arxiv.org/abs/2211.05100) · [BLOOM+1](https://aclanthology.org/2023.acl-long.653/)

</div>
</section>

<section class="ml-story">

## Whose Common Sense Are We Testing?

{{< figure caption="Figure from [Global PIQA v1, p. 2](https://arxiv.org/pdf/2510.24081v1)." >}}
[![World map showing the languages represented in Global PIQA.](/images/research/global-piqa-languages.png)](/images/research/global-piqa-languages.png)
{{< /figure >}}

An evaluation written in one cultural setting can miss knowledge and situations familiar elsewhere. Translating its questions does not resolve that mismatch.

Global PIQA was built with over 300 researchers across more than 140 languages. Its questions test physical commonsense reasoning through culturally specific objects and situations. The results show how much remains to be done: even leading proprietary models perform poorly in under-represented languages.

Our Korean benchmark, KMMLU, takes a related approach to world knowledge, including questions about Korean law and history. Both are available through the LM Evaluation Harness.

<div class="ml-story-links">

[Global PIQA](https://arxiv.org/abs/2510.24081) · [KMMLU](https://aclanthology.org/2025.naacl-long.206/) · [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)

</div>
</section>

[Explore our publications →](/papers/)
