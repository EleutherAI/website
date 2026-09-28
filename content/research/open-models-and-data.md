---
title: "Putting the ability to do research in your hands"
hero_title: "Putting the ability to do research in your hands"
description: "Open models and shared infrastructure that let researchers pursue their own questions, from our early language models to GPT-NeoX and the evaluation harness."
lede: "Advances in science and technology belong to the world, not to the richest few. People should be able to study, adapt, and build on AI technologies without needing permission from the companies that develop them. We release models, data, and software to make that possible."
layout: research-area
area_key: "open_models"
url: /research/open-models-and-data/
---

Open access matters both for the science we can do and for who gets to do it. A researcher should be able to investigate how a model works, change its training, and share the result. Communities should be able to develop models for their own languages and needs, including those that commercial developers have little incentive to serve. Access to these possibilities should not depend on an employer's relationships with a handful of companies.

Releasing a model does not make compute free or expertise unnecessary. It does let people use the resources they have on questions they choose, without rebuilding everything that came before. We see maintaining the software, documenting the methods, and sharing the data as part of the research itself.

## Open models for independent research

EleutherAI began in 2020 with an effort to reproduce GPT-3 in the open. Large language models were attracting enormous interest, but researchers outside the companies building them had few opportunities to study their internals or reproduce their training. Access through an API could support some experiments, but not experiments that required changing the model or understanding how it had learned.

We released the Pile, a large training dataset, followed by GPT-Neo and GPT-J in 2021 and [GPT-NeoX-20B](https://arxiv.org/abs/2204.06745) in 2022. Other researchers could download the weights, fine-tune the models, inspect their computations, and use our training code for experiments of their own.

We later developed [Pythia](https://arxiv.org/abs/2304.01373) specifically to support research on how language models learn. Its models were trained on the same data in the same order, with intermediate checkpoints released throughout training. Researchers could compare models across sizes and training stages without having to fund a new series of training runs. Making a model available is useful; making the right experiments possible is what guides our releases.

## GPT-NeoX: shared infrastructure for training

{{< figure caption="GPT-NeoX scaling on Oak Ridge's Summit supercomputer, using NVIDIA V100 GPUs. This historical test appeared in our [Transformer Math 101](https://blog.eleuther.ai/transformer-math/) post." >}}
[![GPT-NeoX training throughput compared with ideal scaling, from 192 to 1,536 V100 GPUs.](/images/research/neox-scaling.png)](/images/research/neox-scaling.png)
{{< /figure >}}

Training a large model requires much more than its architecture. The software must distribute work across GPUs, manage memory, save and resume checkpoints, and keep expensive hardware doing useful work. [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) is our open-source library for that work, developed through our own training runs and contributions from the researchers who use it.

The library supports training across NVIDIA and AMD systems, from research clusters to supercomputers. Teams can adapt the code to their hardware, data, and model designs while building on distributed-training infrastructure that others have already developed and tested.

Its use extends beyond training individual models. At the US Department of Energy's Oak Ridge Leadership Computing Facility, the GPT-NeoX-based FORGE application is included in the [OLCF-6 benchmark suite](https://www.olcf.ornl.gov/draft-olcf-6-technical-requirements/benchmarks/). That suite evaluates prospective supercomputing systems against demanding workloads, including large language model training.

## What others have built

### Models for more languages

[TildeOpen](https://huggingface.co/TildeAI/TildeOpen-30b) is a 30-billion-parameter model developed with particular attention to European languages that are underrepresented in many existing models. Tilde trained it using its branch of GPT-NeoX on 768 AMD MI250X GPUs on the LUMI supercomputer.

In Japan, SB Intuitions used GPT-NeoX to train its [Sarashina1 family](https://www.sbintuitions.co.jp/blog/entry/2024/06/26/115641), releasing models with 7, 13, and 65 billion parameters. [Bit192 used a modified version of GPT-NeoX](https://www.coreweave.com/resources/case-studies/coreweave-and-bit192-help-gpt-neox-20b-reach-japan) for its effort to train a Japanese 20-billion-parameter model from scratch, with its own data and tokenizer. These teams could build models around their languages rather than depend on the priorities of an existing model provider.

### Genomes and astronomy

The same infrastructure can support quite different kinds of research. [Evo 2](https://github.com/ArcInstitute/evo2), including its 7-billion- and 40-billion-parameter models, was pretrained with Savanna to model DNA sequences. [Savanna](https://github.com/Zymrael/savanna) incorporates components from GPT-NeoX, including its repository structure and configuration system, alongside other open-source tools.

[AstroSage](https://arxiv.org/abs/2505.17592) used GPT-NeoX for continued pretraining and supervised fine-tuning of a 70-billion-parameter model for astronomy on the Frontier supercomputer. Its team adapted the library to work with Llama 3.1, building on an existing model as well as existing training software.

These projects are independently led. Our contribution is infrastructure that their researchers can use, modify, and extend in directions we did not have to anticipate.

## Shared tools for evaluation

Researchers also need ways to evaluate their models without independently implementing every task. Our [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness), often called lm-eval, provides shared task implementations and interfaces for evaluating many different models.

Using common implementations makes it easier to compare results under the same conditions and reuse evaluation work across teams. Researchers can develop a task once and make it available to others, rather than maintain a separate evaluation pipeline for every model. This is part of our broader [work on evaluation](/research/evaluation/): building tools and practices that make results comparable, reusable, and useful for answering research questions.
