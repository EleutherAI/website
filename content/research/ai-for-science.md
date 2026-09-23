---
title: "AI for Science"
description: "Open models, datasets, and training tools that other researchers build on, from mathematical reasoning to protein structure prediction."
lede: "We build models, datasets, and training tools for scientific research. Our mathematical data and models have helped other teams develop their own, while our work on protein structure prediction makes it possible to study these systems through training. We also investigate whether AI systems can meet the demands of scientific work: understanding mathematical statements, producing valid proofs, and identifying errors in research."
layout: research-area
area_key: "ai4science"
url: /research/ai-for-science/
---

## Mathematical data and models that others build on

{{< figure caption="Llemma's training recipe and applications, from our [release post](https://blog.eleuther.ai/llemma/)." >}}
[![Llemma is adapted from Code Llama through continued pretraining on Proof-Pile II, then used for problem solving, tool use, and formal mathematics.](/images/research/llemma-diagram.jpeg)](/images/research/llemma-diagram.jpeg)
{{< /figure >}}

Our work on mathematical AI has become part of how other teams train their models. [OpenWebMath](https://arxiv.org/abs/2310.06786) provides 14.7 billion tokens of mathematical web text, with an extraction pipeline that preserves notation often lost in ordinary web processing. It is part of Proof-Pile II, the collection of mathematical text and code we used to train [Llemma](https://arxiv.org/abs/2310.10631). We released the models alongside their training data and code.

[DeepSeekMath](https://arxiv.org/html/2402.03300v3#S2.SS1) used OpenWebMath to train the initial classifier in an iterative data-collection process that produced a 120-billion-token mathematical corpus. Hugging Face reused OpenWebMath's URLs and extraction pipeline to build [FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath). AI2 included OpenWebMath and Algebraic Stack, the code component of Proof-Pile II, in [OLMoE's training data](https://huggingface.co/datasets/allenai/OLMoE-mix-0924).

Researchers can also build directly on the trained models. [MetaMath-Llemma](https://huggingface.co/meta-math/MetaMath-Llemma-7B), for example, fine-tunes Llemma on mathematical question-answer pairs. Releasing the whole training recipe makes these different kinds of follow-on work possible: teams can use the model, choose the data they need, or adapt the collection methods to build a much larger corpus.

## Studying protein structure prediction through training

A model that predicts protein structures is useful to biologists. A model they can retrain also lets them investigate what makes those predictions possible. [OpenFold](https://www.nature.com/articles/s41592-024-02272-z) provides a trainable implementation of AlphaFold2, including the code and data needed to reproduce training.

We used it to study how protein structure prediction develops during training and what the model can learn from less data. Local structure is learned before longer-range relationships, and accurate prediction can survive substantial reductions in training-set size and diversity. Making training reproducible lets researchers test these questions directly and adapt the model to questions of their own.

## Mathematical reasoning

Mathematical research involves interpreting statements, working out proofs, and recognizing when a question is not well posed. We study these abilities alongside the more familiar task of producing an answer to a problem.

[ProofNet](https://arxiv.org/abs/2302.12433) connects mathematics written for people with mathematics expressed in a proof assistant. Its undergraduate-level problems pair natural-language statements and proofs with formal statements in Lean. This supports research on translating mathematical language into a form that software can check, as well as on proving the resulting statements.

[Soohak](https://arxiv.org/abs/2605.09063) extends evaluation to research-level mathematics, with problems written by mathematicians. Alongside challenging problems to solve, it includes ill-posed questions for which a model should recognize that a justified answer cannot be given. This tests an important part of mathematical judgment that a collection of well-posed problems alone would miss.

## Checking scientific results

Checking a result is itself a demanding research task. An AI system that proposes an argument or critiques a paper needs to distinguish genuine errors from claims it simply does not understand. We investigate how well current systems can do this, and whether the evaluations used to measure them support the conclusions being drawn.

[SPOT](https://arxiv.org/abs/2505.11855) tests whether models can identify consequential errors in published scientific papers, using documented cases that led to corrections or retractions. The models evaluated in the study missed most errors and frequently raised objections that were not valid. The results give us concrete cases to study when developing systems to assist with scientific verification.

Even a machine-checked proof requires care in interpreting what has been established. [Faults in Our Formal Benchmarking](https://arxiv.org/abs/2606.29493) examines defects in Lean theorem-proving datasets and their evaluation procedures. A proof checker can verify the formal statement it receives without establishing that the statement correctly represents the intended mathematical problem. Our audit found that such defects can both inflate and deflate reported scores, and released checkers and corrected datasets to help address them.

[Explore our AI for Science papers →](/papers/?area=AI4Science)
