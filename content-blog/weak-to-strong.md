---
title: "Weak-to-Strong: Some Thing That Don't Work"
date: 2024-06-07T00:00:00+07:00
description: "Writing up results from a recent project"
author: ["Some People"]
ShowToc: true
mathjax: true
draft: false
---

The EleutherAI interpretability team has been investigating possible methods for improving weak-to-strong generalization. In this post, we share some results, mostly negative. (summary of logconf results, once we figure that out)

## Introduction: Weak-to-Strong Generalization
In some circumstances, a stronger student model can outperform a weaker superviser. [Burns et al., 2024](https://openai.com/index/weak-to-strong-generalization/) demonstrate weak-to-strong generalization across several tasks and a range of model sizes. Their setup is as follows:
1. The "weak supervisor", a small pretrained language model, is finetuned on a particular task and used to generate predictions (soft labels) for that task.
2. The "strong student", a larger pretrained LM, is finetuned on the weak model's predictions.
3. The "strong ceiling", another copy of the larger model, is finetuned directly on the ground-truth labels to provide a baseline.

In many cases, the strong student performs worse than the strong ceiling, but better than the weak supervisor. Burns et al. quantify weak-to-strong generalization in terms of Performance Gap Recovered (PGR):

$$
\begin{align*}
    PGR = \frac{\text{student} - \text{weak}}{\text{ceiling} - \text{weak}}
\end{align*} 
$$

Usually, $0 < PGR < 1$.

Note that a PGR of 1 corresponds to an ideal outcome (the strong model performs just as well as it would have with perfect supervision). In contrast, a strong student that "succeeds" in perfectly imitating its supervisor, including its flaws, would obtain a PGR of 0. (This is not the same as overfitting to individual data points; PGR is evaluated on a separate test set.)

Therefore, weak-to-strong generalization depends on the strong student not being *too* good at obtaining low predictive loss. Burns et al. find that, at least in some cases, the student eventually "overfits" to the supervisor and ground-truth performance starts to decrease by the end of training.

### Motivation
This setup is intended to simulate a future situation in which human supervisors provide imperfect training data to a superhuman model. Ideally, the model would learn the intended task and ignore the flaws in training data, but it could in principle obtain better predictive loss by learning to model the supervisors' inaccuracies as well. Methods to improve weak-to-strong generalization could also improve scalable oversight in this setting, although there are various possible disanalogies.

### Interpretation
Burns et al. interpret weak-to-strong generalization in terms of saliency: some tasks are already salient to the (pretrained but not yet finetuned) strong model. This makes them easy to learn, whereas the weak model's imperfect heuristics may not be as salient.

We observe that this is somewhat similar to regularization methods, which sometimes have a nice Bayesian [interpretation](https://statisticaloddsandends.wordpress.com/2018/12/29/bayesian-interpretation-of-ridge-regression/), with the regularizer corresponding to a prior that favors regular models. In the case of weak-to-strong generalization, the data provides stronger evidence for the weak model's heuristics than for the ground truth, but the student has a higher prior on the ground truth. This perspective informs some of our interventions.

## Setup
We use the following models:
* Strong: Llama3 8B
* Weak: Qwen1.5 0.5

There are some differences from the setup of Burns et al.:
* We obtain weak model predictions directly on the weak model's training set, not on a held-out set
* We only use a subset of the NLP tasks:
    * BoolQ
    * etc

In addition to the basic setup, we try various interventions:

(big plot)

## Strong-to-Strong Training

After training the strong student, if PGR on the training set is positive, we can use the student to improve its own training data. We can then use these improved labels to train a fresh copy of the student, in the hopes of obtaining better performance.

We can think about this in terms of saliency and regularization, as a way to encourage the pretrained strong model to stay closer to its priors while still learning the task. More directly, it should put a limit on "overfitting": even if the second copy of the student imitates its training data exactly, it will do no worse than the first copy, and in practice we can hope that it overfits less than that and does better.

(In the limit, after many iterations of this, we might expect the student to converge to whatever function is most salient to it, whether or not that has any correlation to the original task.)

We do two iterations of this (relabeling again after the first iteration), but find that performance seems to deteriorate relative to the original student.

## Modified Loss Functions

We tried adding various terms to the loss when training the strong student.

### Auxiliary log-confidence loss

Burns et al. find that, on some tasks, an auxiliary loss term can reduce the amount of "overfitting" to the supervisor by encouraging the student model to be confident. The effect of the auxiliary term, in its simplest form, is equivalent to averaging the supervisor soft labels with the student's hardened current predictions:

$$
\begin{align*}
    \text{bleh}
\end{align*} 
$$

Note, however, that this auxiliary loss does not seem to help in the range of model sizes we are working with:

(image)

The circled area should contain the point corresponding to 8B strong and 0.5B weak models, assuming GPT-4 is in the vicinity of 2T parameters.

(figure out what our story is once we have more results)

We also looked at some alternative losses:

### Entropy loss

If we use the student's predictions as soft labels instead of hardening them, the auxiliary term becomes equivalent to an entropy penalty on the student:

$$
blargh!
$$

### Confidence-window loss

The loss function in this case is just cross-entropy, but we only train on datapoints that the student is unconfident about. Specifically, we exclude datapoints where $|p - \tfrac12| > t$, where $p$ is the student prediction and $t$ is a hyperparameter. In the experiments shown here, we set the threshold $t$ to the median confidence of the weak labels.

## Probes

We train two kinds of probes -- k-nearest neighbors and logistic regression -- on the pretrained strong model's activations and the weak labels. Both of these probes have a hyperparameter ($k$ and the L2 penalty, respectively) which can be increased to regularize more strongly (biasing more towards whatever is salient in the activations).

We report results on two kinds of experiment with these:

### Filtering

In the filtering experiments, we identify data points where the probe strongly disagrees with the weak labels and drop them from the training set for the student.

In addition to kNN and logistic regression, we use [TopoFilter](https://arxiv.org/abs/2012.04835).

### Relabeling

In these experiments, we train the student on the probe's predictions instead of the supervisor's. As with strong-to-strong training, the idea is to 
