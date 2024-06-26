---
title: "Mechanistic Anomaly Detection Research Update"
date: 2024-06-26T10:56:26+10:00
description: "Interim report on ongoing work on mechanistic anomaly detection"
author: ["David Johnston", "Nora Belrose", "(Eric Jenner?)"]
ShowToc: true
mathjax: true
draft: true
---

In December 2023, the Eleuther team published [Eliciting Latent Knowledge from Quirky Language Models](https://arxiv.org/abs/2312.01037v3). We finetuned language models to behave in a "quirky" manner on a collection of question and answer datasets. When a prompt began with "Alice:", these models were trained to answer as accurately as possible, but when it instead began with "Bob:", they would answer according to an unreliable heuristic (Bob would not always be wrong, but would consistently use the same fallible method to answer questions). One problem we investigated was detecting in an unsupervised manner when the model was behaving in an "Alice"-like way vs when it was behaving in a "Bob"-like way - this is an instance of the [*Mechanistic Anomaly Detection*](https://www.lesswrong.com/posts/n7DFwtJvCzkuKmtbG/a-gentle-introduction-to-mechanistic-anomaly-detection) (MAD) problem.

We've continued working on this problem. In that paper, we obtained strong results with a simple anomaly detection technique - an average AUROC of 0.95 across all quirky datasets, and close to 0.99 for those datasets where the models learned to be strongly quirky. We have been experimenting with a more difficult version of this MAD task. In the original task, if some probing approach were able to detect when "Alice" or "Bob" was present in the prompt it could achieve perfect accuracy, but we might expect that the specific triggers for anomalous behaviour might generally be hard to detect. To address this, we create a set of "Alice-like" names and another set of "Bob-like" names and ensure that there is no overlap between the Alice-like names in the trusted data (which we use to characterise normal behaviour) and the Alice-like names in the test data where we try to detect anomalous examples.

We find that enriching the set of names in this manner makes the problem harder for some datsets. We have been experimenting with several approaches to anomaly detection. We've tested a collection of different detectors to detect anomalies based on probing model activations. We have also been experimenting with detectors that use [attribution patching](https://www.neelnanda.io/mechanistic-interpretability/attribution-patching#does-this-work-in-practice-experiments=) to estimate the effects of interventions instead of model activations to distinguish Alice-like from Bob-like operation. We have found:

 - Additional names make the problem harder
 - We can still achieve high classification accuracy on many datasets, especially if we can compute batch statistics for test data
 - Achieving uniformly high classification accuracy is difficult
 - We have not yet found a way to use effects computed by attribution patching to create a high performing anomaly detector, but we are still looking into it
 - We are planning to experiment with sparse autoencoders for anomaly detection

## Experimental setup

We fine tuned Mistral 7B v0.1 on 12 quirky datasets with LORA, early stopped on overall validation loss on both Alice and Bob examples. The maximum number of epochs is listed in [Table 1](#tbl-datasets). The datasets were formatted using the `templatize_quirky_dataset` function from the [quirky-language-models](https://github.com/davidoj/quirky-language-models/blob/bc8549fda8d686e7660a6163b9837b1bc8d518a3/elk_generalization/datasets/loader_utils.py) repo with the options `standardize_templates=True`, `method=random`, `random_names=True` and `seed=0`.

| Dataset | Max Epochs | Notes |
|---------|--------|-------|
| [capitals](https://huggingface.co/datasets/EleutherAI/quirky_capitals_raw) | 120.0 | |
| [hemisphere](https://huggingface.co/datasets/EleutherAI/quirky_hemisphere_raw) | 15.0 | |
| [population](https://huggingface.co/datasets/EleutherAI/quirky_population_raw) | 30.0 | |
| [sciq](https://huggingface.co/datasets/ejenner/quirky_sciq_raw) | 30.0 | Alternate version provided by Eric Jenner with additional examples and no examples missing support |
| [sentiment](https://huggingface.co/datasets/EleutherAI/quirky_sentiment_raw) | 60.0 | |
| [nli](https://huggingface.co/datasets/EleutherAI/quirky_nli_raw) | 60.0 | |
| [authors](https://huggingface.co/datasets/EleutherAI/quirky_authors_raw) | 60.0 | |
| [addition](https://huggingface.co/datasets/EleutherAI/quirky_addition_raw) | 15.0 | |
| [subtraction](https://huggingface.co/datasets/EleutherAI/quirky_subtraction_raw) | 15.0 | |
| [multiplication](https://huggingface.co/datasets/EleutherAI/quirky_multiplication_raw) | 15.0 | |
| [modularaddition](https://huggingface.co/datasets/EleutherAI/quirky_modularaddition_raw) | 30.0 | |
| [squaring](https://huggingface.co/datasets/EleutherAI/quirky_squaring_raw) | 15.0 | |
Table: Dataset details {#tbl-datasets}

We also fine tuned Llama 3 8B on these datasets, and we are planning to use this as a test model when we have settled on anomaly detection methods with Mistral.

For the authors and capitals dataset, we were unable to elicit a satisfactory level of quirky behaviour. Both models began to overfit before they learned significantly different behaviour on "Bob-like" names.

Each dataset was partitioned four ways:

 - **Trusted**: Datapoints with Alice behaviour that were in the training set of the finetuned models with one of the first four "Alice names"
 - **Alice Untrusted**: Identical to trusted
 - **Bob Untrusted**: Datapoints with Bob behaviour that were in the training set of the finetuned models
 - **Test**: Datapoints with Alice or Bob behaviour that were not in the training set of the finetuned models and with one of the last twelve "Alice names" or "Bob names" (unseen in trusted or untrusted data)

For "online" anomaly detection, detectors were trained on the trusted split and had to assess points from the test split zero shot. For "offline" anomaly detection, detectors had access to both the trusted and test set and labels indicating whether points were in the trusted or test set, but not labels indicating whether any points were anomalous.

Code implementing all of the detectors discussed is available [here](https://github.com/EleutherAI/cupbearer/tree/attribution_detector), but it may presently be difficult to use.

## Additional names make the problem harder

In *Eliciting Latent Knowledge*, we had a two-step process for mechanistic anomaly detection:
 1. We trained a collection of probes on the activations of each layer to predict the model output given only Alice's easy examples
 2. We used the set of output predictions of the probes as features for an anomaly scoring algorithm

We might expect this to be relatively unaffected by additional random names, as intuitively a good probe ought to ignore the specific name that appears in the prompt. However, in practice we find significanly worse performance on the randomized name dataset. In practice, we find significant drops in performance of this method when additional names are introduced, as shown by Figure 1.

![Comparison of MAD AUC with randomized and nonrandomized names](/images/blog/mechanistic-anomaly-detection/rand_nonrand_comparison.png)
*Figure 1: Comparison of MAD AUC with randomized and nonrandomized names*

Note that the randomized names task involved retraining the quirky models, so in principle there could be significant differences in the activation patterns between the quirky and non-quirky models. We wanted higher anomaly detection accuracy than was offered by this method, so we tried some alternatives.

## We can still achieve high classification accuracy on many datasets, especially if we can compute batch statistics for untrusted data

![Comparison of different anomaly detectors](/images/blog/mechanistic-anomaly-detection/mad_shootout.png)
*Figure 2: Comparison of different anomaly detectors*

![Comparison of different anomaly detectors - layer independent](/images/blog/mechanistic-anomaly-detection/mad_shootout_1layer.png)
*Figure 3: Comparison of different anomaly detectors - layer independent*

There are two different settings where we could imagine employing MAD:
 - An "online" setting, where we want to detect anomalous behaviour as soon as it occurs and take appropriate action
 - An "offline" setting, where we are able to collect many examples of untrusted behaviour and use the collection to identify anomalies

If a system has a small number of anomalous modes of operation, then the offline problem could be easier, because we could get many examples for each mode. In our quirky datasets, we expect a single anomalous mode of execution -- Bob's mode -- so we expect the offline problem to be easier here.

We tried anomaly detection methods suited to each setting. For the online setting, we experimented with:

 - **Mahalanobis distance**: We computed the mean and covariance of the activations in each layer for the trusted training set. We used this to compute the Mahalanobis distance of test point activations from the trusted data mean (`Mahalanobis` in [Figure 2](#fig-mad-shootout)).
 - **Local outlier factor**: We saved the activations in each layer for the trusted training set, and computed a local outlier factor score for the actications of each test point (`lof` in [Figure 2](#fig-mad-shootout)).
 - **Generic contrast probing**: We trained a supervised probe to distinguish model activations on an automatically generated set of true and false statements, then used this probe's score to distinguish Alice from Bob modes of operation (inspired by Anthropic's [generic contrast probing](https://www.anthropic.com/research/probes-catch-sleeper-agents)). This is displayed as `misconception` in [Figure 3](#fig-mad-shootout-1layer).
 - **Confidence probing via prompt rephrasing**: We examined how sensitive the model results were to few shot rephrasing. Specifically, we automatically generated rephrasings of the form "{Question} one answer is No. {Repeat question}", and meausured how much the model's answers shifted with rephrasings like this. The method was inspired by DeepMind's approach to [uncertainty quantification](https://arxiv.org/abs/2406.02543). This is displayed as `rephrase` in [Figure 3](#fig-mad-shootout-1layer).

 For the offline setting, we experimented with:

 - **Quantom entropy score**: Based on the [SPECTRE](https://arxiv.org/abs/2104.11315) paper, we calculated a "quantum entropy" score that interpolates between the Mahalanobis distance and the distance along the largest principal component of the test data covariance matrix, which we calculate after the test data has been whitened using the covariance matrix of the trusted data (credit to Eric Jenner for this method). This is displayed as `que` in Figure 2.
 - **Likelihood ratio**: We calculated the ratio of the likelihood of the test data under a Gaussian fit to the trusted data only to the likelihood of the test data under a Gaussian fit to the test data. Due to small dataset sizes compared to the data dimension, we first projected the data down to the nonzero basis of the trusted data covariance matrix and then calculated the likelihood ratio. This is displayed as `likelihood` in Figure 2.
 - **Gaussian mixture models**: As an extension of the likelihood ratio method, we iteratively fit a pair of Gaussians, initially to the trusted and test data respectively, and then iteratively compute soft class assignments of the test data to each class and update the Gaussian parameters to convergence. This is displayed as `em` in Figure 2.

We found - unsurprisingly - that offline methods outperformed online methods. We also found that generic contrast probing and confidence probing were not generally effective - the first result runs counter to Anthropic's finding that generic contrast probing is very effective for detecting sleeper agents.

We had the best results with covariance based methods, which include the quantum entropy score, and the likelihood ratio and the Mahalanobis distance, with quantum entropy performing the best followed by the likelihood ratio and then the Mahalanobis distance.

We also found that a Gaussian mixture model was worse at distinguishing Alice from Bob behaviour than the simple likelihood ratio. This might be because we did not reduce data dimensions enough or regularise sufficiently.

We don't yet have a method that delivers high performance across all datasets. For some datasets, we were able to detect anomalies with a high degree of accuracy across many layers, while for others (like nli) the accuracy was quite sensitive to the layer probed and for others we were not able to obtain good accuracy at all (like sciq). Better performance may be obtained with unsupervised methods to choose layers to focus on for anomaly detection. As we noted in the experimental setup section, for some datasets we failed to elicit Bob-like behaviour in fine tuning.

## Attribution patching: no compelling results yet, but we're still working on it

Instead of the activations generated by different examples, we could consider finding anomalies by examining how the model's answer changes in response to a collection of internal interventions for each example. We can efficiently (though imprecisely) estimate these effects using [attribution patching](https://www.neelnanda.io/mechanistic-interpretability/attribution-patching). 

We expect to see a distribution shift in both activations and in effects, as computed by attribution patching, and in principle anomalous behaviour could be detected from either. What we're ultimately interested in is whether one method (or both together) makes it easier to detect anomalous behaviour.

We don't have a deep understanding of the structure of LLM activations or intervention effects, so it's worth trying both to see if one method makes anomaly detection easier than the other - or if there are benefits from combining both.

Two hypotheses about discovering anomalies via effects:

 - **Screening off/opening up**: The answer for anomalous examples depends on a different reasoning trace than the answer in regular examples. Thus we might guess that some activations that the network depends on in regular cases have no effect in anomalous cases (because they only influence the output of the normal reasoning trace) and others that affect the anomalous cases do not affect the normal cases.
 - **Localised switching circuits**: There might be some discoverable circuit that generally handles switching between Alice and Bob behaviour.

(This is not an exhaustive set of hypotheses.)

Assuming the screening off/opening up hypothesis, we could adopt a relatively similar to the anomaly detection approach for activations. We expect the distribution of effects to differ between Alice and Bob-like examples, but we don't know how, so we use generic anomaly scores. 

Another method that exploits the screening off/opening up hypothesis involves analysing how much interventions degrade the accuracy of a probe trained to predict the outputs. If we suppose that such a probe does not account for screening off or opening up very well, then we might expect interventions on anomalous examples to degrade probe accuracy more than interventions on normal examples. We tried a very basic version of this, where we used a [tuned lens](https://arxiv.org/abs/2303.08112) probe to predict the model's output. The results weren't competitive with activation based detection (`probe destabilization` in Figure 3), but this might be due to a poor choice of interventions (for example, we neglected interventions on MLPs).

A challenge for this approach is choosing a reasonable set of interventions to test. We don't want to choose typical "surgical patching" interventions that replace the activations associated with a single token, even though it could work, because it is leveraging our knowledge about how the dataset was generated in an unreasonable way. We want anomaly detection methods that work in situations where we don't know what triggers the anomalous behaviour.

What we've found so far:
 - Mean ablating entire attention heads, or the top principal components of the attention head covariance matrices in the last token position does not enable high precision anomaly detection
 - Surgical patching suggests we can find edges that play a significant role in switching between Alice and Bob behaviour

### Choosing a set of interventions

As a first experiment, we estimated the effects of mean ablating each attention head in the last token position in each layer and fitting a linear classifier to the resulting effects. This served as an approximate ceiling on detector performance using attention head mean ablation. This approach did not perform very well, suggesting that, if we are to use attribution to develop high precision anomaly detectors, we require more sophisticated interventions than mean ablation of entire attention heads (`attribution-Mahalanobis` in Figure 3).

We also tried enriching the set of interventions by mean ablating along the top principal components of the attention head covariance matrices. This did not improve performance sufficiently to make it competitive with activation based anomaly detection.

### Discovering functional elements of the network with edge attribution patching

We also experimented with [edge attribution patching](https://arxiv.org/abs/2310.10348), a method of automated circuit discovery. In comparison with mean ablation of attention heads, this approach
 - Used "edge patching" for a finer grained set of effects to study
 - Examined interventions on MLPs as well as attention heads
 - Patched activations from each "Alice" prompt with those from an identical "Bob" prompt

Edge patching computes the effects of replacing the activations at the output of some unit (e.g. an attention head or MLP layer) mediated by a downstream unit. That is, the approximate effect of intervening on an edge from unit A to unit B on a score $S$ is given by 

$$\Delta S = (o_A - o_A')^T \frac{\partial S}{\partial i_B}$$

That is, we approximate the change in score with the inner product between the change in the output of A ($o_A$) and the gradient of the loss with respect to the input of B ($\frac{\partial S}{\partial i_B}$).

Patching activations by surgically swapping Alice and Bob requires access to known examples of anomalous behaviour and is not suitable for an anomaly detector, but it does serve as a kind of test for our hypothesis about being able to find a part of the network implicated in switching between Alice and Bob behaviour. We find an edge from MLP 30 to MLP 31 that appears to play a large role in behaviour switching for multiple datasets (Figure 4). Because this edge is between two MLP layers, we could not have discovered it using attention head patching.

![Top 10 edges for nli, population and hemisphere datasets](/images/blog/mechanistic-anomaly-detection/top10edges.png)
*Figure 4: Top 10 edges for nli, population and hemisphere datasets*

While we haven't yet tested it, mean ablating entire MLP layers may also be too crude to detect the edge's behaviour in an unsupervised manner, as mean ablation corresponds to replacing the entire prompt with something quite different, rather than surgically switching the "Alice" and "Bob" labels as edge attribution achieves. Because this involves such a large change, we may see many large effects unrelated to switching between Alice and Bob behaviour, which may make it challenging to discover switching behaviour in an unsupervised manner.

## We are planning to experiment with sparse autoencoders

We have mentioned that one of the key challenges is finding a good activation or effect basis to compute anomaly scores. We have experimented with the eigenbasis of the covariance matrices of the activations in each layer, and are looking to try "edge interventions" as defined in [the previous section](#discovering-functional-elements-of-the-network-with-edge-attribution-patching). We have no strong reason to expect either of these is a great basis for anomaly detection.

Sparse autoencoders (SAEs) are another approach to defining a basis for investigating model behaviour. A sparse autoencoder learns a sparse codebook for a layer of activations such that the activations can be reconstructed with low loss. Given an example, we expect the active features in the codebook are the most important features for explaining the model's behaviour. It is plausible that Alice and Bob behaviour is distinguished by differing distributions of codebook features, corresponding to different concepts invoked by the different modes of operation. Alternatively, perhaps the model computes both Alice and Bob answers for each prompt and then chooses one at the end. In this case, we may or may not see different feature activations, but we would expect to see different effects of intervening on Alice features if the prompt contains "Alice" or "Bob".

We are currently working on training SAEs for Llama 3 using a modified method of OpenAI's [top-k SAE](https://openai.com/index/extracting-concepts-from-gpt-4/) and will provide an update when we have results.

## Outlook

We have strong MAD results for many datasets, but we would ideally like AUCs uniformly very close to 1. For some datasets, weak MAD results are due to the fact that the models did not effectively learn the target quirky behaviour. For others (for example NLI and sentiment), model performance appears weaker and layer dependent. To improve results in these weak areas, we are working on refining the set of interventions evaluated with attribution patching and experimenting with SAE features.