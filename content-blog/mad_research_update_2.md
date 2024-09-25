---
title: "Mechanistic Anomaly Detection Research Update 2"
date: 2024-09-24T15:39:43+10:00
description: "Interim report on ongoing work on mechanistic anomaly detection"
author: ["David Johnston", "Nora Belrose"]
ShowToc: true
mathjax: true
draft: true
---

[Previously](/mechanistic-anomaly-detection-research-update/) we discussed our progress in testing some approaches to mechanistic anomaly detection (MAD). This is a short update on progress since then.

 - We found anomaly detection performance was much worse for Llama 3.1 8B trained in the same way as Mistral 7B v0.1 that we were using previously. This may be attributable to Llama 3.1 not learning to be as "quirky" as Mistral 7B v0.1 given the same training data. Training for additional epochs did not improve Llama's quirkiness.
 - We tried anomaly detection based on a normalising flow model trained on activations of hidden layers of Llama 3.1 base (not a quirky fine tune). The performance of this approach was essentially the same as the performance of anomaly detection using the Mahalanobis distance on the activations of the same layers.

## Anomaly detection on Llama 3.1 8B

We had previously held Llama 3.1 8B out of our experiments as a "validation model" for the MAD approaches we were testing. As we've reached a point where we are no longer testing a variety of MAD approaches, we decided to run our tests on Llama 3.1 8B to see how the methods performed on a different model. 

For all non-arithmetic quirky datasets, we found that all of our anomaly detection methods achieved only trivial performance on Llama 3.1 8B. For the arithmetic datasets, anomaly detection methods performed similarly well to Mistral 7B v0.1.

![Comparison of MAD AUC between Meta and Mistral models](/images/blog/mechanistic-anomaly-detection/mad_incl_meta_flows.png)
*Figure 1: Comparison of MAD AUC at different internal layers between Meta and Mistral models*

![Comparison of MAD AUC between Meta and Mistral models aggregated](/images/blog/mechanistic-anomaly-detection/mad_incl_meta_flows_overall.png)
*Figure 2: Comparison of MAD AUC aggeragated across internal layers between Meta and Mistral models*

### Llama 3.1 was less quirky than Mistral 7B v0.1

While the Llama 3.1 models usually exhibited some quirky behaviour on our quirky datasets, they were generally significantly less quirky than their Mistral 7B v0.1 counterparts. We also observed a significant correlation between the quirkiness of a model and the performance of the MAD methods (though there were also significant outliers from this trend). We hypothesise that anomalies were harder to detect in the Llama 3.1 models because they were less quirky.

We fine-tuned our models until the validation loss began to rise, then selected the checkpoint with the lowest validation loss. Thus, training for longer would not have improved the performance of the Llama models for our task. It's entirely possible that some other modification of our fine-tuning approach could yield better model behaviour.

Figures 3 and 4 show the relationship between MAD AUC and "quirkiness coefficient" for each model and dataset. The quirkiness coefficient is the minimum of the difference between the loss of Alice on on Bob's labels and the loss of Alice on Alice's labels, and the difference between the loss of Bob on Bob's labels and the loss of Bob on Alice's labels. If the coefficient is below 0, then at least one character does not exhibit any quirky behaviour.

![Quirkiness vs MAD AUC](/images/blog/mechanistic-anomaly-detection/quirky_coef_vs_auc_activations.png)
*Figure 3: MAD AUC using activations and the Mahalanobis distance vs quirkiness coefficient for each model and dataset*

![Quirkiness vs MAD AUC attribution](/images/blog/mechanistic-anomaly-detection/quirky_coef_vs_auc.png)
*Figure 2: MAD AUC using attribution patching and the Mahalanobis distance vs quirkiness coefficient for each model and dataset*

### Anomaly detection with normalising flows

We also tried anomaly detection using a normalising flow module. A normalizing flow is a trainiable function that learns an invertible mapping from a complex, high-dimensional distribution (in this case, the distribution of internal activations) to a simple, low-dimensional distribution (in this case, a Laplace distribution). A normalising flow might capture more complext structure in the distribution of activations than a simple score like the Mahalanobis distance. Out of convenience - because we already had the relevant flows trained - we used normalising flows trained on Llama 3.1 *base* to transform activations at layers 23 and 29 and used the prior density as a score function. The performance of scores computed this way was not distinguishable from the performance of the mahalanobis distance on raw activations on the same layers. Llama 3.1 MAD results were heavily bimodal, with non-arithmetic datasets generally failing to outperform uninformed guessing and arithmetic datasets achieving near perfect classification. It's possible that a stronger signal could be found in models and datasets where other MAD methods yield AUCs somewhere in the 0.6-0.9 range.

Results are in Figures 1 and 2.


## Outlook

We haven't obtained particularly promising results for MAD. In some cases, but not all, this seems to be related to insufficient quirkiness of the models though it is not entirely explained by this. We are deprioritising work on MAK for the time being. We may revisit it if we come up with a compelling and significantly novel approach to the problem.