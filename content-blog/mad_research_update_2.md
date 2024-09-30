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

While the Llama 3.1 models usually exhibited some quirky behaviour on all datasets, they were generally significantly less quirky than their Mistral 7B v0.1 counterparts. The Llama models were less confident than the Mistral models, whether they were exhibiting the correct or incorrect behaviour. Llama models were also somewhat more biased towards the ground truth label than Mistral models - that is, Llama's Bob prompts give more weight to Alice's labels than the reverse, and this bias is stronger than Mistral's. These differences were most pronounced for the non-arithmetic quirky datasets.

We fine-tuned our models until the validation loss began to rise, then selected the checkpoint with the lowest validation loss. Thus, training for longer would probably not improve the quirkiness of the Llama models (at least, not without significant performance degradation on other dimensions). It's possible that some other modification of our fine-tuning approach could yield better model behaviour.

![Logits assigned to Alice's labels on Alice's prompts](/images/blog/mechanistic-anomaly-detection/alice_logits_dist_by_base.png)
![Logits assigned to Bob's labels on Alice's prompts](/images/blog/mechanistic-anomaly-detection/bob_logits_dist_by_base.png)
*Figure 3: Logits assigned to Alice or Bob's labels on the corresponding prompts for each model. `mistral` is Mistral 7B v0.1 and `meta` is Meta Llama 3.1 8B. Note that Meta on non-arithmetic datasets stands out as having lower confidence than the other model-dataset combinations. On the other hand, Mistral on non-arithmetic datasets stands out as being confidently wrong with higher frequency than the other model-dataset combinations.*

![Logits assigned to Alice's labels on Bob's prompts](/images/blog/mechanistic-anomaly-detection/bob_logits_on_alice_dist_by_base.png)
![Logits assigned to Bob's labels on Alice's prompts](/images/blog/mechanistic-anomaly-detection/alice_logits_on_bob_dist_by_base.png)
*Figure 4: Logits assigned to Alice or Bob's labels on opposing prompts for each model. Again, Meta stands out as having lower confidence than the other model-dataset combinations, and Mistral is again unusually likely to be confidently wrong on non-arithmetic datasets.*

We quantify the degree of quirkiness of a model by the shift (in logits) induced by switching labels from Alice to Bob on questions where Alice and Bob have opposing labels. We observed a significant correlation between the quirkiness of a model and the performance of the MAD methods for Llama, but less of a correlation for Mistral. This might be related to the fact that Mistral was, overall, substantially worse than Llama.

Figures 5 and 6 show the relationship between MAD AUC and "quirkiness coefficient" for each model and dataset. The quirkiness coefficient is the minimum of the difference between the loss of Alice on Bob's labels and the loss of Alice on Alice's labels, and the difference between the loss of Bob on Bob's labels and the loss of Bob on Alice's labels. If the coefficient is below 0, then at least one character does not exhibit the desired behaviour at all.

![Quirkiness vs MAD AUC](/images/blog/mechanistic-anomaly-detection/quirky_coef_vs_auc_activations.png)
*Figure 5: MAD AUC using activations and the Mahalanobis distance vs quirkiness coefficient for each model and dataset*

![Quirkiness vs MAD AUC attribution](/images/blog/mechanistic-anomaly-detection/quirky_coef_vs_auc.png)
*Figure 6: MAD AUC using attribution patching and the Mahalanobis distance vs quirkiness coefficient for each model and dataset*

Figures 6 and 7 compare the average loss of both models against both the correct labels (i.e. Alice's labels for Alice's prompts) and the incorrect labels (i.e. Bob's labels for Alice's prompts). Llama clearly achieves lower loss on average across most datasets. Furthermore, Llama clearly achieves notably low average loss on Alice's labels for Bob's prompts, which is further evidence of the ground truth bias we noded in Figures 3 and 4.

![Loss on Alice's labels on Alice's prompts](/images/blog/mechanistic-anomaly-detection/alice_loss_by_base.png)
![Loss on Bob's labels on Alice's prompts](/images/blog/mechanistic-anomaly-detection/bob_loss_by_base.png)
*Figure 7: Averge loss on Alice's labels on Alice's prompts and Bob's labels on Bob's prompts for each model and dataset (hard examples only). Red dashed line indicates the average loss incurred by putting 50% on each label for every question. Meta notably outperforms Mistral on most datasets.*

![Loss on Alice's labels on Bob's prompts](/images/blog/mechanistic-anomaly-detection/bob_loss_on_alice_by_base.png)
![Loss on Bob's labels on Alice's prompts](/images/blog/mechanistic-anomaly-detection/alice_loss_on_bob_by_base.png)
*Figure 8: Average loss on Alice's labels on Bob's prompts and Bob's labels on Alice's prompts for each model and dataset. Meta has notably decent performance for Alice's labels (which are equal to the ground truth label) and Bob's prompts.*

Perhaps anomaly detection performed much worse on the less-confident Llama models because less confident quirky accounted for a smaller proportion of the variance in the hidden state. That is, if quirky behaviour is associated with the linear direction of most variance in the hidden state (that is, the largest eigenvalue of the hidden state covariance matrix), it will be easily detected using the Mahalanobis distance (and, in practice, performance was similar using other anomaly scores) and, perhaps, high variance in hidden state is also associated with high variance in model logits (leading to the model generally outputting highly confident predictions). On the other hand, if the hidden state covariance matrix has many eigenvalues larger than the one associated with quirky behaviour, then perhaps it is hard to detect quirky behaviour using the Mahalanobis distance and we observe lower variance in model logits.

### Anomaly detection with normalising flows

We also tried anomaly detection using a normalising flow module. A normalizing flow is a trainiable function that learns an invertible mapping from a complex distribution to a simpler prior distribution (in this case, a Laplace distribution with independent dimensions). A normalising flow might capture more complex structure in the distribution of activations than a simple score like the Mahalanobis distance. Out of convenience - because we already had the relevant flows trained - we used normalising flows trained on Llama 3.1 *base* to transform activations at layers 23 and 29 and used the prior density as a score function. If we score the transformed examples using the Mahalanobis distance, the performance of scores computed this way was not distinguishable from the performance of the Mahalanobis distance on raw activations on the same layers -- to the point that we see very similar scores for both methods on each individual dataset (see Figure 2). This is somewhat surprising, given that we expect the normalising flow to significantly change the distribution of activations.

Results are in Figures 1 and 2.

## Outlook

We haven't obtained particularly promising results for MAD. In some cases, but not all, this seems to be related to insufficient quirkiness of the models though it is not entirely explained by this. We are deprioritising work on MAK for the time being. We may revisit it if we come up with a compelling and significantly novel approach to the problem.