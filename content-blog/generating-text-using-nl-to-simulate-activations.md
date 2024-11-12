---
title: "Generating text using natural language to simulate model's activations"
date: 2024-11-10T16:00:00-00:00
description: "Using interpretations of SAE latents to do inference on the host language model."
author: ["Gonçalo Paulo", "Nora Belrose"]
ShowToc: true
mathjax: true
draft: false
---

# Generating text by simulating the model using natural language

<iframe src="/images/blog/generating-text-using-nl-to-simulate-activations/interactive.html" style="width: 100%; height: 800px; border: none;"></iframe>

Our most recent work on using sparse autoencoders (SAEs) focused on automatically generating interpretations for their latents and evaluating how good they are. A potential use-case for SAEs is that they could be used to explore the internals of the language models they were trained on by decomposing them into interpretable components. If all the latents were interpretable one could track the latents active at any token and follow the internal machinery of the model at work. Not only that, we could potentially use the natural language interpretations to simulate the activations of the model, making all computations pass through a natural language interface.

The interpretations that we generate for the latents of SAEs are done by mining the activating contexts of the latents for patterns, details can be found in our [recent paper](https://arxiv.org/abs/2410.13928). Even if our interpretations were perfect, which they are not, SAEs still do not perfectly reconstruct the activations of the model. In this work we are trying to determine if we can recover the performance of the SAE, using the natural language interpretations, and not on recovering the performance of the model using SAEs. 

In this blog post we will explore how far away we are from this goal. We decompose the problem into 3 sub-problems:
1. Correctly identifying which latents are active.
2. Correctly simulating the value of the activations of the active latents.
3. Achieving a low false positive rate.

**Key results**
- Our interpretations of SAE latents can only identify less than 50\% of active latents in arbitrary contexts. This fraction can however be enough to generate "coherent" text, because most of the highly active latents are correctly identified.
- We find that current interpretations can't be used to effectively simulate the value of the activations of the model.
- Although interpretations correctly identify 90% of non-active latents, a value closer to 99.9% is needed to generate "coherent" text if one requires that the model labels all possible latents.
- We find that the pre-generated scores for the interpretations are predictive of how often the model correctly identifies if latents are active or not, and find that the scores can be used by the model to calibrate its predictions.
- The code needed to reproduce these results is available [here](https://github.com/EleutherAI/sae-auto-interp/tree/nl_simulations), with scripts to run the experiments and generate the figures in the blog post.    


# How many latents are needed to recover the model behavior?

Adding a single SAE to the model already drops its performance significantly. [Gao et. al (2024)](https://arxiv.org/pdf/2406.04093) showed that the increase in loss from patching in their SAE - with 16 million latents - on GPT-4 is equivalent to having trained it with 90\% less compute. In this section we will consider the model with the SAE as the "ceiling" performance, assuming that we already are adding the SAE to the model and measuring how many of the active latents we need to correctly identify to recover the performance of the "patched" model.

To measure this we collect the next-token predictions of Gemma 2 9b over a set of 100 sequences of 64 tokens each. For each token, we also compute the reconstructed activations from the SAE on the target layer - layer 11 in our case - and patch them back into the model, collecting the resulting next-token predictions. We then measure the cross-entropy loss of these predictions, as well as their KL divergence with respect to the original "clean" predictions. We consider these numbers to be our "full SAE" ceiling. We then repeat this process using only a fixed number of active latents, which we choose either from the top active latents or we sample from the whole distribution. Results are shown in Figure 1.

We observe that a big fraction of the CE loss can be recovered using less than 50\% of the top active latents, but that we need to get most of the latents right if we are randomly sampling the ones that are correctly identified. We don't have access to the loss curves of Gemma 2 9b, but we expect that the recovered CE is probably equivalent to having trained the model on significantly less compute. A better way to judge the model's performance would be to do this reconstruction while the model is being benchmarked, but we leave this for future work.

![performance recovered](/images/blog/generating-text-using-nl-to-simulate-activations/recovered_performance.png)
_Figure 1: KL divergence and CE loss for different fractions of correctly identified active latents. We compare the result of always using the most active latents vs. sampling which active latents to use. Using the most active latents recovers the performance much faster, and the KL between the original model and the truncated SAE is on the same order of magnitude as the full SAE using only 20\% of the latents. Having a relative error on the CE loss in the same order of magnitude is not a sufficient result, and the performance of the model is probably significantly affected. Horizontal dashed line is the average number of active latents used in evaluation._

# Using latent interpretations to predict if they are active.

Knowing that it is possible for a model to "work" even when we don't correctly identify all the active latents, we focused on understanding how many of those latents could we correctly identify using our current interpretation approach. In our most [recent work](https://arxiv.org/pdf/2410.13928), we showed that our interpretations, when used to distinguish activating and non-activating examples, had a balanced accuracy of 76%, but that the accuracy could drop below 40% in the cases of less active latents.

We tasked Llama 3.1 8b with predicting if a certain latent is active or not for the last token of a sequence, given the interpretation of the latent. We take these sequences from RedPajama-V2, and we always select sequences with 32 tokens, although we find that these results do not significantly change as long as the sequence is long enough. For each sequence, all the active latents and 1000 randomly selected inactive latents are shown to the model. Consistent with our previous results, we find that the model identifies the active latents more accurately if their activations are higher, see Figure 2 first panel. In this panel we compare interpretations that were generated using the highest activating contexts and also interpretations that were generated by sampling from the different quantiles as described in the article above.

![correctly identified latents](/images/blog/generating-text-using-nl-to-simulate-activations/active_accuracy.png)
_Figure 2: Left panel: Accuracy on active latents as a function of their activation value. Latents with lower activation values are harder to identify. Right panel: Distribution of the alignment between predicted active latents and the ground truth active latents, compared with the distribution of the alignment between random latents and the ground truth active latents._

If we use the probability that the model assigns to the latent being active as a way to sort the latents, and select only as many latents as there are active ones in that specific sequence, we can compare the alignment between the decoder directions of the predicted latents and the ground truth latents. To do this we use the [Hungarian algorithm](https://en.wikipedia.org/wiki/Hungarian_algorithm) to find the optimal matching between these latents - this way, even if the model incorrectly identifies some of the latents, if they have decoder directions that are close to the ground truth active latents, we will pick up that signal. The left panel of Figure 2 shows the distributions of the alignment, as measured by the average cosine similarity between the decoder directions of the aligned predicted active latents and the ground truth active latents, for both the top most activating contexts and the quantile interpretations, compared with the distribution of the alignment between random latents and the ground truth active latents.

We investigated different ways to improve the fraction of correctly identified active latents: we used Llama 3.1 70b instead of Llama 3.1 8b, and found while Llama 70b was better at identifying the active latents, on average it still got less than 50\% of the active latents right. We also tried finetuning Llama 3.1 8b using as training that pairs of interpretations and the ground truth. We find that while finetuning significantly improved the accuracy on non-active latents, it decreased the accuracy on active latents. This is a research direction that we want to explore more in the future, because different mixtures of data might improve the results. We observed that using as "positive" examples only the top most activating ones increases the overall accuracy compared to using all examples, but we believe this to be an under-explored approach. Perhaps the most surprising result is that giving the model access to either the fuzzing score or the detection score improves the accuracy on the active latents the most, more than using the 70b model, although this decreases the accuracy on the non-active latents compared with the 70b model, but not when compared with the Llama 3.1 8b model. 

| Method | Active Latent Accuracy (Recall) | F1 | AUC  | 
|---------------|---------------------------|---------------------------|----------------|
| Quantile interpretations| 0.42 \(0.28-0.54\) | 0.24 \(0.18-0.29\) | 0.82 (0.78-0.86) | 
| Top interpretations | 0.34 \(0.22-0.44\) | 0.24 \(0.18-0.30\) | 0.80 (0.77-0.85) | 
| Finetuned Llama 3.1 8b | 0.28 \(0.18-0.36\) | 0.23 \(0.17-0.29\) | 0.82 (0.79-0.87) | 
| With Fuzzing Score | 0.58 \(0.44-0.71\) | 0.28 \(0.23-0.33\) | 0.84 (0.81-0.88) | 
| With Detection Score | 0.57 \(0.43-0.70\) | 0.28 \(0.23-0.33\) | 0.84 (0.81-0.88) | 
| Llama 3.1 70b | 0.44 \(0.29-0.56\) | 0.27 \(0.21-0.31\) | 0.84 (0.81-0.88) |  

_Table 1: Accuracy of the model to identify active latents, measured by recall, F1 and AUC. We show the average of the results over >1000 prompts and the 25-75\% inter-quartile range. Values are rounded to 2 decimal places._


To summarize these results we compute the KL divergence over 1000 prompts using these different techniques and compare them with that of using the full SAE, see Figure 3. We observe that there is a much larger spread of KL divergences when compared to the full SAE reconstruction, mainly due to the fact that even when the model identifies >50\% of active latents, if it incorrectly identifies some of the top active latents, the KL divergence is very high (>10 nats). To account for this we show the median KL divergence and the 25-75\% inter-quartile range. As expected, the methods with higher accuracy on active latents also have lower KL divergence. This is mainly due to the fact that we are doubly "cheating" - the model is only asked to identify active latents, not their activation values, and we don't ask the model to identify the non-active latents. Below we will discuss how the picture changes when we try these more difficult tasks. 

![correctly identified latents](/images/blog/generating-text-using-nl-to-simulate-activations/kl_divergences.png)
_Figure 3: KL divergence for different methods to identify active latents, with respect with to the model distribution. We show the median KL divergence and the 25-75\% inter-quartile range. Reconstruction refers to using the full SAE, quantiles to using the interpretations generated by sampling from the quantiles of the activation distribution and top to using the interpretations generated by using the top most activating contexts to identify the active latents. The finetuned model uses top interpretations, the fuzzing and detections scores use quantile interpretations, as does the Llama 3.1 70b model._

Due to the fact that, even on this easier task, we are not able to have a satisfactory performance, we believe that auto-interpretability techniques are still far from being able to be used to simulate the activations of the model, replacing the SAE with a natural language interface. Still, we think it could be argued that this experiment can give us some insights: looking at the active latents that were incorrectly identified as non-active, we could look for patterns, and understand if they are pathologically "uninterpretable" latents, or if their interpretation just needs to be improved. Similarly, we could try to narrow the interpretations of the non-active latents that were incorrectly identified as active.


# Predicting the activation value with latent interpretations.

The standard way people evaluate interpretations of latents is by doing [simulation](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html), where the model is tasked to predict the activation of the latent at different token positions and then the correlation between those values and the ground truth activations is used as a proxy for how good the interpretation is. We know from these results that models have a hard time predicting the activation values. Here we focus on the activation value at the last token of a sequence, and evaluate only the latents that are active. Once again this is "cheating", because we are only using latents that we know are active, but at this point we are just focusing on the feasibility of this simpler task rather than that of the real word scenario. 

In this task we ask the model to predict a number, from 0 to 9, indicating how strongly the selected token is active, given the interpretation of the latent. We explicitly tell the model that the latent is active, but we show an example where the value is 0, due to rounding down. We compare the number predicted by the model for each latent with 3 different distributions: its activation value, un-normalized, its activation bin in the activation distribution of that same latent and with the quantiles of the activation distribution. We find that the model predicts low and high values more frequently than the other values. We also use the "logprobs trick", where we compute the probability of each of the 10 values and compute the expected value of the prediction, and find that this distribution is very similar to the distribution of the predicted values. The distribution of the predicted values is shown in the left panel of Figure 4, and compared with the distribution of activation values used, and the distribution of the quantiles of the activation distribution the examples were generated from. We observe that higher predicted values correspond to higher activation values and activation quantiles, see right panel of Figure 4, but the spread is so large that it does not seem possible to reliably predict the activation value of a latent using the interpretations, see Table 2, where we show that correlation between the predicted value and the activation value of the latent and the activation quantile is low for all the methods.


![activation value prediction](/images/blog/generating-text-using-nl-to-simulate-activations/activation_prediction.png)
_Figure 4: Left panel: Distribution of the predicted activation values for each latent, compared with the activation value of the latent. Right panel: Average activation bin and average quantile of the predicted activation values for each predicted value, for the top interpretations, which have the highest correlation._

| Method | Correlation with activation bin | Correlation with distribution quantile |
|---------------|---------------------------|---------------------------|----------------|
| Quantile interpretations| 0.16 | 0.09 | 
| Top interpretations | 0.21 | 0.11 | 
| With Fuzzing Score | 0.18 | 0.08 | 
| With Detection Score | 0.17 | 0.08 | 
_Table 2: Pearson correlation between the predicted activation value and the activation value of the latent and the activation quantile of the latent._


# Correctly identifying non-active latents.


As we said above, current interpretations are better at identifying non-active latents than active latents. Still, because there are so many more non-active latents than active latents, the precision of current interpretations is low. To put things into perspective, around 50 latents are active at any given time for the chosen SAE, and because the SAE has 131k latents, identifying non-active latents correctly 90% of the time means that we incorrectly identify 13k latents as active, several orders of magnitude more than the number of active latents. This means that one needs to identify 99.9 to 99.99\% of non-active latents to not "overload" the reconstructions with false positives. If we threshold the probability at which we consider a latent to be active, such that we only consider the expected number of active latents to be 50, we find that most of the time no active latents are identified as active. In Figure 5 we show the predicted number of active latents as a function of the threshold. We find that the model is consistently overconfident and that even when selecting only the predictions where the model is more than 99.9\% confident, that would still correspond to about 170 active latents, more than 3 times the correct number, and that only 1% of the time a active latent in on that selection. If we set the threshold at 50%, 9000 latents are predicted to be active, and 40% of active features are correctly identified. This means that for this method to ever be successful the model has to be much more accurate at identifying non-active latents.

![threshold accuracy](/images/blog/generating-text-using-nl-to-simulate-activations/threshold_accuracy.png)
_Figure 5: Accuracy of the model to identify non-active and active latents as a function of the threshold. We also report, for this particular method, the expected number of active latents given that threshold. We observe a clear trade-off between the accuracy of non-active and active latents, and that the accuracy of active latents significantly drops as we consider less and less possible active latents. Error bars correspond to 95\% confidence intervals._

Finally, we find that the pre-computed scores - fuzzing and detection - are predictive of whether a latent is correctly identified as either active or non-active, see Figure 6, although the relation is not linear. This is a good verification that these simple scores are able to capture how good interpretations are, although the fact that these scores seem to saturate at both low and high values makes us think that there is still significant room for improvement.

![scores predictiveness](/images/blog/generating-text-using-nl-to-simulate-activations/scores_accuracy.png)
_Figure 6: Accuracy of the model to identify non-active and active latents as a function of the fuzzing and detection scores. The scores are rounded to 1 decimal place and averaged. Error bars correspond to 95\% confidence intervals._

