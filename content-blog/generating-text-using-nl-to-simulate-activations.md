---
title: "Generating text using natural language to simulate model's activations"
date: 2024-10-25T16:00:00-00:00
description: "Using interpretations of SAE latents to do inference on the host language model."
author: ["Gonçalo Paulo", "Nora Belrose"]
ShowToc: true
mathjax: true
draft: false
---

# Generating text by simulating the model using natural language

<iframe src="/images/blog/generating-text-using-nl-to-simulate-activations/activation_demo.html" style="width: 100%; height: 800px; border: none;"></iframe>

Our most recent work on using sparse autoencoders (SAEs) focused on automatically generating interpretations for their latents and evaluating how good they are. A potential draw of SAEs is that they could be used to "white-box" the language models they were trained on. If all the latents were interpretable one could track the latents active at any token and follow the internal machinery of the model at work. Not only that, we could potentially use the natural language interpretations to simulate the activations of the model, making all computations pass through a natural language interface. 

The interpretations that we generate for the latents of SAEs are done by focusing on patterns in activating contexts of the latents, as done in our most [recent paper](https://arxiv.org/abs/2410.13928). Even if our explanations were perfect, which they are not, SAEs still do not perfectly recover the activations of the model. In this work we are focusing on if we can recover the performance of the SAE, using the natural language interpretations, and not on recovering the performance of the model using SAEs. 

In this blog post we will explore how far away we are from this goal, by adressing 3 main problems:
1. Correctly identifying which latents are active.
2. Correctly simulating the value of the activations of the active latents.
3. Having a low rate of false positives.

**Key results**
- Our SAE latents interpretations can only identify less than 50\% of active latents in arbitrary contexts. This fraction can however be enough to generate "coherent" text, because most of the high active latents are correctly identified.
- (We still have not done this but should) We find that explanations can/can't be used to effectively simulate the value of the activations of the model when calibrated.
- Although explanations correctly identify 90% of non-active latents, a value closer to 99.9% is needed to generate "coherent" text if one requires that the model correctly labels all the latents.
- We find that the pre-generated scores for the interpretations are predictive of how frequently the model identifies the latents, and find that the scores can be used by the model to calibrate its predictions.


# How many latents are needed to recover the model behaviour?

Adding a single layer to the SAE model already drops the performance of the model significantly. [Gao et. al (2024)](https://arxiv.org/pdf/2406.04093) showed that the the increase in loss of patching their SAE - with 16 million latents - on GPT-4 is equivalent to having trained it on only 10\% of the total pretraining compute used. In this section we will consider the model with the SAE as the "skyline" performance, considering that we already are adding the SAE to the model and measuring how many of the active latents do we need to correctly identify to recover the performance of the model.

To measure this we collect the output of Gemma 2 9b, over a set of 100 sequences of 64 tokens. We use the base distribution at every token position except the first one - which is a \<bos\> token -  as the "ground truth" distribution for the KL divergence and compute the CE loss that we will use as the metric to recover. Then, for each of these sequences, we compute, for each token position, the activations of the SAE on the target layer - 11 in our case - and at every token position we reconstruct the activations. We then measure the KL divergence and the CE loss, which we consider the "full SAE" skyline. We then repeat this process using only a fixed number of active latents, which we choose either from the top active latents or we sample from the whole distribution. We then add "buffer" latents to the reconstruction, to keep the number of active latents the same as it would be without the intervention. The activations are sampled from the active latents that were not used in the reconstruction. Results are shown in Figure 1.

We observe that a big fraction of the CE loss can be recovered by using less than 50\% of the top active latents, but that one needs to get most of the latents right if one is randomly sampling the ones that are correctly identified. We don't have access to the loss curves of Gemma 2 9b, but we expect that the recovered CE is probably equivalent of having trained the model on significantly less compute. A better judge of model's performance would be to do this reconstruction as the model is being benchmarked, but we leave this for future work.


![performace recovered](/static/images/blog/generating-text-using-nl-to-simulate-activations/recovered_performance.png)
_Figure 1: KL divergence and CE loss for different fractions of correctly identified active latents. We compare the result of using always the most active latents vs sampling which active latents to use. Using the most active latents recovers the performance much faster, and the KL between the original model and the truncated sae is on the same order of magnitude as the full SAE using only 20\% of the latents. Having a relative error on the CE loss in the same order of magnitude is not a sufficient result, and the performace of the model is affected. Horizontal dashed line is the average number of active latents used in evaluation._

# Using latent interpretations to predict if they are active.

Knowing that it is possible to have a model "work" even when we don't correctly identify all the active latents, we focused on understanding how many of those latents could we correctly identify using our current interpretation approach. In our most [recent work](https://arxiv.org/pdf/2410.13928), we showed that our explanations, when used to identify between activating and non-activating examples, had a balanced accuracy of 76%, but that the accuracy could drop below 40% in the cases of less active latents. 

We tasked LLama 3.1 8b with predicting if a certain latent is active or not for the last token of a sequence, given the explanation of the latent. We take these sequences from RedPajama-V2, and we always select sequences with 32 tokens, although we find that these results do not significantly change as long as the sequence is long enough. For each sequence, all the active latents and 1000 randomly selected inactive latents are shown to the model. Coherently with our previous results, we find that the model identifies the active latents with more accuracy if they are their activations are higher, see Figure 2 first panel. In this panel we compare explanations that were generated using the top most activating contexts and also explanations that were generated by sampling from the different quantiles as described in the article above.

![correctly identified latents](/static/images/blog/generating-text-using-nl-to-simulate-activations/active_accuracy.png)
_Figure 2: Left panel: Accuracy on active latents as a function of their activation value. Latents with lower activation values are harder to identify. Right panel: Distribution of the aligment between predicted active latents and the ground truth active latents, compared with the distribution of the alignment between random latents and the ground truth active latents._

If we use the probability that the model assigns to the latent being active as a way to sort the latents that the model assigns as active and then select only as many as there are active latents in that specific sequence, we can compare the alignment between the predicted active latents decoder directions and the ground truth active latents decoder directions. To do this we use the [Hungarian algorithm](https://en.wikipedia.org/wiki/Hungarian_algorithm) to find the optimal alignment between these latents - this way, even if the model incorrectly identifies some of the latents, if they have decoder directions that are close to the ground truth active latents, we will pick up that signal. The left panel of Figure 2 shows the distributions of the aligment, as measured by the average cosine similarity between the decoder directions of the aligned predicted active latents and the ground truth active latents, for both the top most activating contexts and the quantile explanations, compared with the distribution of the alignment between random latents and the ground truth active latents.

We investigated different ways to improve the fraction of correctly identified active latents: we used Llama 70b instead of Llama 3.1 8b, and found while Llama 70b was better at identifying the active latents, on average it still got less than 50\% of the active latents right. We also tried finetuning Llama 3.1 8b using as training that pairs of explanations and the ground truth. We find that while finetuning singificantly improved the accuracy on non-active latents, it decreased the accuracy on active latents. This is a result that we still want to explore more in the future, because different mixtures of data might improve the results. We observed that using as "positive" examples only the top most activating increases the overall accuracy compared on using all examples, but believe this to be an underexplored approach. Perhaps the most surprising results is that giving the model access to either the fuzzing score or the detection score improves the accuracy on the active latents the most, more than using the 70b model, although this decreases the accuracy on the non-active latents compared with the 70b model, but not when compared with the Llama 3.1 8b model. 

| Method | Active Latent Accuracy (Recall) | F1 | AUC  | 
|---------------|---------------------------|---------------------------|----------------|
| Quantile explanations| 0.42 \(0.28-0.54\) | 0.24 \(0.18-0.29\) | 0.82 (0.78-0.86) | 
| Top explanations | 0.34 \(0.22-0.44\) | 0.24 \(0.18-0.30\) | 0.80 (0.77-0.85) | 
| Finetuned Llama 3.1 8b | 0.28 \(0.18-0.36\) | 0.23 \(0.17-0.29\) | 0.82 (0.79-0.87) | 
| With Fuzzing Score | 0.58 \(0.44-0.71\) | 0.28 \(0.23-0.33\) | 0.84 (0.81-0.88) | 
| With Detection Score | 0.57 \(0.43-0.70\) | 0.28 \(0.23-0.33\) | 0.84 (0.81-0.88) | 
| Llama 3.1 70b | 0.44 \(0.29-0.56\) | 0.27 \(0.21-0.31\) | 0.84 (0.81-0.88) |  

_Table 1: Accuracy of the model to identify active latents, measured by recall, F1 and AUC. We show the average of the results over >1000 prompts and the 25-75\% interquartile range. Values are rounded to 2 decimal places._


To summarize this results we compute the KL divergence over 1000 prompts using these different techniques and compare them with that of using the full SAE, see Figure 3. We observe that there is a much larger spread of KL divergence with the all explanation methods when compared with the full SAE reconstruction, mainly due to the fact that even when the model identifies >50\% of active latents, if it incorrectly identifies some of the top active latents, the KL divergence is much larger (>10). To account for this we show the median KL divergence and the 25-75\% interquartile range. As expected, the methods with higher accuracy on active latents also have lower KL divergence. This is mainly due to the fact that we are doubly "cheating" - the model is only asked to identify active latents, and not their activation values, and we don't ask the model to identify the non-active latents. Below we will discuss how the picture changes when we try these more difficult tasks. 

![correctly identified latents](/static/images/blog/generating-text-using-nl-to-simulate-activations/kl_divergences.png)
_Figure 3: KL divergence for different methods to identify active latents, with respect with to the model distribution. We show the median KL divergence and the 25-75\% interquartile range. Reconstruction refers to using the full SAE, quantiles to using the explanations generated by sampling from the quantiles of the activation distribution and top to using the explanations generated by using the top most activating contexts to identify the active latents. The finetuned model uses top explanations, the fuzzing and detections scores use quantile explanations, as does the Llama 3.1 70b model._

Due to the fact that, even at this easier task, we are not able to have a satisfactory performance, we believe that auto-interpretability techniques are still far away from being able to be used to simulate the activations of the model, substituting the SAE with a natural language interface. Still, we think it could be argued that this experiment already can give us some insights: looking at the active latents that were incorrecly identified as non-active, we could look for patterns, and understand if they are patologically "uninterpretable" latents, or if their explanation just needs to be improved. Similarly, looking at the non-active latents that were incorrectly identified as active, something that we will discuss in following sections, we could look try to narrow their interpretations, such that they would be more specific to the contexts where they actually are active.


# Predicting the activation value with latent interpretations.

The standard way people evaluate interpretations of latents is by doing [simulation](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html), where the moder is tasked to predict the activations of the latent at different token positions and then the correlation between those values and the ground truth activations is used as a proxy for how good the interpretation is. We know from these results that models have a hard time predicting the activation values. Here we focus on the activation value at the last token of a sequence, and evaluate only the latents that are active. Once again this is "cheating", because we are only using latents that we know are active, but at this point we are just focusing on the feasability of this simpler task than that of the real word scenario. 

In this task we ask the model to give a number, from 0 to 9, on how much the select token is activated, given the explanation of the latent. We explicitly tell the model that the latent is active, but we show an example where the value is 0, due to rounding down. We compare the number predicted by the model for each latent with 3 different distributions: it's activation value, unormalized, it's activation bin in the activation distribution of that same latent and with the quantiles of the activation distribution. We find that the model predicts 0,8,9 and with more frequency than the other values. We also use the "logprobs trick", where we compute the probability of each of the 10 values and compute the expected value of the prediction, and find that this distribution is very similar to the distribution of the predicted values. Despite the fact that the model predicts high activation values more frequently that it should, we find that there is a monotonic relationship between the predicted binarized value and the average activation value of the bin, with higher predicted values corresponding to higher activation values.  The same is true for the average quantile and average activation bin, but the spread is so large that it does not seem possible to reliably predict the activation value of a latent using the explanations. 


![activation value prediction](/images/blog/generating-text-using-nl-to-simulate-activations/image-4.png)



# Correctly identifying non-active latents.


As we have been discribing the, current explanations are better at identifying non-active latents than active latents. Still, because there are so many more non-active latents than active latents, their precision is low. To put things into perpective, around 50 latents are active at any given time for the chosen SAE, and because the SAE has 131k latents, identifying non-active latents curretly 90% of the time means that we it incorrectly identifies 13k latents as active, several orders of magnitude more than the number of active latents. This means that one needs to identify 99.9 to 99.99\% of non-active latents to not "overload" the reconstructions with false positives. If we calibrate the probability, such that we only consider the expected number of active latents to be 50, we find most of the times, we would not be able to identify any active latents, for instance, the expected number of active latents with assigned probability greater than 90% is 9, while the expected number of non active latents with proability greater than 90\% is 3600. If the calibration probability is 99.94\%, the 49.6 times out of the 50 times, a non-active latent would be identified as active, while only 0.3 times out of the 50 times, an active latent would be identified as active.

**Should there be a figure here or something else?**


Finally, we find that the pre-computed scores - fuzzing and detection - are predictive of whether a latent is correctly identified as either active or non-active, see Figure 6, although the relation is not linear. This is a good verifications that these simple scores are able to capture how good explanations are, although the fact that these scores seems to saturate at both low and high values makes us think that there is still significant room for improvement.


![scores predictiveness](/static/images/blog/generating-text-using-nl-to-simulate-activations/scores_accuracy.png)
_Figure 6: Accuracy of the model to identify non-active and active latents as a function of the fuzzing and detection scores. The scores are rounded to 1 decimal place and averaged. Error bars correspond to 95\% confidence intervals._

