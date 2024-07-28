
## Background
Understanding the internal mechanisms of LLM is an important and hard task to solve. There is a vast literature on interpreting activations of neurons, both in language models([Gandelsman et al.](https://arxiv.org/pdf/2406.04341) 2024, [Gurnee et al.](https://arxiv.org/pdf/2305.01610) 2024) and vision models ([Olah et al.](https://distill.pub/2020/circuits/zoom-in/) 2020), and while there is promising work in this direction, sparse auto-encoders (SAEs) have been presented as a more interpretable lens through which to look at LLMs activations ([Cunningham et al.](https://arxiv.org/pdf/2309.08600) 2023). 

SAEs encode the activations of specific parts of an LLM and convert these activations into few, sparse, features. The intuition is that  that SAEs can help disentangle the [polysemanticity found in neurons](https://transformer-circuits.pub/2022/toy_model/index.html) and that these features would be easy to interpret as a human. Recent work as shown that it is possible to scale SAEs to larger  LLMs ([Templeton et al.](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html#scaling-to-sonnet/%20,) 2024, [Gao et al.](https://arxiv.org/pdf/2406.04093) 2024), and that it is possible to have models generate explanations for those features([Bricken et al.](https://transformer-circuits.pub/2023/monosemantic-features/) 2023), which had already been show to work with neurons ([Bills et al.](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html#sec-algorithm-explain) 2023). 

Being able to massively tag and sort features could have significant implications one the way we use LLMs, as it has been shown that these features can be used to steer [their behaviour](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html).

Sparse autoencoders recover a diversity of interpretable, monosemantic features, but present an intractable problem of scale to human labelers. We investigate different techniques for generating and scoring arbitrary text explanations of SAE features, and release a open source library to allow people to do research on auto-interpreted features.


## Key Findings

- Open source models generate and evaluate text explanations of SAE features reasonably well, albeit somewhat worse than closed models like Claude 3.5 Sonnet.

- Explanations found by LLMs are similar to explanations found by humans.

- Auto-interpreting all 1.5M features of GPT2 with the current pipeline would cost 1300$ by using API calls to Llama 3.1 and around the same price running the explanations locally in a quantized model. Using Claude 3.5 Sonnet to generate and score explanations would cost c.a. 8500$.

- Code can be found at <https://github.com/EleutherAI/sae-auto-interp>

- We built a small dashbord to explore explanations and their scores: https://demo-7m2z.onrender.com/




## Generating Explanations

Sparse autoencoders decompose activations into a sum of sparse feature directions, which hopefully are more interpretable than the activations of neurons of the MLP.

We are interested in automatically generating explanations that are able to predict the degree of activation of a feature in any given piece of text. To do this, we first collect example texts that activate the feature, then prompt an LLM to find the pattern that links these texts based on the activating tokens and the surrounding context. 

In prior work, [Bills et al.](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html#sec-algorithm-explain) (2023) prompt language models with contexts that activate an MLP neuron. Each token is followed by a tab, then the associated neuron activation, then a newline.

We instead highlight max activating tokens in each example with a set of <\<delimiters>>. Optionally, we choose a threshold for which tokens are highlighted. This helps the model distinguish important information for some densely activating features.

    Example 1:  and he was <<over the moon>> to find

    Example 2:  we'll be laughing <<till the cows come home>>! Pro

    Example 3:  thought Scotland was boring, but really there's more <<than meets the eye>>! I'd

We experiment with several methods for augmenting the explanation.

**Chain of thought** improves general reasoning capabilities in LLMs. We might expect that it would enable explanations that capture richer relations between tokens and context. To add CoT, we few-shot prompt the model with several examples of a thought process which captures a human approach to generating explanations.

    Step 1: List a couple activating and contextual tokens you find interesting. 
    Search for patterns in these tokens, if there are any. Don't list more than 5 tokens. 

    Step 2: Write down general shared features of the text examples.

    Step 3: List the tokens that the neuron boosts in the next token prediction 

    Step 4: Write an explanation


**Activations** help distinguish which sentences might be more representative of a feature. We provide the magnitude of activations for highlighted portions of each sentence following each example.

We compute the **logit weights** for each feature through the path expansion WUWD\[f] where WU  is the model unembed and WD\[f] is the decoder direction for a specific feature. This is equivalent to the [logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) (nostalgebraist 2020), but we plan to look at the [tuned lens](https://arxiv.org/abs/2303.08112) equivalent (Belrose et al. 2023) soon. Including the vocab items whose logits are most promoted by a feature according to the logit lens allows the model to consider the feature’s casual effects on the output.


## Scoring explanations 

Text explanations represent interpretable “concepts” in natural language. But how do we evaluate the faithfulness of explanations to the concepts actually contained in SAE features?

We view the explanation as a _classifier_ which predicts whether a feature will fire on a token given a context: if a human or a language model is given this explanation, can they predict the activation pattern of the feature? An explanation should enable prediction with high recall, identifying most activating examples, as well as high precision, distinguishing between activating examples and non-activating ones. 

Consider a feature which activates on the word “stop” after “don’t” or “won’t.” There are two failure modes:

1. The explanation could be **too broad,** identifying the feature as activating on the word “stop”. It would have high recall on held out text examples, but low precision.

2. The explanation could be **too narrow**, stating the feature activates on the word “stop” only after “don’t”. This would have high precision, but low recall.

One approach to score explanations is “simulation scoring” [Bills et al.](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html#sec-algorithm-explain) (2023) which uses a language model to assign an activation to each token in a text given an explanation, then measures the correlation between predicted and real activations. The method is biased toward recall: because adversarial examples for prompts are nontrivial to find, explanations that are broad can can achieve a similar scores to the ones that are more precise, e.g, because most activatign examples will have the token “stop”, an explanation that completely ignores the context can achieve high simulation score. Additionally, simulation is expensive, requiring a prediction for each token. Because SAE features activate very sparsely it is rare in any given context many tokens will have activations greater than 0 and it could be argued that this kind of approach is wasteful for SAEs.

We experiment with different methods for evaluating the precision and recall of SAE features.

**Detection** 

Detection scoring is a coarse approximation of simulation scoring. A language model is presented with the explanation of a feature and a set of contexts which we have activated or not the feature. The model is then used as a binary classifier, deciding if each context activates or not the feature. By classifying whole texts instead of tokens, we can cheaply, scalably evaluate the recall of an explanation.

We draw five activating examples from deciles of the activation distribution and twenty from non-activating examples. Evaluating a large proportion of examples provides a more accurate estimate of recall. 


**Fuzzing** 

We investigate fuzzing, a closer approximation to simulation than detection. Again, we treat the language model as a binary classifier. This time, activating tokens are <<delimited>> in each example. We prompt the language model to identify which examples are correctly marked. Like fuzzing from automated software testing, this method captures specific vulnerabilities in an explanation. Evaluating an explanation on both detection and fuzzing can identify whether a model is classifying examples for the correct reason. 

We draw seven activating examples from deciles of the activation distribution. For each decile, we mark five correctly and two incorrectly for a total of seventy examples. To “incorrectly” mark an example, we choose N non activating tokens to delimit where N is the average number of marked tokens across all examples. Not only are detection and fuzzing scalable to many examples, but they’re also easier for models to understand. Less capable – but faster – models can provide reliable scores for explanations.

Future work might explore more principled ways of creating fuzzed examples. For example, one could replace activating tokens with non activating synonyms or fill in surrounding context with a masked language model.

**Neighbors**

The above methods face similar issues to simulation scoring: they are biased toward recall, and counterexamples sampled at random are a weak signal for precision. As we scale SAEs and features become sparser and more specific, the inadequacy of recall becomes more severe [Gao et al.] 

Motivated by the phenomenon of feature splitting, we use “similar” features to test whether explanations are precise enough to distinguish between similar contexts. We focus on using cosine similarity between the decoding vectors of features to find counterexamples for an explanation. Other ways to find neighbors, like using embeddings of the natural language explanations of features to find similarly sounding explanations, should be explored in future work. 


**Generation**

We provide a language model an explanation and ask it to generate sequences that contain the feature. Explanations are scored by the number of activating examples a model can generate. Generation scoring is more sensitive to explanations providing the correct context as broad explanations are unlikely to create the correct activation context. 


## Results

We conduct most of our experiments using detection and fuzzing as a point of comparison, with Llama 3 70b as the scorer and explainer except where explicitly mentioned. These scorers are inexpensive and scalable while still providing a clear picture of feature patterns and quality. Our vision of an auto interpretability pipeline is one that uses cheap and scalable methods to map out relevant or interesting features and which then is supplemented by more expensive detailed techniques. One could start with self-interpreted features ([Chen et al 2024](https://arxiv.org/abs/2403.10949), [Ghandeharioun et al.](https://arxiv.org/abs/2401.06102) 2024), find where these features disagree significantly with the one generated by a comprehensive pipeline like ours, and use auto-interpretability agents ([Rott Shaham et al.](https://arxiv.org/abs/2404.14394) 2024) to investigate the differences and hone down on the true explanation of features. 

Llama 3 70b is used as an explainer and scorer except where explicitly mentioned. 

## Explainers

### How does base model performance affect explanation quality? 

We evaluate model scale and human performance on explanation quality on the 132k latent GPT-2 Top-K SAEs. Models generate explanations for 350 features while a human (Gonçalo) only evaluates thirty five. Manual labeling is less scalable and wider error bars reflect this fact. As a comparison, we show the performance of a scorer that is given a random explanation for the features.


![](https://lh7-us.googleusercontent.com/docsz/AD_4nXfJn2xvRRirl4AeZnfnv5tBwckjI5jZ70O7q_IwZSJAZ_XB116_7KsG-UlyHJ5GxH9diuGa1ugK-sQf17dCpZJLbff5el5Ys7R2eOiSyjLVDlCL08clMlvMOWTuvnErgzpkwZX2pHTnOX-EjyhASGnkWpLD?key=5hGzhgAbyv361OYwubzqdA)


_Figure 1:_ (left, middle) The first two figures depict explanation quality versus the test example’s activation decile. Q10 is closest to the maximum activation, while Q1 is the lowest. Weakly activating features have lower scores as these examples activate on tokens that may not be linked to the explanation. (right) Better formed explanations have higher balanced accuracy on both fuzzing and detection. Balanced accuracy accounts for the imbalance between the number of non-activating examples (20) and the activating examples (50). 



### Providing more information to the explainer

A human trying to interpret a feature on Neuronpedia might incorporate various statistics before providing an explanation. We experiment with giving the explainer different information to understand whether this improves performance.

![](https://lh7-us.googleusercontent.com/docsz/AD_4nXdHmfBFLGWX7WVoZG1zZbTH-05d58gGVjpSW4MSo4QjRu74sgBOX4L6nRegwPX8kWafau2oFzBubeWNrdzSoenUF8l6diP0BbB0fwo-JffttVq9UVeOwdOKLCnaAYjizgRB6vfLNriqPpupmslgOXTz5S_w?key=5hGzhgAbyv361OYwubzqdA)

_Figure 2: (left) Chain of thought causes models to overthink and focus on extraneous information, leading to vague explanations. (middle) Performance levels out on detection. (right) GPT-2 SAEs are presented as squares and Llama 7b SAEs as diamonds. LLama-3 8b SAE explanations perform worse._ 

Providing more information to the explainer does not significantly improve scores for both GPT-2 (squares) and LLama-3 8b (diamonds) SAEs. Instead, models tend to overthink and focus on extraneous information, leading to vague explanations. This could be due to the quantization and model scale. We plan on investigating this in future work.


### Giving the explainer different samples of top activating examples

Bricken et al. use forty nine examples from different quantiles of the activation distribution for generating explanations. We analyze how varying the number of examples and sampling from different portions of the top activations affects explanation quality.

- **Top activating examples:** The top ten, twenty, or forty examples 

- **Sampling from top examples:** Twenty or forty examples sampled from the top 200 examples 

- **Sampling from all examples:** Ten, twenty, or forty examples sampled randomly from all examples

- **A mixture:** Twenty examples from the top 200 plus twenty examples sampled from all examples

![](https://lh7-us.googleusercontent.com/docsz/AD_4nXdPqqKQKkCs03UTTvU-3rsSMB1VEk7qyu3Qq555noPeNgM63ZVPXecM7Q8UO-dMMeU-nXM_WATQDm2jTcfRMZDiXRij6e0oHqEnE4_jpThApvYpsj3DY9fjay72hvRCIUVlqUN4iNggVzRio4gzhZG3OHTe?key=5hGzhgAbyv361OYwubzqdA)

_Figure 1:  (left) GPT-2 explanations generated from just the top activations perform worse than sampling from the whole distribution. (middle) We see a similar trend in fuzzing with GPT-2 explanations. (right) GPT-2 SAEs are presented as squares and Llama 7b SAEs as diamonds. Again, Llama-3 8b SAE explanations perform worse._ 

Sampling from the top N examples produces narrow explanations that don’t capture behavior across the whole distribution. Instead, sampling evenly from all features produces explanations that are robust to less activating examples. This effect is less prominent in LLama 3 SAEs.


### Visualizing activation distributions

We can visualize explanation quality by scoring a wider set of examples [anthropic]. In the figures below, we evaluate 1,000 examples with fuzzing and detection. We compare explanations generated from the whole distribution (left column) versus explanations generated from the top N examples (right column). Explanations “generalize” better when the model is presented with a wider array of examples.

![](https://lh7-us.googleusercontent.com/docsz/AD_4nXenSQFlDcxG2oAN6fb5agNerHGbHYml-cNtdtS3_U03L-_CIfy7O8lAvMBgqbJoRq7cMff3jLH62O-oHwufFzPDLWrhtXN4Os3fIBGQKm7n9xKgEFmtssMfMiApb5ZWN234dQ3nw5aABgbN0qjEQqR5AGia?key=5hGzhgAbyv361OYwubzqdA)

_Figure 3:  For these plots, the 1,000 examples are binned in 20 activation intervals, and in the conditional plots we represent the fraction of the 4 boolean possibilities corresponding to the combination of fuzzing and detection scoring. We selected randomly from the top 50 features of layers 0, 1, and 2. Specifically, the features are feature 14 of layer 0, feature 6 of layer 2 and feature 24 of layer 2._


## Scorers

### How do methods correlate with simulation?

The average balanced accuracy on detection and fuzzing correlates with the simulation scoring proposed by Bills et al. (Pearson correlation of 0.61). We do not view simulation scoring as a “ground-truth” score, but we feel that this comparison is an important sanity check since we expect our proposed methods to correlate reasonably with simulation.

![](https://lh7-us.googleusercontent.com/docsz/AD_4nXc7YUf7uOMEuWJApy3-658bWvrNUIMl_VjioTmEgJD6OjbWJMxPtOgmNN-yR7CgLe8us_cF3_74vgVmft2qyqoCtVlmlK1zgM1s1x0l-0aX2zDDU5ZWQoguhi66hooo2JJrqpjZ8-iPB6VFw1W5fvoBuMKw?key=5hGzhgAbyv361OYwubzqdA)

_Figure 5: We plot the correlation between the balanced accuracy of our metrics and the simulation scores._


### How does scorer model size affect scores? 

We see that both detection and fuzzing scoring are affected by the size of the evaluator model, even when given the same explanation. Still we observe that the scores given by the bigger models correlate with the scores given by the smaller models, and some kind of “calibration” curve could be estimated.

![](https://lh7-us.googleusercontent.com/docsz/AD_4nXcUJUyz3Fe_ckzOb3-LSB_IG_HEcS0UcvCHnZjSKgDxkHwJgOrhlglHmaGupxBHSTSdOeWYA1yzUAY17m5m6wrmEDGhfpF4C6Xw815XV7kjcPuD70kXf0GGiJMpg_Vvg-vsKv22t6pFuJG4GjeFDq8hCtWL?key=5hGzhgAbyv361OYwubzqdA)

![](https://lh7-us.googleusercontent.com/docsz/AD_4nXf3W8IkoCaBSt5QOz0BxCoDXcWsrohqC4N9gJgHjhm6nHfK59op7DVopFftF4xzSR9bC5pMo71-xjA_Nk1yTRULHkLS8Ur81ckvaBMfL3e4tq0cTSUk604XqAqxHio1T-FP2ir_3mSIXPfTJISHN_SEN7fV?key=5hGzhgAbyv361OYwubzqdA)

![](https://lh7-us.googleusercontent.com/docsz/AD_4nXf1lVrF5aJNUXAr8dX_676KdNP3Ssa5tUUJpOsGGpkyKdjHqyHk7a0W2JbW0Brkgp8OCNH2mLNHUbgKTxadJoxackHUMvDDpDRGgpfn9nZqzPAg5VEUVTasQj88KtC-NGsNvVrHfP5q6um8vLNdwH2EwsNh?key=5hGzhgAbyv361OYwubzqdA)


### How much more scalable is detection/fuzzing? 

|                   |               |                      |               |                      |
| ----------------- | ------------- | -------------------- | ------------- | -------------------- |
| Method            | Prompt Tokens | Unique Prompt Tokens | Output Tokens | Runtime in seconds   |
| Explanation       | 397           | 566.45 ± 26.18       | 29.90 ± 7.27  | 3.14 ± 0.48          |
| Detection/Fuzzing | 725           | 53.13 ± 10.53        | 11.99 ± 0.13  | 4.29 ± 0.14          |
| Simulation        | –             | 24074.85 ± 71.45     | 1598.1 ± 74.9 | 73.9063 ± 13.5540 \* |

We measure token I/O and runtime for explanation and scoring. Tests are run on a single NVIDIA RTX A6000 on a quantized Llama-3 70b with VLLM prefix caching. Simulation scoring is slower as we used [Outlines’s](https://outlines-dev.github.io/outlines/reference/serve/vllm/) (a structured generation backend) logit processing to enforce valid JSON responses. 

|                   |               |               |                                   |                                         |
| ----------------- | ------------- | ------------- | --------------------------------- | --------------------------------------- |
| Method            | Prompt Tokens | Output Tokens | GPT 4o mini(per million features) | Claude 3.5 Sonnet(per million features) |
| Explanation       | 963.45        | 29.90         | 160 $                             | 3400 $                                  |
| Detection/Fuzzing | 778.13        | 11.99         | 125 $                             | 2540 $                                  |
| Simulation        | 24074.85      | 1598.1        | 4700 $                            | 96K $                                   |

**How precise is scoring without adversarial examples?**

 ********

We find that using adversarial examples significantly lowers the balanced accuracy of the features we score. Explanations generated by Llama 70B are not precise enough to distinguish between similar features.  Another reason for which this score is worse than without adversarial examples could be due to not filtering for co-ocurance between neighboring examples, which we plan to do in the future. It is possible that similar features co-occurr ([Brussman et al. 2024](https://www.alignmentforum.org/posts/baJyjpktzmcmRfosq/stitching-saes-of-different-sizes)), and so finding examples that can tell them apart might require more sophisticated techniques.

As the distance between features increases, the explanations are good enough to distinguish between very similar features. We believe that it is possible to design iterative processes that would distinguish between very similar features, and believe that it is a promising line of research.

****![](https://lh7-us.googleusercontent.com/docsz/AD_4nXcgKR5SRSllX5vOu2MyXLUCQGGo_2H4uywgPOhY1mAJto-UzH_YM84waNaK_KC5rrHUCqrqp_XuQII1bWq62n9w6PcLTRqT_2wgPJTD70UMw37ylYsy2UOIbsh_KRPCv2cY3OhlOJZ_2qm2-Qi27OQCGoHw?key=5hGzhgAbyv361OYwubzqdA)****

_Figure 4: (Left) Balanced detection accuracy when showing examples from neighboring features as non activating examples. The balanced accuracy drops from >80% to close to random, showing that the explanations generated are not specific enough to distinguish very similar contexts. (Right) As the distance between the features increases the accuracy increases again._  

Another way to measure the precision of explanations is to use generation scoring. In generation scoring we prompt the model to generate 10 examples that would activate a feature given a certain explanation. We find that a significant fraction of explanations do not generate sentences that activate the corresponding features-. This could be due to the quality of the generating model or due to the fact that the explanations miss critical context that does not allow the models to correctly generate activating contexts. In future work we will explore how generation score can be used to identify context dependent features which have explanations that are too broad, and will measure the effect of model size in generation scoring.

![](https://lh7-us.googleusercontent.com/docsz/AD_4nXf6k0naPU1N9cWKOA66O9h_vQ4Kz_GifIONNVraaKraJmvicMHPWZhlM8By4J78wJdvwqz92TqRwmVOFKY7piVayG1G36aC4Pz3avXwvu00_QuxoOAvvia3BzvtTphP-E4Ux-smsirf5Kt1Af4RqzLDoIho?key=5hGzhgAbyv361OYwubzqdA)


## Filtering with known heuristics

Automated interpretability pipelines might involve a preprocessing step that filters out features for which there are known heuristics. We demonstrate a couple simple methods for filtering out context independent unigram features and positional features. 


### Positional Features

Some neurons activate on absolute position rather than on specific tokens or context. We cache activation frequencies for each feature over the entire context length of GPT2 and filter for features with high mutual information with position ([Voita et. al](https://arxiv.org/pdf/2309.04827) 2023).

![](https://lh7-us.googleusercontent.com/docsz/AD_4nXfuJtkPnW6-I2n5tc_bAsrT-VlDApaSC-QL_8sNuJDk9xgQq-FueZqZ_YfihiUzl7pNF-6aarmE7UNXExAvluO9P7v0nB9Lp_sxlqfD4eULi1H54Fhoc5CNKStzDOkVHvs5U5ux69MTvJ0I4P7NcmglbLf0?key=5hGzhgAbyv361OYwubzqdA)

Similar to Voita et al. 2023, we find that earlier layers have a higher number of positional features, but that these features represent a small fraction (<0.1%) of all features of any given layer. 

![](https://lh7-us.googleusercontent.com/docsz/AD_4nXe5X1gnKsJane59UZk1EioobKl90YmYDO0GKNf6TnV3fGkuqMns_Q2X2VMZQuNq3YhRhHI6fGSPvdvZQ0rgmw0FIErmtM7-viznZwNVzC_38QZ68Jrmtm6vjV48FTA5Z3cns43dJquCc-jc0ue62XAYUbMN?key=5hGzhgAbyv361OYwubzqdA)

_Figure N: Number of positional features by layer in GPT-2. Layer 0 indicates the SAE trained on the residual stream after layer 0. The colors represent thresholds of mutual information. Voita et al. select features with I(act, pos) > 0.05._


### Unigram features

Some features activate on tokens independent of the surrounding context. We filter for features which have twenty or fewer unique tokens among the top eighty percent of their activations. To verify that these features are context independent, we create sentences with 19 tokens randomly sampled from the vocabulary plus a token that activates the feature. 

We do this twice per token in the unique set, generating upwards of forty scrambled examples per feature. We run batch through the autoencoder and measure the fraction of scrambled sentences with nonzero activations.

![](https://lh7-us.googleusercontent.com/docsz/AD_4nXcrkGa40OwPxGdDWXYvOuJB4WoE0weK6JbZSSU6Eclk0C9Ne6uVPFUJfkDWjrO6b8dXUh2I2W8aObL2fK6YedxlgufeG5ZmFaPAsm-lcwrKhTM1i2xXFHQ7mIyjew4tbhGXfiC1MzRDgQg11pEt-M26tK76?key=5hGzhgAbyv361OYwubzqdA)

_Figure N: Fraction of context independent features in odd layers of GPT-2 (0-indexed). Layer 0 indicates the SAE trained on the residual stream after layer 0. The scale indicates a threshold for the scrambled sentences. For example, the yellow line marks features for which > 90% of scrambled sentences still activate._

We analyze a random sample of 1k features from odd layers in GPT-2. Earlier layers have a substantial portion of context independent features. 

Some features also activate following specific tokens. Instead of saving features with twenty or fewer activating tokens, we search for features with <= twenty unique prior tokens. This process only yields a handful of features in our sample.


## Sparse Feature Circuits

![](https://lh7-us.googleusercontent.com/docsz/AD_4nXdH_zTzgsAXmi3BOhAdfsJfNx6HGMTVFyyQVO08qecXuqy0ndisKuMAwHeFZZc66UyDd_QMGfTGvb5hu4B7FpLpvfCz5zsGaOvQZk-l5GahmT8W5UUkk8CbqP84vdnzOgQqtzzJ0RCpd8xoZSFxLtDklrUW?key=5hGzhgAbyv361OYwubzqdA)

We demonstrate our automated interpretability pipeline by explaining and scoring all features in the Bias in Bios classifier task from the Sparse Feature Circuits paper ([Samuel Marks](https://arxiv.org/abs/2403.19647) et al 2024). We CoT prompt LLama-3 70b to generate an explanation given a feature’s top logits and activations (above 70% the max activation). Explanations are scored with detection and fuzzing. A full dashboard is available [here](https://demo-7m2z.onrender.com/).

**Some features cannot be explained from their activations**

Toward the end of the circuit \[[L4\_12420](https://demo-7m2z.onrender.com/pythia/resid_4-12420.html), [ATTN3\_2959](https://demo-7m2z.onrender.com/pythia/attn_3-2959.html)], features activate on dense sets of unrelated tokens. Note LLama’s initial confusion at providing an explanation. 

|                                                                                                                                                                                                                                                                                     |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ACTIVATING TOKENS: Various function words and punctuation marks.PREVIOUS TOKENS: No interesting patterns.Step 1.- The activating tokens are mostly function words (prepositions, conjunctions, auxiliary verbs) and punctuation marks.- The previous tokens have nothing in common. |

However, the top logits provide some signal. Llama picks up on this and correctly revises its explanation to include the information. 

|                                                                                                                                                                                            |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| (Part 2)SIMILAR TOKENS: \[' her', ' she', ' herself', ' She', ' hers', 'she', ' Her', 'She', 'Her', 'her'].- The top logits list suggests a focus on pronouns related to a female subject. |

This example highlights a distinction between features that activate on inputs and features that promote certain outputs. 

This example underscores a broader issue with current explainer and scorer methods. While some features seem to be related just to the types of tokens they activate on, other features are better explained by the tokens they promote. Because don’t have a method that correctly scores these features their explanations will have low scores even if they have good explanations for the causal effect of the feature.

Future work might investigate ways to automatically filter for causal features at scale.


## Future Directions

**More work on scoring**

Generation scoring seems promising. Some variations we didn’t try include: 

- Asking a model to generate counterexamples for classification. This is hard as models aren’t great at consistently generating negations or sequences that \*almost\* contain a concept. 

- Using a BERT model to find sentences with similar embeddings or perform masked language modeling on various parts of the context, similar to fuzzing. \[<https://arxiv.org/abs/1907.11932>]

Scoring based on the downstream effects, or even just based on logits seems important as discussed throught the post. Some features are better explained not by their activations but by which tokens they promote, and current scoring techniques don’t do a better job at ranking them. 

**Human evaluation of generated examples**

[Neuronpedia](https://www.neuronpedia.org/) is set to upload the GPT-2 SAEs we have looked at. We will be uploading our explanations and scores such that people can red-team and evaluate the explanations provided by our auto-interp pipeline. For now we have a small dashboard which allows people to explore explanations and their scores 

<https://demo-7m2z.onrender.com/>.

### Formal Grammars for Autointerp

Perhaps automated interpretability with language models is too unreliable. With a bunch of known heuristics for SAE features, maybe we can generate a domain specific language for explanations, and use in-context learning or finetuning to generate explanations using that grammar, which could potentially be used by an external verifier. 

    <explanation> ::= “Activates on ” <subject> [“ in the context of ” <context>]

    <subject> ::= <is-plural> | <token>

    <is-plural> ::= “the tokens ” | “the token ”

    <token> ::= (* a generated token or set of related tokens *)

    <context> ::= (* etc. *)

The (loose) grammar above defines explanations like: “Activates on the token pizza in the context of crust”.


### Debate

Debate is a suggested approach to supervising superhuman AI systems. 

Our debaters are presented an identical twenty examples from the top 100 max activating examples of a given feature, shuffled for each debater. Each debater has access to a scratchpad and a quote tool. Thoughts in the scratchpad are hidden from the judge which is instructed to only accept verified quotes. After a bout of reasoning, the debaters present an opening argument consisting of three direct, verified quotes and an explanation sampled at high temperature. 

The “arguments” and explanations from N debaters are passed to a weaker judge model without access to chain of thought or the original text. The judge chooses the top explanation from presented arguments. We would carefully monitor argument length and order to remove biases in judging.

**More information to generate explanations.**

To generate better and more precise explanations we may add more information to the context of the explaining model, like results of the effects of ablation, correlated tokens and other information that humans use to try to come up with new explanations.

**Acknowledgements** 

We would like to thank Joseph Bloom, Sam Marks, Can Rager, Jannik Brinkmann and Sam Marks for their comments and suggestions, and to Neel Nanda, Sarah Schwettmann and Jacob Steinhardt for their discussion.

**Contributions**

Caden Juang wrote most of the code and devised the methods and framework. Caden did the experiments related to feature sorting and Sparse Feature Circuits. Gonçalo Paulo ran the experiments and analysis related to explanation and scoring, including hand labeling a set of random features. Caden and Gonçalo created the write up. Nora Belrose supervised, reviewed the manuscript and trained the Llama 3 8b SAEs. Jacob Drori designed many of the prompts and initial ideas. Sam Marks suggested the framing for causal/correlational features in the SFC section.

