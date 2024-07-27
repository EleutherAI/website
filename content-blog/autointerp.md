
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
  


## Generating Explanations

Sparse autoencoders decompose activations into a sum of sparse feature directions. We are interested in automatically generating explanations that are able to predict the degree of activation of a feature in any given piece of text. To do this, we first collect example texts that activate the feature, then prompt an LLM to find the pattern that links these texts based on the activating tokens and the surrounding context. 

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

One approach to score explanations is “simulation scoring” \[Bills et al.] which uses a language model to assign an activation to each token in a text given an explanation, then measures the correlation between predicted and real activations. The method is biased toward recall: because adversarial examples for prompts are nontrivial to find, explanations that are broad can can achieve a similar scores to the ones that are more precise, e.g, because most activatign examples will have the token “stop”, an explanation that completely ignores the context can achieve high simulation score. Additionally, simulation is expensive, requiring a prediction for each token. Because SAE features activate very sparsely it is rare in any given context many tokens will have activations greater than 0 and it could be argued that this kind of approach is wasteful for SAEs.

We experiment with different methods for evaluating the precision and recall of SAE features.

**Detection** 

Detection scoring applies the language model as a binary classifier over a set of texts, provided an explanation. By classifying whole texts instead of tokens, we can cheaply, scalably evaluate the recall of an explanation.

We draw five activating examples from each of ten quantiles of the activation distribution and twenty from non-activating examples. Evaluating a large proportion of examples provides a more accurate estimate of recall in a highly scalable manner. 

**Fuzzing** 

We investigate fuzzing, a closer approximation to simulation than detection. Again, we treat the language model as a binary classifier. This time, activating tokens are <\<delimited>> in each example. We prompt the language model to identify which examples are correctly marked. Like fuzzing from automated software testing, this method captures specific vulnerabilities in an explanation. Evaluating an explanation on both detection and fuzzing can identify whether a model is classifying examples for the correct reason. 

Five correctly marked examples are drawn from ten quantiles of the activation distribution along with two additional examples which are incorrectly marked for a total of seventy examples. Not only are detection and fuzzing scalable to many examples, but they’re also easier for models to understand. Less capable – but faster – models can provide reliable scores for explanations.

**Neighbors**

The above methods face similar issues to simulation scoring: they are biased toward recall, and counterexamples sampled at random are a weak signal for precision. As we scale SAEs and features become sparser and more specific, the inadequacy of recall becomes more severe \[Gao et al.] 

Motivated by the phenomenon of feature splitting, we use “similar” features to better test whether explanations are precise enough to distinguish between similar contexts. We focus on using cosine similarity between the decoding vectors of features to find counterexamples for an explanation. Other ways to find neighbors, like using embeddings of the natural language explanations of features to find similarly sounding explanations, should be explored in future work. 

**Generation**

We provide a language model an explanation and ask it to generate sequences that contain the feature. Explanations are scored by the number of activating examples a model can generate. 

However, generation has its shortcomings: 

- To reliably generate counterexamples, one might ask a language model to write sentences which _don’t_ contain, or _almost_ contain an idea. However, models struggle to avoid mentioning a topic if explicitly stated in the prompt ([Castricato et al.](https://arxiv.org/abs/2402.07896) 2024).

* Generation might miss specific modes of a feature’s activating contexts. Consider the broad explanation for “stop”. A generator might only write counterexamples that contain “don’t” but miss occurrences of “stop” after “won’t”. 

* Each feature requires a generation pass to create examples and a scoring pass to count activations. This is hard to scale. 


## Results

We conduct most of our experiments using detection and fuzzing as a point of comparison, with Llama 3 70b as the scorer and explainer except where explicitly mentioned. These scorers are inexpensive and scalable while still providing a clear picture of feature patterns and quality. Our vision of an auto interpretability pipeline is one that uses cheap and scalable methods to map out relevant or interesting features and which then is supplemented by more expensive detailed techniques. One could start with self-interpreted features ([Chen et al 2024](https://arxiv.org/abs/2403.10949), [Ghandeharioun et al.](https://arxiv.org/abs/2401.06102) 2024), find where these features disagree significantly with the one generated by a comprehensive pipeline like ours, and use auto-interpretability agents ([Rott Shaham et al.](https://arxiv.org/abs/2404.14394) 2024) to investigate the differences and hone down on the true explanation of features. Our proposed method, and most proposed up until this point, miss some types of features that can’t be explained by just looking at the top activating examples, but scalability allows us to estimate the number of features that the current pipeline fails to explain.


## Explainers

### How does base model performance affect explanation quality? 

We evaluate model scale and human performance on explanation quality on the 132k latent GPT-2 Top-K SAEs. Models generate explanations for 350 features while a human (Gonçalo) only evaluates thirty five. Manual labeling is less scalable and wider error bars reflect this fact. As a comparison, we show the performance of a scorer that is given a random explanation for the features.


![](https://lh7-us.googleusercontent.com/docsz/AD_4nXe1eNCKC7DRZhEUBcNrtOAugobXvSsrP0BbiL8AklKwrcQI5SuRBGxsqtBBP7tKj7ZTkIrPHIRJzdiy0os0pUJNYlC8BoYVs_fsDDUo52qVvreCl58kSWZ9jcc4KqLSEY3SADfFjPZA_SOuab51UHTCgILP?key=5hGzhgAbyv361OYwubzqdA)

_Figure 1:_ The first two figures depict explanation quality versus the test example’s activation quantile. Q10 is closest to the maximum activation, while Q1 is the lowest. Weakly activating features have lower scores, as these examples activate on tokens that may not be linked to the explanation. Understanding if there is a significant difference between Top-K SAEs, which have a fixed number of activating features per example, and other types of SAEs is left for future work. The third figure shows the balanced accuracy of detection vs the balanced accuracy of fuzzing, where better formed explanations would be in the top right. Balanced accuracy takes into account the imbalance between the non-activating examples (20) and the activating examples (50). 



### Providing more information to the explainer

A human trying to interpret a feature on Neuronpedia might incorporate various statistics before providing an explanation. We experiment with giving the explainer different information to understand whether this improves performance.

![](https://lh7-us.googleusercontent.com/docsz/AD_4nXdHmfBFLGWX7WVoZG1zZbTH-05d58gGVjpSW4MSo4QjRu74sgBOX4L6nRegwPX8kWafau2oFzBubeWNrdzSoenUF8l6diP0BbB0fwo-JffttVq9UVeOwdOKLCnaAYjizgRB6vfLNriqPpupmslgOXTz5S_w?key=5hGzhgAbyv361OYwubzqdA)

_Figure 2: (left) Chain of thought causes models to overthink and focus on extraneous information, leading to vague explanations. (middle) Performance levels out on detection. (right) GPT-2 SAEs are presented as squares and Llama 7b SAEs as diamonds. LLama-3 8b SAE explanations perform worse._ 

Providing more information to the explainer does not significantly improve scores for both GPT-2 (squares) and LLama-3 8b (diamonds) SAEs. We note that CoT leads to worse performance, because models tend to overthink, making their explanations overly broad and vague when given too much information, or focused too much on extraneous information. This could be due to the quantization and model scale. We plan on investigating this in future work.

For both GPT-2 SAEs and Llama 7b SAEs, more information does not significantly improve scores. We’d like to note that our scoring method does not require predicting any logit information, so it is possible that that short coming influences the fact that adding that information does not increase the scores generated. Having human evaluators, or more precise scores might desambiguate that. 


### Giving the explainer different samples of top activating examples

Bricken et al. use forty nine examples from different quantiles of the activation distribution for generating explanations. We analyze how varying the number of examples and sampling from different portions of the top activations affects explanation quality.

- **Top activating examples:** The top ten, twenty, or forty examples 

- **Sampling from top examples:** Twenty or forty examples sampled from the top 200 examples 

- **Sampling from all examples:** Ten, twenty, or forty examples sampled randomly from all examples

- **A mixture:** Twenty examples from the top 200 plus twenty examples sampled from all examples

![](https://lh7-us.googleusercontent.com/docsz/AD_4nXdPqqKQKkCs03UTTvU-3rsSMB1VEk7qyu3Qq555noPeNgM63ZVPXecM7Q8UO-dMMeU-nXM_WATQDm2jTcfRMZDiXRij6e0oHqEnE4_jpThApvYpsj3DY9fjay72hvRCIUVlqUN4iNggVzRio4gzhZG3OHTe?key=5hGzhgAbyv361OYwubzqdA)

_Figure 1: (left) GPT-2 explanations generated from just the top activations perform worse than sampling from the whole distribution. (middle) We see a similar trend in fuzzing with GPT-2 explanations. (right) GPT-2 SAEs are presented as squares and Llama 7b SAEs as diamonds. Again, Llama-3 8b SAE explanations perform worse._ 

Sampling from the top N examples produces narrow explanations that don’t capture behavior across the whole distribution. Instead, sampling evenly from all features produces explanations that are robust to less activating examples. This effect is less prominent in LLama 3 SAEs.


### Visualizing activation distributions

We can visualize explanation quality by scoring a wider set of examples \[anthropic]. In the figures below, we evaluate 1,000 examples with fuzzing and detection. We compare explanations generated from the whole distribution (left column) versus explanations generated from the top N examples (right column). Explanations “generalize” better when the model is presented with a wider array of examples.

![](https://lh7-us.googleusercontent.com/docsz/AD_4nXenSQFlDcxG2oAN6fb5agNerHGbHYml-cNtdtS3_U03L-_CIfy7O8lAvMBgqbJoRq7cMff3jLH62O-oHwufFzPDLWrhtXN4Os3fIBGQKm7n9xKgEFmtssMfMiApb5ZWN234dQ3nw5aABgbN0qjEQqR5AGia?key=5hGzhgAbyv361OYwubzqdA)

_Figure 3: For these plots, the 1,000 examples are binned in 20 activation intervals, and in the conditional plots we represent the fraction of the 4 boolean possibilities corresponding to the combination of fuzzing and detection scoring. We selected randomly from the top 50 features of layers 0, 1, and 2. Specifically, the features are feature 14 of layer 0, feature 6 of layer 2 and feature 24 of layer 2._


## Scorers

**How precise is fuzzing and detection without adversarial examples?**

 ********

We find that using adversarial examples significantly lowers the balanced accuracy of the features we score, as the explanations generated by Llama 70B are not precise enough to distinguish between very similar features. Because similar features are frequently co-occuring ([Brussman et al. 2024](https://www.alignmentforum.org/posts/baJyjpktzmcmRfosq/stitching-saes-of-different-sizes)) , finding examples that can tell them apart might require more sophisticated techniques.

As the distance between features increases, the explanations are good enough to distinguish between very similar features. We believe that it is possible to design iterative processes that would distinguish between very similar features, and believe that it is a promising line of research. 

****![](https://lh7-us.googleusercontent.com/docsz/AD_4nXebOx8Th95G9mnncZQhphzgbQhi5mJRB4wWpcpU4-5B5gzfoAqQN76PTrgXK64u5Yr5PJNW3z0_fgq9Ll9LvqM-Iul4BjxVCuCeJ7rlvLtXZ1HZSZY-v6cYF3FiVRZX52BtvLIdh0VLrmjYZuer2DEZbDKk?key=5hGzhgAbyv361OYwubzqdA)****

We find that a significant fraction of explanations do not generate sentences that activate the corresponding features. This could be due to the quality of the generating model or due to the fact that the explanations miss critical context that does not allow the models to correctly generate activating contexts. In future work we will explore how generation score can be used to identify context dependent features which have explanations that are to broad.

![](https://lh7-us.googleusercontent.com/docsz/AD_4nXewHmphWwOJRVIp8zWwdULT9Ljgut03_kPLi7LkXVVqXPkwaounySKVMDli_t1ACWkvCLv4wYq8jUab1aBnFR4SvGe5yY4P8EejGHuJx19UoH7RM4LJcCPSPKcE27T2ed1fvXS8AC0h1JNtVJ_xylwoKHv1?key=5hGzhgAbyv361OYwubzqdA)


### How does base model performance affect scores? 

We see that both detection and fuzzing scoring are affected by the size of the evaluator model, even when given the same explanation. Still we observe that the scores given by the bigger models correlate with the scores given by the smaller models, and some kind of “calibration” curve could be estimated.

![](https://lh7-us.googleusercontent.com/docsz/AD_4nXdg1IZG1FGpjUCHIIlTG2jlzMeso-c8vpPjmypsIiIEy4EbFV7tQTN9Ifz9AzTno67ZsBWDjcOFGRuues2QENXmCLA-x69KbMmtfh6CNgkc6Gg6sRt9MmTWoK1poa0JhoKBLFwpfscqQXkv9Qdsq_gcMt_A?key=5hGzhgAbyv361OYwubzqdA)

![](https://lh7-us.googleusercontent.com/docsz/AD_4nXeV91_mzKlFanebdvp3aRbttYZns4t9URh03RTmfscXhNBiom9EfqVE71xXt4Zil_BF4PIDxl7PBiACCiw90WlXI1gZ1X_A6SjD8WAny8d3s3af7NCBif153h3CcTw6rlvwPvo_k8bqwkoevO03Nhn2QH8P?key=5hGzhgAbyv361OYwubzqdA)

![](https://lh7-us.googleusercontent.com/docsz/AD_4nXeTajnjQC4FZJQQCbcqCMEqTlSIzs0W77tNtdMQDGbTLlGuc6WlGwm0V-DQMs5_JTaogQw5v6IdjoizMC8sBa_jrA9hJGd9382CjW3QDvFgIX7pURCsyrloo94qHF8uxqFUSuUgmoZ2_3GkhRns_A-ksz8?key=5hGzhgAbyv361OYwubzqdA)


### How do methods correlate with simulation?

We find that the average detection and fuzzing balanced accuracy for each example correlates with simulation scoring proposed by Bills et al (Pearson correlation of 0.61). We don’t take simulation scoring as the “ground-truth” score, but feel that this kind of comparison is an important sanity check, since we would expect that our proposed methods should correlate reasonably strongly with simulation.

![](https://lh7-us.googleusercontent.com/docsz/AD_4nXeM5xoPZ38jo5J_4m-mz2jGcMlqiQkZD03vOhpu-Qqju9hukQCPGx-tp_IJ7u-O2wJyUv1ouZaWNdA1v7R_UxIm4jWgYFFWL33lev-5Dk51Z34VS7EH3MiD-2lWHPKgDdpVdwt5xKPNbGN86k_niqi5d0qY?key=5hGzhgAbyv361OYwubzqdA)


### How much more scalable is classification? 

|                |               |                      |               |                      |
| -------------- | ------------- | -------------------- | ------------- | -------------------- |
| Method         | Prompt Tokens | Unique Prompt Tokens | Output Tokens | Runtime in seconds   |
| Explanation    | 397           | 566.45 ± 26.18       | 29.90 ± 7.27  | 3.14 ± 0.48          |
| Classification | 725           | 53.13 ± 10.53        | 11.99 ± 0.13  | 4.29 ± 0.14          |
| Simulation     | –             | 24074.85 ± 71.45     | 1598.1 ± 74.9 | 73.9063 ± 13.5540 \* |

We measure token I/O and runtime for explanation and scoring. Tests are run on a single NVIDIA RTX A6000 on a quantized Llama-3 70b with VLLM prefix caching. Simulation scoring is slower as we used [Outlines’s](https://outlines-dev.github.io/outlines/reference/serve/vllm/) (a structured generation backend) logit processing to enforce valid JSON responses. 

|                |               |               |                                   |                                         |
| -------------- | ------------- | ------------- | --------------------------------- | --------------------------------------- |
| Method         | Prompt Tokens | Output Tokens | GPT 4o mini(per million features) | Claude 3.5 Sonnet(per million features) |
| Explanation    | 963.45        | 29.90         | 160 $                             | 3400 $                                  |
| Classification | 778.13        | 11.99         | 125 $                             | 2540 $                                  |
| Simulation     | 24074.85      | 1598.1        | 4700 $                            | 96K $                                   |


## Sorting with known heuristics

Automated interpretability pipelines might involve a preprocessing step that filters out features for which there are known heuristics. We demonstrate a couple simple methods for filtering out context independent unigram features and positional features. 


### Positional Features

Some neurons activate as a function of absolute position rather than on specific tokens or context. We cache activation frequencies for each feature over the entire context length of GPT2 and filter for features with high mutual information with position I(act, pos) > 0.05 ([Voita et. al](https://arxiv.org/pdf/2309.04827) 2023).

![](https://lh7-us.googleusercontent.com/docsz/AD_4nXemOR6Vz0zv-CjBGKsuNcDc2026QGP2p7X0553X_r8TJY3990D8s0BvrAZF1ujdnUaHa_pHU5H2cAxh16g86R8i8IzBmGM7QMQzjQ5PPJwJGW13mYe3wOJkh73DeObwIkKvBm8mVhtPj0HRqb7_yZJXHRq4?key=5hGzhgAbyv361OYwubzqdA)

We find that layer layers have a higher number of positional features that earlier layers, but that these features represent a small fraction (<0.1%) of all features of any given layer.  

![](https://lh7-us.googleusercontent.com/docsz/AD_4nXcv7jOigxgnDBTMzzF1Mfpmwm6dWPS9YnrC6jc-q9lQ5kmdfGwbfOfyDh4V0bZBciX5C4EnJPkPOxkPi0wpVLc-cY0yJRdzJAl1QufonwdfX8DwwV8Ts4uIhNpNxr9Fw6PZVrEeJYOy2ur-j8nke84slWJQ?key=5hGzhgAbyv361OYwubzqdA)


### Unigram features

Some features activate on tokens independent of the surrounding context. We filter for features which have twenty or fewer unique tokens among the top eighty percent of their activations. To verify that these features are context independent, we insert the activating tokens into random contexts and verify the features still activate. 

![](https://lh7-us.googleusercontent.com/docsz/AD_4nXfDOYLTzkCb_y7opO45RTrAe0EeMWbLvcJD8ys8GsuyywWJS7Tuxa5ygeU92ZBuJ3h0w04NzDZ5_O3YCzZ7dIoYaabGomaSfqAUrgoXAUn79VgZ-Iqlw8yp9flWimAI-DtShC3CJAMwxVXhJWK6M-Z2_wbk?key=5hGzhgAbyv361OYwubzqdA)

For each token, we 


## Sparse Feature Circuits

![](https://lh7-us.googleusercontent.com/docsz/AD_4nXflbDfu3WHNnudsiONYZda6RKMeuHpyaKvsaz5EcRa2eZM5nEWaTDHSDgkouMb5iQnvy_LpnRlHSk0-2b1ImEv6dz1mlKLShP_spyxphtLYRnnUMvIdIPxgo5ixB436h3LQ8a6RDCN9ByN3tErjjtG5Azr0?key=5hGzhgAbyv361OYwubzqdA)

We demonstrate our automatic interpretability pipeline by labeling and scoring all features in the Bias in Bios classifier task from the Sparse Feature Circuits paper ([Samuel Marks](https://arxiv.org/abs/2403.19647) et al 2024).

<https://demo-7m2z.onrender.com/>


## Future Directions

**Human evaluation of generated examples**

[Neuronpedia](https://www.neuronpedia.org/) is set to upload the GPT-2 SAEs we have looked at. We will be uploading our explanations and scores such that people can red-team and evaluate the explanations provided by our auto-interp pipeline.

**More work on generation scoring**

Generation scoring seems promising. Some variations we didn’t try include: 

- Asking a model to generate counterexamples for classification. This is hard as models aren’t great at consistently generating negations or sequences that \*almost\* contain a concept. 

- Using a BERT model to find sentences with similar embeddings or perform masked language modeling on various parts of the context, similar to fuzzing. \[<https://arxiv.org/abs/1907.11932>]


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

We would like to thank Joseph Bloom, Can Rager and Jannik Brinkmann for their comments and suggestions, and to Neel Nanda, Sarah Schwettmann and Jacob Steinhardt for their discussion.

**Contributions**

Caden Juang wrote most of the code and designed the framework. Caden did the experiments related to feature sorting and Sparse Feature Circuits. Gonçalo Paulo ran the experiments related to explanation generation and scoring. Caden and Gonçalo wrote the write up. Nora Belrose supervised and reviewed the manuscript. Jacob Drori designed many of the prompts and initial ideas. 


## Appendix

Non-cherry picked examples from the first 50 features of different layers of GPT2 131k latents per layer. We plan on substituting this section with examples from Neuronpedia such that people can red-team and evaluate the explanations provided by our auto-interp pipeline.

Suffixes \\"-emic\\", \\"-emia\\", and \\"-enza\\" in medical and scientific contexts, often related to diseases, epidemics, or academic fields. - Detection 0.83 Fuzzing 0.9

![](https://lh7-us.googleusercontent.com/docsz/AD_4nXc5JhtOr__C_xg1JeZFgz1DVd9IH35WnNNUJYx7ukC9ubq3qvrPVtKgTKK74HmT4yYThRo-i_7-TOqYiH-TVPYtInhY1t9d5MfS802DKPlXl5B9ZjBeq-GOj_Sypk8-_lTP4daXUaq9DJfc2EtdPqOiMYY?key=5hGzhgAbyv361OYwubzqdA)

The word \\"other\\" or \\"others\\" in various contexts, often indicating a contrast or addition to something previously mentioned. Detection 0.96 Fuzzing 0.76

![](https://lh7-us.googleusercontent.com/docsz/AD_4nXeCvR3OL2KCr1-EsOpwyey9TfNuguFd0LbUoWvU6YfWVgLeYyiqPDBsuJ-I55kEnGFL8fQclofxJ1hKViAOJHwWBmf-d33cB19cY4Y47ajWKQ-uT49OyIsdLQcRb-s5B3LlLDPBPw_e_V2lVlaTzFdh5Esw?key=5hGzhgAbyv361OYwubzqdA)

Punctuation marks (<< >>, :, etc.) and suffixes (-s, -es) in text, often indicating a reference to a separate section, FAQ, or resource. Detection 0.76 Fuzzing 0.64

![](https://lh7-us.googleusercontent.com/docsz/AD_4nXc0NfehoTGR72yIxfrdbjvVWd4hZ5QSJA5rWhxLPcAPP40KYLervcR1vYQmKEl3OTKj1Mk5pCwkbfejiHCwdVtp2Sfz7PL72_askHpdX55HFVEA-LiGgnHkPsuGW7HzywD3RreE6irzIP6lL2AC5iCE-jqd?key=5hGzhgAbyv361OYwubzqdA)

Punctuation marks (e.g., ',', '.', '->', '::', '@'), conjunctions (e.g., 'or', 'to'), and prepositions (e.g., 'in', 'to', 'of') in a variety of contexts, including URLs, email addresses, and formal writing. Detection 0.67 Fuzzing 0.5

![](https://lh7-us.googleusercontent.com/docsz/AD_4nXdyqrIl3l5J-2Xic57U_QHZazPuwyawU2wYq5xiM5MNnzaZ8GcZUajkcg_ToUFPessV7AWgSNFSrk6gQQ68m4cuznrKMqfvJiiTUctbQiD8hUCaXIOIxPrT5dDq9Ft14QnRi9CoQFNf6jQ4d-ed0nB6QIEW?key=5hGzhgAbyv361OYwubzqdA)

Part of a word, usually a prefix or suffix, in a proper noun or a technical term. Detection 0.74 Fuzzing 0.5

![](https://lh7-us.googleusercontent.com/docsz/AD_4nXd6g1UKp7Ogd9Yaa4PJbjq2QenO5m-pdSUMm8mwrG23jnb19p3vjkKcGYjrkBqPjHMXIX0CXP18NVlwIIWfBCQfTkB2Hhancg_V8R46CFE2SSL4K2mP7dqeH2pwdsh7D3zZuW4a0M5Yax_tzA2BoqzIW35r?key=5hGzhgAbyv361OYwubzqdA)

Adjectives and adverbs related to being busy, crowded, or active, often in the context of places or situations. Detection 0.67 Fuzzing 0.6

![](https://lh7-us.googleusercontent.com/docsz/AD_4nXcm5r3qT1TRWyGQOvHwmszgx8XHiXv0wbGXkh7iWaWvB2TR2QP697q1uuSraQ39Ucy1yXbsXXEKxMNnjrVtJUuH0dCZnf5n-x3dGlUFCfKddyhaX6KejJk80fTWqM-Xtw8rVq4zzGB0IgjTVPcXGuTd0y0h?key=5hGzhgAbyv361OYwubzqdA)

Possessive pronouns, articles, and common function words (e.g., \\"you\\", \\"a\\", \\"the\\", \\"or\\", \\"up\\", \\"in\\"), often appearing in specific grammatical contexts. - Detection 0.44 Fuzzing 0.5

![](https://lh7-us.googleusercontent.com/docsz/AD_4nXdxfsAmL8nMiKcTpllxAsEs72YOsJHpGcA-sXas-ibqsWVU2q9ZFc5HuVwksxkzPPnRVp-8rpWj4XiZP3m_WlnG3f9rKfyxaIbCk0w6vc9i9KTn7skslYA5JDCQhTXQP942JzkK2ftzYALHiDtDLXVjsGQV?key=5hGzhgAbyv361OYwubzqdA)
