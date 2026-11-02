---
title: "Investigating the Stability of Natural Language Autoencoders"
categories: ["Research Notes"]
author: ["Kaylee Vo", "Daniel Yoo", "Abhinav Chand", "Gonçalo Paulo"]
description: "Testing how robust Natural Language Autoencoder verbalizations are to changes in training seed, paraphrasing, and targeted text edits. Work done during SOAR 2026, by Kaylee, Daniel and Abhinav, with author order randomized. "
date: 2026-09-22
ShowToc: true
mathjax: true
draft: true
cover:
    image: /images/blog/nla-stability/training-splits-tree.png
    caption: "The four NLAs trained for the seed-stability experiments differ only in LoRA initialization, RL data order, or GRPO rollout stochasticity."
    relative: True
---

## Introduction

[Natural Language Autoencoders (NLAs)](https://transformer-circuits.pub/2026/nla/) encode model activations into natural-language explanations, then reconstruct the activations from those explanations. They are constituted by an activation verbalizer (AV), which produces the an explanation text, and an activation reconstructor (AR), which maps it back to an activation vector. This language bottleneck offers a way to inspect model representations and intervene on them by editing text.

We measure the NLA reconstruction of the activations using the fraction of variance explained (FVE): how much reconstruction reduces squared error relative to predicting the mean activation. Higher FVE means better activation reconstruction, but it is possible that the explanation's readable content does not describe what the activation encodes. We separately measure passage support: whether statements in the explanation are supported by the source passage. Passage support is narrower than faithfulness to an activation, since an activation may encode predictions or other information not stated in the passage. There is no easy way to have a ground truth for faithfulness as we don't actually know what the activations represent.

We investigate three questions: Do explanations transfer between independently trained NLAs? Does better reconstruction correspond to greater passage support? Can editing explanations reliably redirect model outputs? Our training experiments use LoRA-based [EasyNLA](https://github.com/asherps/EasyNLA) models, and our Gemma-3-12B-IT experiments use either the released NLAs ([AV](https://huggingface.co/kitft/nla-gemma3-12b-L32-av), [AR](https://huggingface.co/kitft/nla-gemma3-12b-L32-ar)) or our custom EasyNLA checkpoints, as specified in each subsection.

We find that:

- [Explanations transfer across training runs](#seed-stability) with a smaller reconstruction loss increase than paraphrasing causes. A probe trained on embeddings of text can distinguish these runs, and one human evaluator distinguishes them above chance, though less accurately than the probes.
- [Greedy Coordinate Gradient (GCG) improves reconstruction](#optimizing-reconstruction-beyond-readable-meaning) while making explanations less readable and without improving passage support. This demonstrates that just reconstruction does not garantee faithful explanations.
- After joint reinforcement learning, [AV explanations reconstruct better than Sonnet explanations](#passage-support) despite having much lower passage support.
- [Paraphrasing adds little task-performance degradation](#robustness-to-paraphrasing) beyond NLA reconstruction on the tested simple tasks. However, some tasks remain solvable even with mismatched activations at the patched position, limiting what robustness establishes.
- [Structured explanation edits](#which-edits-need-to-be-done-to-change-the-answer) can redirect outputs toward supplied target answers, including [answers associated with a different task](#using-explanation-edits-to-change-the-task).

## Results

### Training Interventions

#### Seed stability

*Experimental setup: [Seed stability setup](#seed-stability-setup), [NLA identification with logistic regression methodology](#nla-identification-with-logistic-regression-methodology), [NLA discrimination by human test](#nla-discrimination-by-human-test).*

If NLAs are to be used to explain activations of models, we would like these explanations to be robust, not only to the sampling of the verbalizer model but also the training procedure of the NLAs. If independently trained NLAs can exchange verbalizations with little reconstruction loss, this argues against codes specific to an individual AV–AR pair. Still, it does not exclude nonsemantic conventions shared across runs with the same base model and training procedure, or establish that the readable explanation is faithful to the activation.

We train four LoRA-based EasyNLA AV–AR pairs on Qwen2.5-7B-Instruct activations at layer 20 (zero-indexed decoder-block output). The cross-reconstruction, probe, and human-discrimination tests all use these same four NLAs. We find that NLAs trained with different data seeds and initialization can share the reconstructed text, and that while the FVE decreases, it does not decrease more than simple paraphrasing, see left panel of [Figure 1](#figure-1). We note that the diagonal FVE scores are higher than the off-diagonal terms, which is expected as these corresponding pairs representing the diagonals were trained jointly to minimize the reconstruction loss as they come from the same NLAs. At the same time, we note that the off-diagonal scores are close to the diagonal scores, and in fact, they are much better than the FVE with paraphrased verbalizations.

On the other hand, this text is clearly distinguishable when using LLM encodings, as can be seen by the right panel of [Figure 1](#figure-1), which illustrates that logistic regression probes trained on the embeddings of the original LLM can differentiate any pair of NLAs with a high accuracy. We also find that a human can distinguish between the text generated by two seeds, although at a much lower rate than the logistic regression probes. An author of this work performed 42 trials of matching text generated by the different NLAs, see the [human-test methodology](#nla-discrimination-by-human-test), and was able to correctly pair them 55% of the time, above random chance at 33% of the time. 

{{<figure id="figure-1" src="/images/blog/nla-stability/swap-matrix-identification.png" width="100%" align="center" caption="Figure 1. Results of cross swapping AV-AR modules, and the logistic regression identification of NLAs. Both panels use Qwen2.5-7B-Instruct, layer 20 activations and the same 127 held-out prompts. Left panel: AV_i's explanation scored by AR_j. The last column is for comparison with paraphrasing. Right panel: 5-fold CV pooled accuracy on logistic regressions for pairs of NLAs."/>}}

#### Optimizing reconstruction beyond readable meaning

*Experimental setup: [Optimizing reconstructed text to improve activation reconstruction](#optimizing-reconstructed-text-to-improve-activation-reconstruction).*

We use the released layer-32 NLA for Gemma-3-12B-IT ([AV](https://huggingface.co/kitft/nla-gemma3-12b-L32-av), [AR](https://huggingface.co/kitft/nla-gemma3-12b-L32-ar)) to test whether optimizing explanations for reconstruction also improves passage support or readability. GCG substantially improves reconstruction over the original AV explanations. The original AV explanations achieve an FVE of 72.1% ± 0.5%. Without the NLL penalty against AV policy (β=0), GCG increases FVE to 80.7% ± 0.5%. Adding a small NLL penalty (β=0.01) gives the highest reconstruction, with an FVE of 81.4%, see panel (a) of [Figure 2](#figure-2). The optimized text is much less readable and much less likely for the model under study.

The original AV explanations have a mean NLL of 3.84 nats, while explanations optimized with β=0.01 have a mean NLL of 4.63 nats, and when β=0, mean NLL increases further to 5.58. GCG explanations also have lower passage support (14.3% supported compared to 17.5%, using the [passage-support judging prompt](#prompt-for-judging-passage-support) in the Appendix). Mean edit counts were 4.36, 14.36, and 16.41 for β=0.1, 0.01, and 0, respectively. As a control, for each example and β, we made the same number of edits but on fully random positions and tokens.

{{<figure id="figure-2" src="/images/blog/nla-stability/gcg-reconstruction-optimization.png" width="100%" align="center" caption="Figure 2. Panel (a): reconstruction FVE for GCG edits and edit-count-matched random substitutions at each NLL-penalty weight β; the dotted line is the unedited AV baseline. Panels (b–d): the input passage and selected token, original AV explanation, and explanation after GCG with β=0.01. Highlighting marks changed tokens; the example improves FVE from 0.744 to 0.908."/>}}

These results show that token substitutions can improve reconstruction without improving readable meaning or passage support. This is consistent with nonsemantic encoding or exploitation of the reconstructor, but does not establish that the original AV explanations use steganography. Nor does the gap between AV and GCG reconstruction quantify how much current training exploits such mechanisms.

We also investigate warm starting with three EasyNLA training runs on Qwen2.5-7B-Instruct at layer 20, comparing SFT initialization with cold-start RL with and without KL regularization. We find that doing RL without warm starting leads to a lower but similar final FVE when training the model with EasyNLA, see [Skipping Warm-Start](#skipping-warm-start) in the Appendix. On the other hand, these explanations are slightly less natural looking.

### Passage support

*Experimental setup: [Prompt for judging passage support](#prompt-for-judging-passage-support).*

We test whether explanations with greater passage support reconstruct better. The NLA objective does not directly require statements in an explanation to be supported by the source passage. Our judge measures that support, rather than whether the explanation faithfully describes the activation itself. We evaluate an additional 2,000 FineFineWeb examples that were not used to train the NLA. For each example, we use a Sonnet explanation generated using the same procedure used to create the NLA warm-start data.

We compare the AV explanation and the Sonnet explanation on the final checkpoint after RL and the warm-start SFT checkpoint. We pass each explanation through the AR and measure reconstruction quality using fraction of variance explained (FVE), following the NLA evaluation setup. We use our [custom EasyNLA for Gemma-3-12B-IT at layer 32](https://huggingface.co/achand45/gemma-3-12b-it-nla-L32) because we do not have access to the warm-start checkpoint and do not know what examples were used to train the [official NLA](https://huggingface.co/kitft/nla-gemma3-12b-L32-ar).

{{<figure id="figure-3" src="/images/blog/nla-stability/passage-support-combined.png" alt="Left: reconstruction FVE for AV and Sonnet explanations after RL and at warm-start. Right: source passage and corresponding AV and Sonnet explanations with support highlights." caption="Figure 3. **Left:** Reconstruction FVE for two explanation sources at two training stages. From left to right: the post-RL AR reconstructs explanations from its paired AV (70.06%) or from Sonnet (53.83%); the warm-start SFT AR reconstructs explanations from its paired AV (48.22%) or from Sonnet (60.79%). Blue bars use AV-generated explanations; red bars use Sonnet-generated explanations. Higher bars mean better activation reconstruction, not greater passage support. **Right:** A source passage ending at the selected token, “such” (top), and AV and Sonnet explanations of the same example (bottom), with reconstruction FVE of 0.801 and −0.162. Green and red highlights mark text presented as grounded and confabulated, respectively."/>}}

The relationship between passage support and reconstruction differs between checkpoints. At the warm-start SFT checkpoint, Sonnet explanations reconstruct better than AV explanations, with an FVE of 60.8% compared with 48.2%.

We classified each sentence in the AV and Sonnet explanations using GPT-5.6 Luna (full prompt in [Prompt for judging passage support](#prompt-for-judging-passage-support)). **73.1% of Sonnet sentences are supported by the passage, compared with 6.3% of AV sentences.** Unsupported sentences account for 26.6% and 93.3%, respectively; the remaining 0.3% and 0.4% are illegible.

Note that the AV explanations contain next-token predictions, while Sonnet imagines the internal processing of a hypothetical language model reading the passage rather than summarizing it. Predictions about unseen continuations cannot be verified from the passage alone; this difference in content limits interpretation of the support-rate comparison.

After RL, AV explanations reach an FVE of 70.1%, while Sonnet explanations reach 53.8%. This difference is not explained by the AR reconstructing arbitrary explanations well. Using an explanation from a different example gives strongly negative FVE: (-74.8%) for random AV explanations and (-65.6%) for random Sonnet descriptions. Joint RL trains the AV to produce explanations that its paired AR can reconstruct well, so reconstruction quality should not be treated as a measure of passage support.

Sonnet and AV explanations differ in style, so their reconstruction gap may reflect AV–AR co-adaptation. We investigate this further in the [surgical-edit experiment](#surgical-corrections-to-av-explanations) in the appendix.


### Inference-time Interventions

#### Robustness to paraphrasing

*Experimental setup: [Evaluating robustness under perturbation](#evaluating-robustness-under-perturbation).*

We investigate how different levels of [paraphrasing](#paraphrasing-definition) affect model performance when the reconstructedactivation is inserted back into the model's computation path. For this experiment, we use the released layer-32 NLA for Gemma-3-12B-IT ([AV](https://huggingface.co/kitft/nla-gemma3-12b-L32-av), [AR](https://huggingface.co/kitft/nla-gemma3-12b-L32-ar)) and we measure the reconstruction effect on some benchmarks. NLA patching makes almost no difference on ARC-Easy, where answers are short. It hurts performance more on GSM8K and IFEval, where the model generates longer responses. Somewhat surprisingly, some of the tasks can be solved even when a random vector is injected instead of a real reconstruction, potentially because the model can recover from this in the next 16 layers of computation.

{{<figure id="figure-4" src="/images/blog/nla-stability/benchmarks-allpos-accuracy-hires.png?v=bc42b0c6fedd" caption="Figure 4. Accuracy on ARC-Easy, GSM8K, and IFEval (strict average), comparing the unpatched baseline, NLA round-trip, medium and heavy paraphrases, and random-vector patching. Error bars are the archived 95% confidence intervals."/>}}

We also evaluate the model's accuracy on 6 tasks: antonym, capitalize, country-capital, English-French, present-past, and singular-plural, as defined in [Todd et al. (2023)](https://arxiv.org/abs/2310.15213). For each task, we compare four conditions — no intervention, NLA reconstruction, NLA reconstruction with [paraphrase](#paraphrasing-definition) (medium), and paraphrase (heavy). We also include three controls: a random activation vector and a foreign-task vector, coming either from a different example of the same task or a different task. Even though these experiments used the intruction tuned model, we performed them using ICL, without the prompt template. We performed some experiments with and without the prompt template that makes us think that the results can still be trusted, but we wanted to highlight this difference.

{{<figure id="figure-5" src="/images/blog/nla-stability/paraphrasing-task-accuracy.png?v=65988538f6a3" width="100%" align="center" caption="Figure 5. Accuracy on six ICL tasks for the unpatched baseline, NLA round-trip, medium and heavy paraphrases, same-task and cross-task mismatched activations, and a random vector. Paraphrasing has a smaller effect than replacing the activation with a mismatched example on most tasks. Two demonstrations and 50 examples per task; error bars show one standard error."/>}}

[Figure 5](#figure-5) shows that paraphrasing can slightly decrease the accuracy on these tasks but that this decrease is not clearly tied to task difficulty. We don't observe that heavy paraphrasing significantly decreases the accuracy but this might be linked to the fact that for some tasks, the correct reconstruction of the NLA is not required for the task to be completed. In fact we observed that for the "capitalize" task, the same-task mismatched function vector (FV) can still complete the task about 40% of the time.

Capitalization is an easy task, and the tested patch leaves downstream layers that may recover the answer despite the mismatched injection. One possible explanation is that later layers recover information from unpatched prompt positions. The absence of an answer from the verbalization alone does not establish this mechanism. The [capitalization experiment in the appendix](#capitalize-task) tests a later layer (layer 40), where same-task mismatched patching reduces accuracy to zero.

#### Changing the answer of an easy task

*Experimental setup: [IOI task setup](#ioi-task-setup).*

We use Gemma-3-12B-IT with four custom EasyNLA AV–AR pairs at layers [24](https://huggingface.co/achand45/gemma-3-12b-it-nla-L24), [32](https://huggingface.co/achand45/gemma-3-12b-it-nla-L32), [40](https://huggingface.co/achand45/gemma-3-12b-it-nla-L40), and [47](https://huggingface.co/achand45/gemma-3-12b-it-nla-L47), using zero-indexed decoder-block outputs, matching the checkpoint names and Figures 6 and 7. We test indirect object identification (IOI): given a sentence such as “Alice and Bob went to the shop. Alice gave a pear to…”, the model should answer “Bob.” We ask whether mismatched activation patching disrupts this task, and whether editing an NLA explanation can change the answer.

{{<figure id="figure-6" src="/images/blog/nla-stability/ioi-layer-patching.png?v=e05fa2a58c04" width="100%" align="center" caption="Figure 6. IOI accuracy by layer, with bars comparing the baseline, NLA round-trip, mismatched donor, random-vector, and zero-vector controls at the final prompt position. Layers 24, 32, 40, and 47 match the uploaded checkpoint names. The plot uses two demonstrations, with 200 examples at layers 24, 32, and 40 and 189 at layer 47; error bars are 95% Wilson intervals."/>}}

[Figure 6](#figure-6) compares the unpatched baseline, NLA round-trip, mismatched-donor patching, and random- and zero-vector controls across four layers. The unpatched model is highly accurate. NLA round-trips preserve most of that accuracy at layers 24, 32, and 40, but perform worse at layer 47. A mismatched donor preserves high accuracy at layer 24, with progressively lower accuracy at later layers; random- and zero-vector controls largely disrupt the answer. Thus, the effect of replacing the final-position activation depends on both the replacement and the layer.

We then edit the verbalizations themselves. [Figure 7(a)](#figure-7) tests name substitutions among examples where the NLA round-trip answered correctly and its explanation named the IO. layer 24 has no eligible examples: none of its correct-round-trip explanations mentions the IO name. This leaves no occurrence to substitute, rather than demonstrating that substitutions fail. The [appendix](#ioi-verbalizations-by-layer) shows one prompt’s explanations across layers. [Figure 7(b)](#figure-7) instead starts from incorrect round-trips and tests whether supplying the correct name, alone or in an answer sentence, recovers the answer.

{{< figure-pair id="figure-7" left="/images/blog/nla-stability/ioi-text-interventions-compact-a.png?v=fef438776679" right="/images/blog/nla-stability/ioi-text-interventions-compact-b.png?v=db8615438f61" leftAlt="Name-substitution success across layers 24, 32, 40, and 47; layer 24 has no eligible examples" rightAlt="Recovery of wrong IOI answers at layers 24, 32, 40, and 47" caption="Figure 7. **(a)** Among correct NLA round-trips whose explanations name the indirect object, substitute the subject’s name, another name in the prompt, or a name outside the prompt. **(b)** Among incorrect round-trips, supply the correct name alone or in an answer sentence. Rates measure the patched first token and pool 0/2/4/6/8-shot settings. Error bars are 95% prompt-cluster bootstrap intervals; n counts distinct eligible prompts, in each layer and panel. layer 24 has no points in (a): none of its correct-round-trip verbalizations mentions the IO name, so there is no name to substitute. This is an empty selection, not a zero success rate. See [Methods](#summarizing-ioi-text-interventions) and the [full breakdown](#full-ioi-text-intervention-results)." >}}

In [Figure 7(a)](#figure-7), substituting the subject’s name redirects the first token in about 76% of eligible configurations at layer 32 and 73% at layer 40. Substituting another name from the prompt reaches about 63% and 68%, respectively. Names outside the prompt are less effective at these layers: 0% at layer 32 and about 8% at layer 40. At layer 47, all three substitution types exceed 93% success on their eligible configurations.

In [Figure 7(b)](#figure-7), replacing the explanation with the answer sentence recovers the correct first token in 85–100% of the tested configurations, depending on the layer. Supplying the name alone is less consistent: it recovers about 72% at layer 24 but none at layer 47. At layer 47, appending the answer sentence once or twice reaches about 19% and 30%, respectively, compared with 100% when replacing the explanation. The placement and form of the supplied answer therefore matter.

These panels answer different questions on different subsets: redirecting an initially correct answer versus recovering an initially wrong one. Success measures whether the first generated token matches the target name, not whether the model generates the complete name. The [full appendix breakdown](#full-ioi-text-intervention-results) separates shot counts and includes the additional edit conditions.


#### Which edits need to be done to change the answer?

We want to compare three types of explanation edits:

* Query-and-answer substitution: replace references to the base query and answer with the target query and answer.
* Final-paragraph rewrite: replace only the explanation's final paragraph with a fixed template specifying the target answer.
* Combined edit: apply the substitutions and the final-paragraph rewrite.

The final-paragraph condition tests whether editing only the explanation's description of the predicted continuation is sufficient to redirect the answer, as [the last paragraph carries most of the reconstruction value](https://www.lesswrong.com/posts/ahu9BCHPFDg9AJF75/some-observations-about-nla-explanations).

Our main experiments patch activations at layer 32, either at the final prompt token or at the last four prompt tokens. As a reference intervention, we also perform donor patching: we replace activations in the base prompt with activations from a prompt containing the target query and the same demonstrations. We report target-answer rates for each condition. The appendix reports [donor patching across layers](#donor-patching-result), [NLA interventions at layers 24 and 40](#other-layers-result), and the [chat-template variant](#chat-template-answer-redirection).

{{<figure id="figure-10" src="/images/blog/nla-stability/example-edits-country-capital-hires.png?v=71a043d64abc" width="100%" align="center" caption="Figure 10. Schematic of combined explanation edits for country–capital retrieval: substitute “China” with “Chile” and “Beijing” with “Santiago,” and rewrite the final paragraph to specify “Santiago.” The reconstructed activation is patched at the final answer delimiter (:) at layer 32 of Gemma-3-12B-IT. All block numbers are zero-indexed. Excerpts and answers come from the saved seed-0 example."/>}}

{{<figure id="figure-11" src="/images/blog/nla-stability/icl-redirect-raw-L32-hires.png" width="100%" align="center" caption="Figure 11. Target-answer rates for raw-text ICL prompts at layer 32. Colors compare query-and-answer substitution, final-paragraph rewrite only (“Final-unit rewrite only” in the legend), combined edits (“Full edit”), and donor activations. Hatched bars patch the final prompt position; solid bars patch the last four positions. The rightmost group aggregates tasks."/>}}

With patching at the final prompt token alone, substitutions reach 64%, the final-paragraph rewrite reaches 46% and combined edits reach 79%, compared with 81% for donor patching. Combining the edits therefore improves single-token redirection. Capitalize remains weak under both combined edits and donor patching, suggesting that the selected single-token intervention location limits redirection on this task.

With four-token patching, query-and-answer substitutions and combined edits both reach 98%, compared with 59% for the final-unit rewrite alone and 97% for donor patching. These observed rates are close to those for donor patching; without uncertainty estimates, they do not establish equivalent reliability.

#### Using explanation edits to change the task 

We again use the released layer-32 NLA for Gemma-3-12B-IT ([AV](https://huggingface.co/kitft/nla-gemma3-12b-L32-av), [AR](https://huggingface.co/kitft/nla-gemma3-12b-L32-ar)), patching the final prompt position or the last four positions. We test whether explanation edits can redirect answers to a different task while keeping the query fixed. We evaluate three switches: French translation to Spanish translation, capital retrieval to currency retrieval, and antonym generation to comparative generation.

We compare four types of explanation edits:

* Answer substitution: replace references to the base answer with the target answer.
* Task-description substitution: replace descriptions of the base task with descriptions of the target task.
* Task-and-answer substitution: apply both substitutions.
* Combined edit: apply both substitutions and the final-paragraph rewrite.

{{<figure id="figure-12" src="/images/blog/nla-stability/icl-task-switch-raw-L32-hires.png" width="100%" align="center" caption="Figure 12. Target-answer rates at layer 32 for French-to-Spanish translation, capital-to-currency retrieval, and antonym-to-comparative generation, with the query held fixed. Colors compare answer-only, task-only, joint task-and-answer, combined edits including a final-paragraph rewrite, and donor activations. Hatched bars patch the final position; solid bars patch the last four positions. The rightmost group aggregates switches."/>}}

With patching at the final prompt token alone, task-and-answer substitutions reach 0.440, and combined edits reach 0.847, compared with 0.840 for donor patching. The final-paragraph rewrite therefore improves the observed target-answer rate for single-token patching.

With four-token patching, combined edits achieve target-answer rates close to donor patching on the task-switching experiments. Task-and-answer substitutions reach 0.470, while combined edits reach 0.877, compared with 0.897 for donor patching.

Answer-only and task-only substitutions yield low average target-answer rates of 0.057 and 0.003, respectively. These results show that edited NLA reconstructions can redirect answers to a different task while the query remains fixed, with joint task-and-answer edits more effective than either substitution alone. Because the answer edits supply the target answer, success does not establish that the model has adopted a new task rule. Testing that would require generalization to queries whose target answers were not supplied in the edits. The appendix reports the [chat-template variant](#chat-template-task-switching) of this experiment.

## Conclusions

Explanations transfer across NLAs trained with slightly different trajectories, providing evidence against codes specific to an individual AV–AR pair. Shared nonsemantic conventions remain possible. GCG improves reconstruction while degrading readability and passage support, showing that the reconstruction objective can reward token choices whose value is not apparent from their readable meaning.

Reconstruction quality alone does not establish passage support or activation-level faithfulness. After RL, AV explanations reconstruct better than Sonnet explanations despite having much lower passage support. These results motivate evaluating reconstruction and explanation content separately.

Our paraphrasing experiments find limited additional task-performance degradation beyond NLA reconstruction on simple tasks. Mismatched-activation controls also show that some tasks remain solvable despite replacing the activation at the tested position. Structured explanation edits can redirect outputs toward supplied target answers, including answers from a different task, with observed rates close to donor patching in some settings. Whether these edits induce a task rule that generalizes beyond the supplied answers remains open.


## Methods

### Natural Language Autoencoders

NLAs consist of two modules: the Activation Verbalizer (AV) and the Activation Reconstructor (AR). Both of these modules are LLMs. The AV is initialized as a copy of the model to be analysed. AR is initialized with the [1,L] truncation of the base model where L stands for the index of the layer where the input activation for the NLA lives. The AV takes an input activation which is injected in a place-holder token within a prompt-template which is meant to direct the AV towards explaining the activation (see AV prompt in appendix). The AR takes the corresponding verbalization from the AV and reconstructs the activation. The AV is analogous to an encoder while AR is analogous to a decoder. Instead of outputting a token, the AR has a value head that outputs the reconstructed activation which is supposed to match the original input activation in terms of mean-squared-error loss. We note that the activation is normalized for the loss computation and hence, the AR's objective is to reconstruct its direction.

The GCG, benchmark/ICL paraphrasing, and main answer/task-redirection experiments use the [NLAs released by Anthropic](https://github.com/kitft/natural_language_autoencoders) for Gemma-3-12B-IT at layer 32 ([AV](https://huggingface.co/kitft/nla-gemma3-12b-L32-av), [AR](https://huggingface.co/kitft/nla-gemma3-12b-L32-ar)). Passage support, IOI, and the additional layer-24/layer-40 experiments use our custom Gemma EasyNLA checkpoints. Seed stability and the warm-start ablation use Qwen2.5-7B-Instruct EasyNLAs at layer 20.

[AV prompt-generation code](https://github.com/kayleeisokay/soar_nla/blob/main/soar-nla-daniel/training/nla/datagen/stage3_build.py#L71)

### Evaluating NLAs reconstruction

We list the definition of the metrics below:

$$\text{normalize}(v) = \frac{v}{\|v\|} \cdot \sqrt{d_{model}}$$

$$SE = \sum_{v_i \,\in\, \text{samples}} \left(\text{normalize}(v_i) - \text{normalize}(\hat{v}_i)\right)^2$$

$$\text{baseline} = \sum_{v_i \,\in\, \text{samples}} \left(\text{mean}(\{\text{normalize}(v_i)\}_{v_i \in \text{samples}}) - \text{normalize}(v_i)\right)^2$$

$$FVE = 1 - \frac{SE}{\text{baseline}}$$

where $\hat{v}_i$ is the AR's reconstruction of $v_i$.

#### Evaluating robustness under perturbation

To evaluate inference time interventions to NLAs we did interventions on the verbalized text, then used that text to generate a new reconstructed activation and used that reconstructed activation as if it was the real reconstruction, as done in [Fraser-Taliente et al. (2026)](https://transformer-circuits.pub/2026/nla/).

We used two baselines: the random-vector control tests whether patching any off-manifold vector into that layer disrupts or helps generation, independent of content. For each query, we sample Gaussian noise and rescale it to match the $L_2$ norm of the real activation. The random vector can be interpreted as a random direction, but matched in magnitude to the real activation. It isolates whether patching in something content-free but scale-appropriate hurts performance, so a drop in accuracy can be attributed to the content of the patched vector rather than just to the model being perturbed by an out-of-distribution norm; and the foreign-vector control, which instead tests content specificity rather than on/off-manifold sensitivity: here we patch in a real activation vector, captured the same way as the original, but drawn from a different task's ICL prompt (e.g., patching an "antonym" activation into a "country-capital" query).

Experiment code: [benchmark and ICL evaluation](https://github.com/kayleeisokay/soar_nla/blob/main/natural_language_autoencoders_kaylee/notebooks/executor.py), [function-vector experiments with a chat template](https://github.com/kayleeisokay/soar_nla/tree/main/soar-nla-daniel/soar_nla/fv_chat_template), [paraphrasing prompts](https://github.com/kayleeisokay/soar_nla/blob/main/natural_language_autoencoders_kaylee/notebooks/executor.py#L520-L535), [layer-40 capitalization experiment](https://github.com/kayleeisokay/soar_nla/tree/main/natural_language_autoencoders_kaylee/easy_nla/capitalize_L40).

### Training NLAs and Easy NLA

Because each part of the NLA is basically a full model, we fork [EasyNLA](https://github.com/asherps/EasyNLA), a repository that creates NLAs by training LoRAs instead of doing full model finetuning. We found similar performance using LoRAs at a much lower training cost, appendix. Compared to official NLA, EasyNLA uses ~62% less RL data, but reaches 75.4% FVE, compared to 75.2% for official NLA. We trained four NLAs that share the same base model and layer: Qwen2.5-7B-Instruct, layer 20 (0-indexed; layer 20 refers to the output of block 20, equal to hidden_states[21]), using the default configuration of EasyNLA, which can be found in the appendix.

The code to generate all the experiments in this post is available in this [repo](https://github.com/kayleeisokay/soar_nla/tree/main).

#### Seed stability setup

For this set of experiments, we adapt ceselder/qwen3-8b-nla-L24-finefineweb-100k by regenerating the layer-20 activations for Qwen2.5-7b. In this dataset, for each text sample from FineFineWeb 10 randomly sampled positions (without replacement) from position indices >= 50 were selected. The training set goes through a 25-25-50 split to get the corresponding AR SFT-AV-SFT-RL data partitions. All four NLAs share identical partitions.

For each sample in the AV SFT and AR SFT datasets, "gold explanations" refer to the summary generated by an LLM for the text up to the last token. The summary explanations in ceselder/qwen3-8b-nla-L24-finefineweb-100k were generated using Claude Sonnet 4.6. These explanations are used for the AR SFT and AV SFT.

{{<figure id="figure-13" src="/images/blog/nla-stability/training-splits-tree.png" width="100%" align="center" caption="Figure 13. Training branches for four NLAs, varying LoRA initialization, RL data order, and rollout stochasticity. NLAs 1 and 2 share the SFT adapters and RL data order; NLA 4 changes RL data order, and NLA 3 follows a separate initialization branch."/>}}

| NLA | SFT | RL data order | Differs from NLA 1 in |
| --- | --- | --- | --- |
| 1 | `--init-seed 0 --data-seed 0` (init controls lora adapter initialization and data seed control SFT data order) | `--seed 0` | — |
| 2 | reuses NLA 1's SFT adapters | `--seed 0` | RL rollout stochasticity only |
| 3 | new SFT, `--init-seed 1 --data-seed 0` | `--seed 0` | SFT LoRA init |
| 4 | reuses NLA 1's SFT | `--seed 1` | RL data order |

Table 1. Description of the four NLAs.

Experiment code and outputs: [cross-NLA reconstruction evaluation](https://github.com/kayleeisokay/soar_nla/blob/main/natural_language_autoencoders_kaylee/notebooks/executor.py), [swap-matrix paraphrase generation](https://github.com/kayleeisokay/soar_nla/blob/main/easy_nla_abhinav/notebooks/paraphrase/paraphrase_generations.py#L57-L89), [saved verbalizations](https://github.com/kayleeisokay/soar_nla/tree/main/easy_nla_abhinav/notebooks/paraphrase/results/generations/temp0).

##### NLA Identification with logistic regression methodology

To measure if the text produced by each of the NLAs is identifiable as being from a different distribution, we encode it using the original LLM and extract the activations at layer 5, layer 13, and layer 20 activations mean-pooled over the verbalization tokens to get data points for three independent regressions. Features are centred per-dimension and divided by a single scalar: the training rows' RMS distance from their mean with both statistics estimated on the training fold only.

We perform a 5 fold grouped by prompt nested CV and report the accuracy. The nested CV is performed for hyperparameter selection—in this case, C where C is the inverse L2 regularisation strength. The configs we used were: L2 penalty, max_iter=5000, default solver and intercept. The gridsearch values for C in the inner folds were logspace(-4, 4, 9).


##### NLA Discrimination by human test

Our dataset is 128 held-out documents used for the FVE calculation during training. We used the four NLAs and sampled each of them 4 times per document at temperature 1.0 and seeds 0-3. From this set we built 42 trials as follows:

- For each of the 6 NLA pairs (NLA_i vs. NLA_j), we constructed 7 trial slots to get 42 trials, and randomly shuffled the order of trials.
- Then we randomly draw 42 documents (without replacement) from the 128-doc dataset and assign it to the slots.
- For each trial, we show four verbalizations of that document: two from NLA_i and two from NLA_j, each pair using two distinct seeds randomly selected from 0-3. They are placed in random slot order and labelled A, B, C, D.
- The reader picks one of the three possible splits (AB/CD, AC/BD, AD/BC) to put the same-NLA texts together. The correct button is balanced across the trials by construction (14/14/14).

Each trial is a 3-way choice, so a guesser with no information is right with probability 1/3 per trial, independently. Our null and alternative hypotheses are:

- H0: the reader cannot tell the pairs apart; correct count X ~ Binomial(42, 1/3).
- H1: the reader is right more often than 1/3.

Under H0 the expected score is 14/42. The p-value is the one-sided tail:

$$P(X \geq k) = \sum_{i=k}^{42} \binom{42}{i} \left(\frac{1}{3}\right)^i \left(\frac{2}{3}\right)^{42-i}$$


[Experiment code](https://github.com/kayleeisokay/soar_nla/tree/main/easy_nla_abhinav/notebooks/human_distinguish); [human-test notebook](https://github.com/chand-ab/easy_nla/tree/main/notebooks/human_distinguish).

### Optimizing reconstructed text to improve activation reconstruction

In this section, we test whether optimizing an explanation for reconstruction also improves its passage support. We use [Greedy Coordinate Gradient (GCG)](https://arxiv.org/abs/2307.15043) to search for token substitutions that reduce reconstruction error. The activation reconstructor maps the explanation to a reconstructed activation through a differentiable path from the input embeddings, so we can optimize the reconstruction error with respect to token substitutions. We initialize the search from the AV explanation. We run GCG on 1,543 Alpaca rollouts with 20 optimization steps, a batch size of 64, and top-k=64 evaluated on all tokens in AV explanations.

Unconstrained GCG can reduce reconstruction error by producing uninterpretable text. To test whether we can improve reconstruction while keeping the explanation likely under the AV, we experiment with different NLL penalties against the AV policy. We test $\beta \in \{0, 0.01, 0.1\}$.

$$L(x) = \frac{\text{MSE}(AR(x), a)}{\text{Var}(v)} + \beta \cdot \text{NLL}_{AV}(x \mid a)$$

As a control, for each example and β, we made the same number of edits but on fully random positions and tokens.

[GCG experiment code](https://github.com/kayleeisokay/soar_nla/tree/main/soar-nla-daniel/soar_nla/gcg).

### Warm-start ablation

We train EasyNLA AV–AR pairs on Qwen2.5-7B-Instruct at layer 20, comparing the original SFT warm-start run with cold-start RL runs with and without KL regularization. We evaluate held-out reconstruction and use Gemma-3-27B-IT to infer speaker attributes from AV explanations of 2,585 SelfDescribe examples, with mismatched explanations as a control. The [appendix](#skipping-warm-start) reports the results and judge prompt.

[Warm-start ablation code](https://github.com/kayleeisokay/soar_nla/tree/main/soar-nla-daniel/soar_nla/warmstart_ablation).

### IOI task setup

All IOI layer numbers refer to zero-indexed decoder-block outputs, matching the uploaded checkpoint names: layers 24, 32, 40, and 47.

We took the following prompt with n demonstrations (n=0, 2, 4, 6, 8):

```
Answer with a single word only. No punctuation, no explanation.
Q: Friends Sarah and Alice found a coconut at the park. Alice gave it to
A: Sarah.
…
…
Q. When Mark and Mona got a pineapple at the library, Mona decided to give it to
A:
```

The model is supposed to name the IOI name for the corresponding question. Our experiments were performed with Gemma 12B IT (temperature=0) and we measured the accuracy based on whether the first token that the model generates is equal to the first token of the IOI name. Note that any special token, including new lines or spaces, would be considered a wrong answer. We took this approach because we noticed that if the model is allowed to generate several tokens, it uses the first few special tokens to recover the actual answer, poisoning the effect of the patching.


[Experiment code](https://github.com/kayleeisokay/soar_nla/tree/main/easy_nla_abhinav/notebooks/ioi)

#### Summarizing IOI text interventions

Figure 7 summarizes saved per-example results from [the IOI experiment repository](https://github.com/kayleeisokay/soar_nla/tree/c415cc158157fc01e57290136182db37d0ea991c/easy_nla_abhinav/notebooks/ioi). No new inference is run. We pool the 0-, 2-, 4-, 6-, and 8-shot settings at each layer. Panel (a) includes only examples whose unedited round-trip answered correctly and whose explanation contained the IO name. Panel (b) includes examples whose unedited round-trip answered incorrectly. Both panels score the first generated token, as in the original plot; their selected samples therefore differ.

For panel (a), the subject-name condition substitutes the other name in the query; the other-in-prompt condition substitutes a name from a demonstration. The latter and the outside-prompt condition each pool up to three alternative-name draws per example. For panel (b), “Name only” replaces the explanation with the correct name; the other conditions append the fixed answer sentence once or twice, or replace the explanation with that sentence. The sentence template is: `Final token "to" is a preposition mid-sentence, immediately expecting "{name}" as the answer.` The unedited recovery rate is zero at every tested layer and is omitted from the compact panel. Additional edit variants and the shot-specific results remain in the [full breakdown](#full-ioi-text-intervention-results).

In the full breakdown, “In prompt” and “Outside” substitute other demonstration names and names absent from the prompt, respectively. Their ×3 variants insert three copies of the replacement name at each substitution site. “Sentence: subject,” “Sentence: in prompt,” and “Sentence: outside” replace the entire explanation with the same fixed sentence naming the chosen person. In panel (b), “answer” means the correct IO name and “subject” means the other name in the query; “no name” removes the name from the sentence. “Prepend” places the answer sentence before the explanation. Unedited bars in (a) score the original correct IO name, while edited bars score the substituted name. Sample counts for every layer, shot count, and condition accompany the [downloadable full results](/images/blog/nla-stability/ioi-text-interventions-full.csv?v=59ab578762d3).

Point estimates pool all available eligible configuration/draw outcomes for each condition. To account for repeated prompts across shot counts and alternative-name draws, we resample distinct prompt IDs with replacement 5,000 times, retaining all outcomes for each sampled prompt. Error bars show the 2.5th and 97.5th percentiles of the resampled pooled rates. The n labels count distinct eligible prompts in each panel and layer; some individual edit conditions have fewer eligible prompts when a required name is unavailable. Layer 24 remains on the axis in panel (a), but has no points because none of its correct-round-trip explanations contains the IO name. Substituting a name is therefore inapplicable; the missing points do not indicate zero intervention success. These intervals describe variation across the sampled prompts, not across training runs, and are degenerate for conditions where every observed outcome is identical.

<!--
## Bibliography

- [https://transformer-circuits.pub/2026/nla/#characterizing-nla-confabulations](https://transformer-circuits.pub/2026/nla/#characterizing-nla-confabulations)
- [https://www.lesswrong.com/posts/LQXWiF8PyJ5ojNsEv/how-robust-are-natural-language-autoencoders-to](https://www.lesswrong.com/posts/LQXWiF8PyJ5ojNsEv/how-robust-are-natural-language-autoencoders-to)
- [https://www.lesswrong.com/posts/QQQAcKuWK6k98FivY/can-activation-verbalizers-surface-an-internal-chain-of-1](https://www.lesswrong.com/posts/QQQAcKuWK6k98FivY/can-activation-verbalizers-surface-an-internal-chain-of-1)

 Section not yet written in the source document. -->

## Appendix

### Difference between [Fraser-Taliente et al.](https://transformer-circuits.pub/2026/nla/)'s NLA and EasyNLA

AV injection difference between kitft (original NLA repo) and EasyNLA:

```
kitft:
marker token → embedding lookup → REPLACED by the activation vector → block 0 → block 1 → … → block 47 → verbalization
```

The marker's own embedding is discarded. The activation enters as if it were a token embedding and gets processed by every block.

```
EasyNLA:
marker token → embedding lookup (kept) → block 0 → block 1 → at block 1's OUTPUT: resid += ‖resid‖ · v/‖v‖ → block 2 → … → block 47 → verbalization
```

### AV Prompt

```python
_DEFAULT_ACTOR_TEMPLATE = """You are a meticulous AI researcher conducting an important investigation into activation vectors from a language model. Your overall task is to describe the semantic content of that activation vector.

We will pass the vector enclosed in <concept> tags into your context. You must then produce an explanation for the vector, enclosed within <explanation> tags. The explanation consists of 2-3 text snippets describing that vector.

Here is the vector:

<concept>{injection_char}</concept>

Please provide an explanation."""
```

### Prompts used for Gemma 3 27B IT paraphrasing in the swap matrix computations

```python
PARAPHRASE_PROMPT_MEDIUM = (
    "Please paraphrase the text provided below, preserving its exact meaning "
    "but expressing it in substantially different words and sentence structure "
    "than the original. Do not simply swap individual words for synonyms -- "
    "restructure the sentence(s) genuinely.\n\n"
    "<text>\n{text}\n</text>\n\n"
    "Include your final paraphrased text in <transformed_text> tags."
)
PARAPHRASE_PROMPT_HEAVY = (
    "Please paraphrase aggressively. Preserve meaning and uncertainty, but "
    "substantially change wording, syntax, and organization. Do not simply swap individual words for synonyms -- "
    "restructure the sentence(s) genuinely.\n\n"
    "Text:\n{text}\n\n"
    "Respond with only the paraphrased text, wrapped exactly like this:\n"
    "<transformed_text>your paraphrase here</transformed_text>"
)
FIDELITY_CLAUSE = (
    "\n\nPreserve every specific detail. Do not remove, alter, generalise or "
    "summarise any fact, number, date, name, or quoted string -- every specific "
    "that appears in the original must appear in your paraphrase. Rewrite how it "
    "is said, not what is said."
)
```

### Verbalizations from the four NLAs

```
NLA 1:
"explanation": "Historical/analytical essay introducing a book or topic about global urban poverty, establishing \"the city\" as central subject matter requiring elaboration.\n\nThe opening paragraph's argumentative momentum suggests a pivot to a specific scholar or book review — \"In recent years, the global city has become a central topic of discussion\" implies a named author or work offering insight into this phenomenon.\n\nFinal token \"Ross\" is a proper name (\"Stephen Ross\") mid-sentence or mid-paragraph, immediately requiring a title, affiliation, or verb phrase like \"has recently argued\" or \"in his book\" to introduce a specific scholar's perspective on global poverty."

NLA 2:
"Academic essay tone establishing a historical or analytical argument about urban poverty and inequality, likely referencing a specific book or scholar discussing gentrification or urban change.\n\nThe sentence structure \"But there is one place where the world has changed dramatically\" signals a pivot to a specific example or expert, expecting elaboration on a particular city or phenomenon related to urban poverty and inequality.\n\nFinal token \"Ross\" begins \"Mark Ross\" — a proper name mid-phrase, \"Mark Ross\" likely a professor or author being cited, inside \"See also\" or \"Mark Ross\" as subject of a new sentence, expecting continuation like \"in his book/article\" or \"is a sociologist at...\""

NLA 3:
"Academic/journalistic register introducing a book or essay about urban poverty and neoliberalism in America, establishing context about cities and economic crisis.\n\nNarrative momentum: \"The best way to understand the current economic crisis in America's cities\" sets up a framing question or thesis, signaling a solution or expert commentary is being introduced.\n\nFinal token \"Ross\" begins a proper noun phrase \"Mark Ross\" mid-sentence (\"Mark Ross\"), likely a person's name mid-phrase (\"Mark Ross, a sociologist...\"), expecting a verb like \"has\" or \"argues\" to continue the sentence introducing a specific author or scholar."

NLA 4:
"Academic/journalistic essay tone, establishing a thematic argument about global poverty and urbanization (\"the world's most important story\").\n\nThe text promises a specific book or essay title as a source of further reading or analysis — \"A new book offers a provocative look at the world's poor\" suggests an author citation or review-style introduction follows.\n\nFinal token \"Ross\" is a surname beginning a byline or citation (\"Mark Ross\"), part of an attribution or author introduction (\"Mark Ross\" — likely a professor or journalist), immediately following \"A new book by\" — the sentence \"Mark Ross\" opens a noun phrase or direct quote, expecting a title, description, or continuation like \"of New York University\" or \"has written a fascinating account of...\""
```

### IOI verbalizations by layer

Below is one two-shot prompt and its verbalizations at the four tested layers. The [experiment results directory](https://github.com/kayleeisokay/soar_nla/tree/main/easy_nla_abhinav/notebooks/ioi/results) contains the per-example outputs in `ioi_verbalizations_*shot_L*.json` files. The text, figures, and filenames all use the same zero-indexed layers: `L24`, `L32`, `L40`, and `L47`.

```
prompt:
Answer with a single word only. No punctuation, no explanation.
Q: Then, Kristina and Susan had a lot of fun at the market. Kristina gave a honeydew to
A: Susan
Q: While Ethel and David were commuting to the river, Ethel gave a grape to
A: David
Q: Then, Mona and Dianne went to the library. Mona gave a banana to
A:
```

```
Layer 24:
"explanation": "Pattern of structured Q&A examples: each sentence follows \"Person A gives toy to Person B\" with the recipient named.\n\nRepetition/continuation pattern: \"In the first example, Alice gives the doll to Bob. The recipient is Bob.\" establishes a template requiring the reciprocal action \"the doll to [the other person's name].\"\n\nFinal token \"to:\" is a preposition mid-sentence, \"Alice gave the ball to Bob, giving the doll to:\" — immediately expects \"Bob\" or \"the other person\" to complete the reciprocal exchange.\n\"to:\" closes the parallel structure \"who gave the toy to the other person.\"\n\"In the second example, Alice gives the doll to Bob. The recipient is:\"\n\"Alice gave the ball to:\""

Layer 32:
"explanation": "Repetition/continuation pattern: each paragraph is a duplicated version of the previous, with \"Dana gives a book to Dana and Dana gives a book to\" repeated verbatim.\n\nNarrative momentum: the duplicated paragraph structure mirrors the original, cycling through Dana→Dana and Dana→Dana exchanges, now reaching the final sentence.\n\nFinal token \"to\" closes \"Dana gives a book to Dana and Dana gives a book to,\" immediately requiring \"Dana\" to continue the duplicated sentence pattern (\"Dana gives a book to Dana and Dana gives a book to Dana\"), completing the repeated pair. e.g., \"Diana\" or \"Diana and Dana.\"\ne.g., \"Diana and Dana\"\ne.g., \"Diana\"\ne.g., \"Diana and Dana\""

Layer 40:
"explanation": "Repetition/continuation pattern: each paragraph is a duplicated version of the previous, with \"Dana gives a book to Dana and Dana gives a book to\" repeated verbatim.\n\nNarrative momentum: the duplicated paragraph structure mirrors the original, cycling through Dana→Dana and Dana→Dana exchanges, now reaching the final sentence.\n\nFinal token \"to\" closes \"Dana gives a book to Dana and Dana gives a book to,\" immediately requiring \"Dana\" to continue the duplicated sentence pattern (\"Dana gives a book to Dana and Dana gives a book to Dana\"), completing the repeated pair. e.g., \"Diana\" or \"Diana and Dana.\"\ne.g., \"Diana and Dana\"\ne.g., \"Diana\"\ne.g., \"Diana and Dana\""

Layer 47:
"explanation": "Structured table/data pattern: pipe-delimited table listing book titles and ownership counts, establishing a consistent format throughout.\n\nNarrative momentum of the \"Who Owns More Books\" section: each row presents a person's name and a comparison of book counts, building toward a complete answer.\n\nFinal token \"and\" is a conjunction mid-row in \"The Number of Books That Susan and [Name] Own,\" immediately requiring \"Clarence\" or \"Clarence and\" to complete the comparison. \"Susan and\" is followed by \"Clarence\" and \"The Number of Books That Susan and Clarence Own.\"\n\"Clarence\" or \"Clarence and The Number of Books That Susan and\"\n\"Gretchen\" or \"Gretchen.\"\n\"The Number of Books That Susan and\"\n\"Gretchen.\""
```

### Full IOI text-intervention results

The full breakdown retains the original displayed edit conditions and separate 0-, 2-, 4-, 6-, and 8-shot results summarized in [Figure 7](#figure-7). The figure is redrawn with shorter labels and a shared legend. Its error bars retain the original row-based Wilson intervals rather than the prompt-cluster bootstrap used in the compact plot. [Open the vector figure](/images/blog/nla-stability/ioi-text-interventions-full.svg?v=30c6eaf91d05) or [download the values, sample counts, and intervals](/images/blog/nla-stability/ioi-text-interventions-full.csv?v=59ab578762d3).

{{<figure id="figure-a1" src="/images/blog/nla-stability/ioi-text-interventions-full.png?v=10d3002e5bb9" caption="Figure A1. Full IOI text-intervention results, separated by layer and shot count. **(a)** Name substitutions and sentence replacements among correct round-trips whose explanations name the IO. **(b)** Answer recovery among incorrect round-trips. Colors show the number of demonstrations. Condition labels are shared down each column. Error bars retain the original 95% row-based Wilson intervals; absent bars indicate unavailable conditions or empty selections, not zero success. See [Methods](#summarizing-ioi-text-interventions) for condition definitions."/>}}

### Paraphrasing robustness

#### Paraphrasing Definition

{{<figure id="figure-14" src="/images/blog/nla-stability/paraphrase-prompt-example.png" width="100%" align="center" caption="Figure 14. Example original explanation, medium and heavy paraphrases, and the corresponding paraphrasing prompts. Both prompts request preservation of meaning; the heavy prompt additionally requests substantial changes to wording, syntax, and organization."/>}}

#### Capitalize task

For this follow-up, we use Gemma-3-12B-IT with our [custom EasyNLA at layer 40](https://huggingface.co/achand45/gemma-3-12b-it-nla-L40). In the layer-32 reconstruction-robustness experiment, the "capitalize" task stood out. The mismatched FV (same-task) accuracy was roughly 40%, higher than any other task. We hypothesized this was because the model had enough remaining layers to recover the true activation on its own. To test this, we reran the experiment with an NLA trained on a later layer (layer 40). As shown in [Figure 15](#figure-15), the mismatched FV (same-task) accuracy drops to 0 at this later layer, consistent with our hypothesis. This is consistent with reduced downstream recovery, but does not establish that mechanism: the information available at unpatched positions and the role of the patched position also vary with depth.

{{<figure id="figure-15" src="/images/blog/nla-stability/capitalize-later-layer.png" width="100%" align="center" caption="Figure 15. Capitalization accuracy with an NLA trained at layer 40, comparing the unpatched baseline, NLA round-trip, medium and heavy paraphrases, same-task and cross-task mismatched activations, and a random vector."/>}}

### Donor Patching Result

This sweep uses Gemma-3-12B-IT donor activations directly, without NLA reconstruction, at the layers shown in the plot. Donor patching reveals a layer-dependent change in which token positions are sufficient for answer redirection. Patching the query token yields target-answer rates of 94-98% through layer 24, falling to 47% at layer 28 and 7% at layer 32. Patching the final answer delimiter (`:`) shows the opposite pattern. It has no effect through layer 24, reaches 38% at layer 28, and rises to 78% at layer 32 and 98% at layer 40.

Patching the last four tokens maintains target-answer rates of 94-98% through layer 40. This region includes both the query token and the final delimiter, making it effective across the transition in which single-token position supports redirection. These findings identify positions sufficient for redirection at different depths.

{{<figure id="figure-16" src="/images/blog/nla-stability/icl-donor-sweep-raw-L32-hires.png" width="100%" align="center" caption="Figure 16. Donor target-answer rate by zero-indexed patched layer. Green squares patch the query token; orange triangles patch the final answer delimiter (:); blue circles patch the last four prompt positions. The vertical dotted line marks layer 32, used for the main NLA interventions."/>}}

### Other layers result

We use Gemma-3-12B-IT with our custom EasyNLA AV–AR pairs at [layer 24](https://huggingface.co/achand45/gemma-3-12b-it-nla-L24) and [layer 40](https://huggingface.co/achand45/gemma-3-12b-it-nla-L40). These layer numbers are zero-indexed, matching the plot and checkpoint names.

{{<figure id="figure-17" src="/images/blog/nla-stability/icl-redirect-raw-L24-L40-hires.png" width="100%" align="center" caption="Figure 17. Explanation-edit target-answer rates at layer 24 (top) and layer 40 (bottom). Colors compare query-and-answer substitution, final-paragraph rewrite only, combined edits, and donor activations. Dotted bars patch the query position; hatched bars patch the final position; solid bars patch the last four positions."/>}}

At layer 24, query-and-answer substitutions redirect answers reliably when patched at the query token, reaching 96%, close to donor patching (97%). Adding the final-unit rewrite reduces performance to 88%, while the rewrite alone reaches 19%. Four-token patching shows the same ordering: substitutions reach 92%, combined edits 81%, and the rewrite alone 17%. Final-token edits are nearly ineffective. These results show that NLA edits can redirect answers at the earlier query position, where lexical substitutions outperform the final unit rewrite.

At layer 40, redirection works at the end of the prompt, while query-token edits produce no target answers. Single-token substitutions, final-unit rewrites, and combined edits reach 70%, 58%, and 70%, respectively, compared with 97% for donor patching. Four-token patching raises these rates to 87%, 87%, and 92%, approaching the donor rate of 98%. Capitalize remains weak under single-token combined edits (40%), despite successful donor patching (100%), but improves to 92% with four-token edits. Thus, multiple-position patching still benefits NLA edits even where a single donor activation is sufficient.

### Chat template (answer redirection)

We use the released layer-32 NLA for Gemma-3-12B-IT ([AV](https://huggingface.co/kitft/nla-gemma3-12b-L32-av), [AR](https://huggingface.co/kitft/nla-gemma3-12b-L32-ar)) with chat-template prompts, patching the final assistant-header position or all four header positions.

{{<figure id="figure-18" src="/images/blog/nla-stability/icl-redirect-chat-L32-hires.png" width="100%" align="center" caption="Figure 18. Explanation-edit target-answer rates with chat-template prompts at layer 32. Colors compare query-and-answer substitution, final-paragraph rewrite only, combined edits, and donor activations. Hatched bars patch the final assistant-header position; solid bars patch all four assistant-header positions. The rightmost group aggregates tasks."/>}}

With chat-template prompts, combined NLA edits achieve higher target-answer rates than donor patching at the tested positions: 72% versus 8.7% for single-token patching and 92% versus 84% for four-token patching. Donor patching is also less effective with the chat template than in raw text, where it reaches 81% and 97% for single-token and four-token patching, respectively.

The final-paragraph rewrite provides a larger benefit with the chat template. Single-token substitutions reach 48% with the template and 64% in raw text. Adding the rewrite raises these rates to 72% and 79%. With four-token patching, substitutions reach 79% with the template and 98% in raw text, while combined edits reach 92% and 98%. Thus, the rewrite substantially improves chat-template interventions, including in the four-token setting where it provides no additional benefit in raw text.

### Chat template (task switching)

We repeat the task-switching experiments using the released layer-32 NLA for Gemma-3-12B-IT ([AV](https://huggingface.co/kitft/nla-gemma3-12b-L32-av), [AR](https://huggingface.co/kitft/nla-gemma3-12b-L32-ar)) with a chat template, patching either the final assistant-header token or all four assistant-header tokens. Single-token task-and-answer substitutions reach 0.463, while combined edits reach 0.950, compared with 0.517 for donor patching. With four-token patching, substitutions reach 0.543, and combined edits reach 0.983, compared with 0.783 for donor patching.

Combined edits achieve higher observed target-answer rates with the chat template than in raw text: 0.950 versus 0.847 for single-token patching and 0.983 versus 0.877 for four-token patching. Both controls again produce no target answers. The two prompt formats use the same NLA pair, but their instructions and patched positions differ, so this comparison does not isolate the effect of chat formatting.

### Skipping Warm-Start

These three runs train EasyNLA AV–AR pairs on Qwen2.5-7B-Instruct activations at layer 20; their checkpoints are in the [warm-start ablation model repository](https://huggingface.co/Yooniel/qwen2.5-7b-instruct-nla-L20). The NLA warm start uses supervised pairs of activations and llm-generated summaries before RL begins. These summaries may contain information that is present in the source context but not encoded in the target activation. If so, warm-start SFT could encourage the AV to produce details that are not supported by the activation and contribute to later confabulations. The warm-start is also expensive, since it requires generating a large dataset of summaries with a capable external model. We therefore test whether an NLA can be trained with RL directly from the base model, without SFT initialization. This removes the synthetic summary supervision and tests whether the reconstruction objective alone is sufficient to learn a useful natural-language bottleneck.

{{<figure id="figure-19" src="/images/blog/nla-stability/warmstart-training-fve-hires.png" width="100%" align="center" caption="Figure 19. Training FVE over RL steps for the original warm-start run (red), a cold-start run without KL regularization (teal), and a cold-start run with KL regularization (blue). Annotations show initial and peak FVE for each run."/>}}

RL can train an NLA without an SFT warm start, although the reconstruction is slightly lower. The original run with warm-start achieves the best reconstruction, reaching a peak FVE of 75.9%. Without KL regularization, the cold-start run without KL-penalty reaches 72.2% FVE, while the cold-start run with KL-penalty reaches 68.9%. Held-out FVE follows the same ordering at 74.4%, 70.9%, and 66.5%, respectively.

We then test whether the resulting explanations contain information that can be recovered by an independent judge. For 2,585 [SelfDescribe](https://huggingface.co/datasets/Transluce/SelfDescribe-Llama-3.1-70B-Instruct/) examples, we generate one explanation from each AV and ask Gemma-3-27B-IT to infer a known speaker attribute from the explanation alone. As a control, we repeat the evaluation using an explanation from a different example.

{{<figure id="figure-20" src="/images/blog/nla-stability/warmstart-selfdescribe-judge-hires.png" width="100%" align="center" caption="Figure 20. SelfDescribe attribute-inference accuracy from AV explanations: the original warm-start run, cold-start without KL, and cold-start with KL, grouped by gender, occupation, country, religion, and pooled examples. Yellow bars show the random-control reference; the evaluation procedure also uses mismatched explanations as a control."/>}}

The original run with warm-start reaches 69.9% overall accuracy, compared with 68.2% for the cold-start run without KL and 60.9% for the cold-start run with KL. Control accuracy remains close to chance for all three runs. The cold-start model therefore achieves comparable performance without SFT, despite having a lower peak FVE.

Judge Prompt:

| This is a description of some text. Read the description and answer: {question} {options} Description: """ {explanation} """ Choose the single best option even if you are unsure. Answer with one letter (A, B, C, D or E) and nothing else. |
| :---- |

Example verbalization:

| Source text: March 19th was quite a colorful day at Washington's Lake Sammamish State Park as Vibha Seattle successfully celebrated their second big Holi Event, 'Holi Hai'. The festivities were a hit, as children and adults of all ages and backgrounds came out to support Vbha and partake in the vibrant activity accompanied with traditional food and music. While the Vibha Seattle action center is rather young, only launching in October of 2012, it has accumulated more than 150 volunteers spreading the word about Vibha. The annual |
| :---- |

Original NLA explanation

| Narrative momentum: student blog post about Hindu Student Council's community service and fundraising activities at UC Davis, establishing context for describing the India Day event. The paragraph structure is building toward detailing the significance of the India Day fundraiser — "The HSC has been working hard to raise awareness of Indian culture" sets up elaboration on specific programs or events. Final token "annual" is mid-sentence after "The annual," requiring a noun phrase next (likely referencing a specific event like "India Day fundraiser" or "India Day scholarship") to complete the sentence. |
| :---- |

No SFT, KL 0.01

| This symbol  \<concept\> annual \</concept\> likely represents a concept related to yearly events or cycles. It could be associated with periodic occurrences such as anniversaries, seasons, or recurring activities.\</ex\> "In many cultures and contexts, 'annual' signifies a cycle of one year, often used in planning or scheduling." "This symbol might also indicate a focus on long-term trends or events that repeat every twelve months." "Given its usage in Hindi, it could represent concepts like festivals or significant dates that occur annually." "The symbol could be utilized in contexts where yearly achievements or events are highlighted." "In the Indian culture, this symbol represents a cycle of giving back to society through volunteering." "The annual event organized by the club has been successful in raising awareness about volunteerism." "The annual fundraising event is a tradition for the organization. The organization hosts the annual marathon to raise funds." |
| :---- |

No SFT, KL 0.0

| This vector "\<concept\> annual\</concept\>" means that the Asian Indian Cultural Association at UCSF has several traditions of volunteering events for their philanthropy fair at MIT in America; one of their major annual tradition is the donation of these bookmarks to all members of Pi Sigma Alpha at NYU for our volunteer project called Water Awareness. Therefore many students at our club have suggested the Winter fundraiser as one of the main reasons why Water Festival is held every year in USA! The organization hosts various events related to this charity – one of the highlights is their annual water donation campaign. All AWARE events are motivated by the water festival theme; thus the winter fundraiser is one of the key activities. The annual |
| :---- |

### Prompt for judging passage support

| You are checking whether a statement from an explanation is supported by a source passage. The passage may end mid-sentence or mid-word.   SOURCE PASSAGE:   {{source}}   FULL EXPLANATION:   {{explanation}}   STATEMENT:   {{statement}}   Classify the statement as:   - Supported: Every substantive claim about the observed passage is supported.   - Unsupported: At least one substantive claim about the observed passage is contradicted by, or absent from, the passage.   - Illegible: The statement is incoherent and cannot be judged reliably.   Answer with only one label (Supported, Unsupported, or Illegible) and nothing else. |
| :---- |


### Surgical corrections to AV explanations

However, the lower FVE of Sonnet explanations after RL may be the result of co-adaptation between the AV and AR, rather than evidence that passage support does not matter. Sonnet explanations differ from AV explanations in style, and the AR is trained only on the AV's style during RL. To reduce differences in style, we surgically correct passage-inconsistent claims in the AV's own explanations while preserving their wording and structure and leaving continuation predictions unchanged. We then compare reconstruction before and after editing.

We use the official Gemma-3-12B NLA at layer 32 on 1,573 Alpaca prompts. For 200 of these examples, we ask GPT-5.6 Sol to edit the AV explanation against the passage (full prompt in [Prompt for editing explanations](#prompt-for-editing-explanations)). The editor makes surgical edits on confabulations, and predictions about the next tokens are left unchanged. We pass the original and edited explanations through the AR and compare their FVE.

Correcting factual errors has little average effect on reconstruction, but the effect varies across subsets. Across all 200 examples, the change in FVE is −0.2 percentage points. We find that the AR depends on the explanation's description of the passage's final token, consistent with this [blogpost](https://www.lesswrong.com/posts/ahu9BCHPFDg9AJF75/some-observations-about-nla-explanations). When the editor kept the explanation's description of the passage's final token, FVE increased by 3.1 percentage points. When the editor changed it, FVE decreased, even when the AV's token was wrong (−5.4 percentage points). When we exclude white-space and punctuation tokens, the edits increase FVE by 2.4 percentage points, but this is still lower than GCG on the same examples (8.4 percentage points when β=0.01).

#### Prompt for editing explanations

| You are surgically correcting an explanation that describes a passage. The passage may end mid-sentence or mid-word. Copy the explanation exactly, changing only specific words or phrases that conflict with the passage. Claims about quotations, exact wording, and the observed final token must accurately match the passage. When correcting a quotation, replace it with a verbatim passage span containing the same number of whitespace-delimited words as the original quotation. Leave predictions about text after the passage unchanged. {3 few-shot examples} Now correct the following explanation PASSAGE: {passage} ORIGINAL EXPLANATION: {explanation} |
| :---- |

## Acknowledgments

We thank CoreWeave for the compute that enabled this research. Work done during the [SOAR program](https://www.eleuther.ai/soar).
