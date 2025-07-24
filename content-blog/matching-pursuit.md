---
title: "Matching pursuit SAEs are less interpretable than regular SAEs"
date: 2025-07-23T22:00:00-00:00
description: "Applying matching pursuit to sparse autoencoders and transcoders"
author: ["Stepan Shabalin", "Gonçalo Paulo"]
ShowToc: true
mathjax: true
draft: false
---


## Introduction

[Matching pursuit](https://en.wikipedia.org/wiki/Matching_pursuit) (MP) is a well-known algorithm for decomposing vectors into a sum of sparse components from a dictionary. MP and its variants (specifically, gradient pursuit) have been applied to pre-trained sparse autoencoder (SAE) dictionaries to improve reconstruction quality (see [Smith 2024](https://www.alignmentforum.org/posts/C5KAZQib3bzzpeyrg/progress-update-1-from-the-gdm-mech-interp-team-full-update#Replacing_SAE_Encoders_with_Inference_Time_Optimisation), [Engels et al. 2024](https://arxiv.org/abs/2410.14670)).

Recently, [Costa et al.](https://arxiv.org/abs/2506.03093) published a paper applying MP to sparse transcoders during training. They found that MP significantly improves the reconstruction quality of sparse transcoders, and that it allows the encoder to express complex correlation structures between latents. There is an open source implementation by the authors at [https://github.com/mpsae/MP-SAE](https://github.com/mpsae/MP-SAE).

In this blog post, we will replicate matching pursuit with [sparsify](https://github.com/EleutherAI/sparsify/tree/mp-sae) and apply it to sparse autoencoders and transcoders.

## Methods and results
We trained sparse autoencoders and transcoders on [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) and [Llama 3.2 1B](https://huggingface.co/meta-llama/Llama-3.2-1B), on layers 10, 15, 20 and 4, 8, 12 respectively. Unlike Costa et al., we untie the encoder and decoder weight matrices and use the decoder dictionary to subtract from the input vector. We use Adam with a learning rate of 1e-3, 100 warmup steps and a batch size of $2^{16}$ tokens. We otherwise follow default settings of sparsify.

These are the results we got for the residual stream of SmolLM2. Blue: non-MP. Red: MP.

<img src="/images/blog/matching-pursuit/runtime_fvu/base_layers.10_ef64.png" width="48%" style="display: inline-block"/>
<img src="/images/blog/matching-pursuit/runtime_fvu/base_layers.15_ef64.png" width="48%" style="display: inline-block"/>
<img src="/images/blog/matching-pursuit/runtime_fvu/base_layers.20_ef64.png" width="48%" style="display: inline-block"/>
<img src="/images/blog/matching-pursuit/runtime_fvu/base_layers.10_ef128.png" width="48%" style="display: inline-block"/>
<img src="/images/blog/matching-pursuit/runtime_fvu/base_layers.15_ef128.png" width="48%" style="display: inline-block"/>
<img src="/images/blog/matching-pursuit/runtime_fvu/base_layers.20_ef128.png" width="48%" style="display: inline-block"/>

It can be seen that for k=16,32 MP underperforms, but at k=64 and higher the FVU becomes lower than or comparable to the corresponding non-MP SAE.

We test several variations on this architecture. They are:

* ITO ([Smith 2024](https://www.alignmentforum.org/posts/C5KAZQib3bzzpeyrg/progress-update-1-from-the-gdm-mech-interp-team-full-update#Replacing_SAE_Encoders_with_Inference_Time_Optimisation)) - replace the SAE encoder with a maching pursuit-like algorithm.
* Encoder/decoder slicing - for each of the k active latents, use a unique slice of the encoder/decoder weight matrices. This means the time to compute the encoder forward pass is equal to the time taken by a regular SAE instead of being multiplied by k.
* Big decoder - like slicing, but the size of the slice is equal to the size of the original encoder, and the encoder is unchanged. This means there is a unique decoder of the same size for each latent.

<img src="/images/blog/matching-pursuit/mp_ablation_fvu/layers.10.png" width="48%" style="display: inline-block"/>
<img src="/images/blog/matching-pursuit/mp_ablation_fvu/layers.15.png" width="48%" style="display: inline-block"/>
<img src="/images/blog/matching-pursuit/mp_ablation_fvu/layers.20.png" width="48%" style="display: inline-block"/>

The ITO architecture has worse FVU on all layers we tested, while big decoder and encoder/decoder slicing perform similarly to the standard MP SAE. On Llama 3.2 1B, MP SAEs outperform at all hookpoints:

![Llama SAE FVU](/images/blog/matching-pursuit/mp_llama_ablation_fvu/layers.12.mlp.png)

On SmolLM MLPs, results are also clearly in favor of MP:

<img src="/images/blog/matching-pursuit/runtime_fvu/base_layers.10.mlp_ef64.png" width="48%" style="display: inline-block"/>
<img src="/images/blog/matching-pursuit/runtime_fvu/base_layers.15.mlp_ef64.png" width="48%" style="display: inline-block"/>
<img src="/images/blog/matching-pursuit/runtime_fvu/base_layers.20.mlp_ef64.png" width="48%" style="display: inline-block"/>
<img src="/images/blog/matching-pursuit/runtime_fvu/base_layers.10.mlp_ef128.png" width="48%" style="display: inline-block"/>
<img src="/images/blog/matching-pursuit/runtime_fvu/base_layers.15.mlp_ef128.png" width="48%" style="display: inline-block"/>
<img src="/images/blog/matching-pursuit/runtime_fvu/base_layers.20.mlp_ef128.png" width="48%" style="display: inline-block"/>

### Autointerp

Strikingly, MP SAEs perform much worse on autointerpretability metrics (computed with [delphi](https://github.com/EleutherAI/delphi), 500 latents on 1 million tokens of [fineweb-edu-dedup-10b](https://huggingface.co/datasets/EleutherAI/fineweb-edu-dedup-10b)).

<img src="/images/blog/matching-pursuit/autointerp_comparison/sae-k128-ef64_mp-sae-k128-ef64_fuzz.png" width="48%" style="display: inline-block"/>
<img src="/images/blog/matching-pursuit/autointerp_comparison/sae-k64-ef32_mp-sae-k64-ef32_fuzz.png" width="48%" style="display: inline-block"/>

<img src="/images/blog/matching-pursuit/autointerp_comparison/sae-k128-ef64_mp-sae-k128-ef64_detection.png" width="48%" style="display: inline-block"/>
<img src="/images/blog/matching-pursuit/autointerp_comparison/sae-k64-ef32_mp-sae-k64-ef32_detection.png" width="48%" style="display: inline-block"/>

Specifically, there are many latents with very low autointerpretability. Judging from manual inspection, they do not seem to be particularly meaningful. There are also many dead latents for all of the SAEs we trained, potentially confounding the results:

![](/images/blog/matching-pursuit/dead_pct_fvu/base_layers.10_k64.png)



### Transcoders

There is no straightforward way to apply matching pursuit to transcoders. The architecture relies on it being possible to subtract a row from the input vector and add the same row to the output vector; if the objective is not straightforward reconstruction, this means assuming the function being learned is linear.

We have not been able to improve on the performance of vanilla [skip transcoders](https://arxiv.org/abs/2501.18823) with matching pursuit-based transcoders. Our best-performing architecture was as follows:

* The encoder dictionary is normalized to have unit norm.
* We use the encoder dictionary like in Costa et al.: it is used for determining the next latent, its activation strength, and the vector that is subtracted from the input.
* The decoder dictionary is learned separately and is used at the end to determine the output.
* The encoder and decoder are sliced like in the previous section's ablation study.

We tested ITO with SSTs as well, and it was also not an improvement. Other things that did not work:

* Not normalizing the encoder dictionary
* Using a separate learned decoder for subtracting from the input
* Using the primary decoder for subtracting from the input
* ITO (with otherwise default settings)

<img src="/images/blog/matching-pursuit/mp_sst_ablation_fvu/layers.10.mlp.png" width="48%" style="display: inline-block"/>
<img src="/images/blog/matching-pursuit/mp_sst_ablation_fvu/layers.15.mlp.png" width="48%" style="display: inline-block"/>
<img src="/images/blog/matching-pursuit/mp_sst_ablation_fvu/layers.20.mlp.png" width="48%" style="display: inline-block"/>

## Conclusion

Overall, **we did not find matching pursuit to be an improvement to sparse autoencoders or transcoders**. While they sometimes improve reconstruction quality, they seem to be significantly less interpretable than regular SAEs, which is a major drawback.