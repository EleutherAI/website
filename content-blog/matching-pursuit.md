---
title: "Matching Pursuit SAEs replication"
date: 2025-05-30T22:00:00-00:00
description: "Applying matching pursuit to sparse autoencoders and transcoders"
author: ["Stepan Shabalin", "Gonçalo Paulo"]
ShowToc: true
mathjax: true
draft: false
---


## Introduction

[Matching pursuit](https://en.wikipedia.org/wiki/Matching_pursuit) (MP) is a well-known algorithm for decomposing vectors into a sum of sparse components from a dictionary. MP and its variants (specifically, gradient pursuit) have been applied to pre-trained sparse autoencoder (SAE) dictionaries to improve reconstruction quality (see [Smith 2024](https://www.alignmentforum.org/posts/C5KAZQib3bzzpeyrg/progress-update-1-from-the-gdm-mech-interp-team-full-update#Replacing_SAE_Encoders_with_Inference_Time_Optimisation), [Engels et al. 2024](https://arxiv.org/abs/2410.14670)).

Recently, [Costa et al.](https://arxiv.org/abs/2506.03093) published a paper applying MP to sparse transcoders during training. They found that MP significantly improves the reconstruction quality of sparse transcoders, and that it allows the encoder to express complex correlation structures between latents.

In this blog post, we will replicate matching pursuit with [sparsify](https://github.com/EleutherAI/sparsify) and apply it to sparse autoencoders and transcoders.

## FVU results

Blue: non-MP. Red: MP.

![](/images/blog/matching-pursuit/runtime_fvu/base_layers.10_k64.png)
![](/images/blog/matching-pursuit/runtime_fvu/base_layers.10_k128.png)
![](/images/blog/matching-pursuit/runtime_fvu/base_layers.15_k64.png)
![](/images/blog/matching-pursuit/runtime_fvu/base_layers.15_k128.png)
![](/images/blog/matching-pursuit/runtime_fvu/base_layers.20_k64.png)
![](/images/blog/matching-pursuit/runtime_fvu/base_layers.20_k128.png)
![](/images/blog/matching-pursuit/runtime_fvu/base_layers.10.mlp_k64.png)
![](/images/blog/matching-pursuit/runtime_fvu/base_layers.10.mlp_k128.png)
![](/images/blog/matching-pursuit/runtime_fvu/base_layers.15.mlp_k64.png)
![](/images/blog/matching-pursuit/runtime_fvu/base_layers.15.mlp_k128.png)
![](/images/blog/matching-pursuit/runtime_fvu/base_layers.20.mlp_k64.png)
![](/images/blog/matching-pursuit/runtime_fvu/base_layers.20.mlp_k128.png)

Llama 3.1 1B:
![](/images/blog/matching-pursuit/mp_llama_ablation_fvu/layers.12.mlp.png)

Ablations:
MP, layer 10:
![](/images/blog/matching-pursuit/mp_ablation_fvu/layers.10.png)
MP, layer 15:
![](/images/blog/matching-pursuit/mp_ablation_fvu/layers.15.png)
MP, layer 20:
![](/images/blog/matching-pursuit/mp_ablation_fvu/layers.20.png)

MP SST, layer 10 (MLP):
![](/images/blog/matching-pursuit/mp_sst_ablation_fvu/layers.10.mlp.png)
MP SST, layer 15 (MLP):
![](/images/blog/matching-pursuit/mp_sst_ablation_fvu/layers.15.mlp.png)
MP SST, layer 20 (MLP):
![](/images/blog/matching-pursuit/mp_sst_ablation_fvu/layers.20.mlp.png)

