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

![]
