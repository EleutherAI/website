---
title: "Gpt-NeoX"
date: 2019-04-26T20:18:54+03:00
layout: project-page
---

## {class="content-block"}
- ![alt](../../images/art50.png)
- ## GPT-NeoX
    GPT-Neo is the code name for a series of transformer-based language models loosely styled around the GPT architecture that we plan to train and open source. Our primary goal is to replicate a GPT-3 sized model and open source it to the public, for free.

    NeoX is an implementation of model parallel GPT-3-like models on GPUs, based loosely around the DeepSpeed framework and Megatron Library. Designed to be able to train models in the hundreds of billions of parameters or larger.

    Along the way we will be running experiments with [alternative](https://arxiv.org/abs/1701.06538) [architectures](https://arxiv.org/abs/1911.03864) and [attention](https://arxiv.org/abs/2006.16236) [types](https://www.aclweb.org/anthology/2020.acl-main.672.pdf), releasing any intermediate models, and writing up any findings on our blog.

    We have two repositories in development, [GPT-Neo](https://github.com/EleutherAI/gpt-neo/) (for training on TPUs) and [GPT-NeoX](https://github.com/EleutherAI/gpt-neox/) (for training on GPUs). 


## {class="content-block"}
- ### Progress:
    - The codebase (as of 29/03/2021) is fairly stable. Training with deepspeed, 3D parallelism and ZeRO are all working nicely. We are currently optimizing the performance as much as possible whilst we await the requisite hardware for training.
- ### Next Steps:
    - We are currently waiting for Coreweave to finish building the final hardware we'll be training on, and in the meantime, are optimizing GPT-NeoX to run as efficiently as possible on said hardware.

