---
title: "Gpt-Neo"
date: 2019-04-26T20:18:54+03:00
project_image: "images/the-pile.png"
layout: project-page
---

## {class="content-block"}
- ![alt](../../images/art49.png)
- ## GPT-Neo 
    GPT-Neo is the code name for a series of transformer-based language models loosely styled around the GPT architecture that we plan to train and open source. Our primary goal is to replicate a GPT-3 sized model and open source it to the public, for free.

    Along the way we will be running experiments with [alternative](https://arxiv.org/abs/1701.06538) [architectures](https://arxiv.org/abs/1911.03864) and [attention](https://arxiv.org/abs/2006.16236) [types](https://www.aclweb.org/anthology/2020.acl-main.672.pdf), releasing any intermediate models, and writing up any findings on our blog.

    We have two repositories in development, [GPT-Neo](https://github.com/EleutherAI/gpt-neo/) (for training on TPUs) and [GPT-NeoX](https://github.com/EleutherAI/gpt-neox/) (for training on GPUs). 


## {class="content-block"}
- ### Progress:
    - We are now focusing development on the GPT-NeoX library, as cloud compute provider [Coreweave](https://coreweave.com/) has offered to provide us the GPUs necessary to train a GPT-3 scale language model! Our devs' mental health will greatly benefit from moving the codebase over to pytorch.
    - We have released two mid-sized (1.3B and 2.7B parameter) models in our [GPT-Neo library](https://github.com/EleutherAI/gpt-neo#pretrained-models)/
    - GPT-Neo models are [now available on HuggingFace](https://github.com/huggingface/transformers/pull/10848)!
- ### Next Steps:
    - We are currently waiting for Coreweave to finish building the final hardware we'll be training on, and in the meantime, are optimizing GPT-NeoX to run as efficiently as possible on said hardware.

