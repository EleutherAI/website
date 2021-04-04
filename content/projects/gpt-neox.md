---
title: "GPT-NeoX"
date: 2019-04-26T20:18:54+03:00
layout: project-page
---

## {class="content-block"}
- ![alt](../../images/art50.png)
- ## GPT-NeoX
    GPT&#8288;-&#8288;Neo is the code name for a family of transformer-based language models loosely styled around the GPT architecture. Our primary goal is to replicate a GPT&#8288;-&#8288;3&nbsp;DaVinci-sized model and open-source it to the public, for free.

    [GPT&#8288;-&#8288;NeoX](https://github.com/EleutherAI/gpt-neox) is an implementation of 3D-parallel GPT&#8288;-&#8288;3-like models on distributed GPUs, based upon [DeepSpeed](https://www.deepspeed.ai/) and [Megatron&#8288;-&#8288;LM](https://github.com/NVIDIA/Megatron-LM).

    We have been graciously been offered high-performance GPU compute by [CoreWeave](https://www.coreweave.com/), an NVIDIA Preferred Cloud Services Provider. CoreWeave is excited by the open nature of the project and is very keen in helping us to break the OpenAI-Microsoft monopoly on massive autoregressive language models.


## {class="content-block"}
- ### Progress:
    - As of {{<date year="2021" month="03" day="31">}}, the codebase is fairly stable. DeepSpeed, 3D-parallelism and ZeRO are all working properly.
- ### Next Steps:
    - We are currently waiting for CoreWeave to finish building the final hardware we'll be training on. In the meantime, we are optimizing GPT&#8288;-&#8288;NeoX to run as efficiently as possible on that hardware.

