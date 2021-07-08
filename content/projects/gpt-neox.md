---
title: "GPT-NeoX"
date: 2019-04-26T20:18:54+03:00
cover:
    image: ../../images/art50.png
    relative: True
layout: page
hideMeta: True
status: "In Progress"
domain: "Language Modeling"
deliverables: ["Code","Model"]
description: An implementation of 3D-parallel GPT⁠-⁠3-like models for distributed GPUs.
---

[GPT-NeoX](https://github.com/EleutherAI/gpt-neox) is an implementation of 3D-parallel GPT-3-like models on distributed GPUs, based upon [DeepSpeed](https://www.deepspeed.ai/) and [Megatron-LM](https://github.com/NVIDIA/Megatron-LM).


Progress:
: As of {{<date year="2021" month="03" day="31">}}, the codebase is fairly stable. DeepSpeed, 3D-parallelism and ZeRO are all working properly.

Next Steps:
: We are currently waiting for CoreWeave to finish building the final hardware we'll be training on. In the meantime, we are optimizing GPT-NeoX to run as efficiently as possible on that hardware.

