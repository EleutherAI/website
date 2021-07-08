---
title: "GPT-Neo"
date: 2019-04-26T20:18:54+03:00
cover:
    image: ../../images/art49.png
    relative: True
layout: page
hideMeta: True
status: "Completed"
domain: ["Language Modeling"]
deliverables: ["Code","Model"]
description: An implementation of model & data-parallel GPT-2 and GPT-3-like models with Mesh Tensorflow.
---

[GPT-Neo](https://github.com/EleutherAI/gpt-neo) is an implementation of model & data-parallel GPT-2 and GPT-3-like models, utilizing [Mesh Tensorflow](https://github.com/tensorflow/mesh) for distributed support. This codebase is designed for TPUs. It should also work on GPUs, though we do not recommend this hardware configuration.

## Progress:
- GPT-Neo should be feature complete. We are making bugfixes, but we do not expect to make any significant changes. 
- As of {{<date year="2021" month="03" day="21">}}, 1.3B and 2.7B parameter GPT-Neo models are available to be run with [GPT-Neo](https://github.com/EleutherAI/gpt-neo).
- As of {{<date year="2021" month="03" day="31">}}, 1.3B and 2.7B parameter GPT-Neo models are [now available on Hugging Face Model Hub](https://huggingface.co/EleutherAI)!


## Next Steps:
- We continue our efforts in in our GPU codebase, [GPT-NeoX](/projects/gpt-neox/).

