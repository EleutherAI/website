---
title: "GPT-Neo"
date: 2019-04-26T20:18:54+03:00
project_image: "images/the-pile.png"
layout: project-page
---

## {class="content-block"}
- ![alt](../../images/art49.png)
- ## GPT-Neo 
    GPT&#8288;-&#8288;Neo is the code name for a family of transformer-based language models loosely styled around the GPT architecture. Our primary goal is to replicate a GPT&#8288;-&#8288;3&nbsp;DaVinci-sized model and open-source it to the public, for free.

    [GPT&#8288;-&#8288;Neo](https://github.com/EleutherAI/gpt-neo) is an implementation of model & data-parallel GPT&#8288;-&#8288;2 and GPT&#8288;-&#8288;3-like models, utilizing [Mesh&nbsp;Tensorflow](https://github.com/tensorflow/mesh) for distributed support. This codebase is optimized for TPUs, but should also work on GPUs.

    Along the way we will be running experiments with [alternative](https://arxiv.org/abs/1701.06538) [architectures](https://arxiv.org/abs/1911.03864) and [attention](https://arxiv.org/abs/2006.16236) [types](https://www.aclweb.org/anthology/2020.acl-main.672.pdf), releasing any intermediate models, and writing up any findings on our blog.


## {class="content-block"}
- ### Progress:
    - GPT&#8288;-&#8288;Neo should be feature complete. We are making bugfixes, but we do not expect to make significant changes. 
    - As of {{<date year="2021" month="03" day="21">}}, 1.3B and 2.7B parameter GPT&#8288;-&#8288;Neo models are available to be run with [GPT&#8288;-&#8288;Neo](https://github.com/EleutherAI/gpt-neo).
    - As of {{<date year="2021" month="03" day="31">}}, 1.3B and 2.7B parameter GPT&#8288;-&#8288;Neo models are [now available on Hugging Face Model Hub](https://huggingface.co/EleutherAI)!
- ### Next Steps:
    - We continue our efforts in in our GPU codebase, [GPT&#8288;-&#8288;NeoX](/projects/gpt-neox/).

