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

    Our models are built in Tensorflow-mesh, which will allow us to scale up to GPT-3 sizes and beyond using simultaneous model and data parallelism.


## {class="content-block"}
- ### Progress:
    - We have the bulk of the model built, GPT-2 size models trained, and several experimental architectures implemented.

    - Our current codebase should be able to scale up to GPT-3 sized models

- ### Next Steps:
    - We are currently working on wrapping up GPT-2-sized model replication, looking mostly at evaluations there.

    - The largest model we've gotten to train for a single step so far has been 200B parameters.

