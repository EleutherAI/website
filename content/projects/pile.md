---
title: "The Pile"
date: 2019-04-26T20:18:54+03:00
layout: page
---

##  ## {class="content-block"}
- ![alt](../../images/art43.png)
- ## The Pile
    #### An 800GB Dataset of Diverse Text for Language Modeling
    The Pile is a large, diverse, open source language modelling data set that consists of many smaller datasets combined together. The objective is to obtain text from as many modalities as possible to ensure that models trained using The Pile will have much broader generalization abilities.

    # [Pile Paper (arXiv)](https://arxiv.org/abs/2101.00027) {class="button center"}

##  ## {class="content-block"}
- ## Download {class="text-title"}
    The Pile is hosted by [the Eye](https://the-eye.eu/).
    # [Download](https://the-eye.eu/public/AI/pile/) {class="button unfilled"}
    The format of the Pile is [jsonlines](https://jsonlines.org/) data compressed using [zstandard](https://facebook.github.io/zstd/).

    Have a model that uses or evaluates on the Pile? [Let us know!](mailto:contact@eleuther.ai)

    ## Why is the Pile a good training set? {class="text-title"}
    Recent work has shown that especially for large models, diversity in data sources improves general cross-domain knowledge of the model, as well as downstream generalization capability. In our evaluations, not only do models trained on the Pile show moderate improvements in traditional language modeling benchmarks, they also show significant improvements on Pile BPB. 

    ## Why is the Pile a good benchmark? {class="text-title"}
    To score well on Pile BPB (bits per byte), a model must be able to understand many disparate domains including books, github repositories, webpages, chat logs, and medical, physics, math, computer science, and philosophy papers. Pile BPB is a measure of world knowledge and reasoning ability in these domains, making it a robust benchmark of general, cross-domain text modeling ability for large language models. 

    ## Citing {class="text-title"}
    If you use the Pile or any of the components, please cite us! 
    ```
    @article{pile,
        title={The {P}ile: An 800GB Dataset of Diverse Text for Language Modeling},
        author={Gao, Leo and Biderman, Stella and Black, Sid and Golding, Laurence and Hoppe, Travis and Foster, Charles and Phang, Jason and He, Horace and Thite, Anish and Nabeshima, Noa and Presser, Shawn and Leahy, Connor},
        journal={arXiv preprint arXiv:2101.00027},
        year={2020}
    }
    ```
- ## Leaderboard {class="text-title" style="min-width:280px"}
    \* indicates potential test-set overlap. Zero-shot indicates that not all of the components of the Pile were present in the training data. 
    | Rank 	| Model	| Test BPB |
    | :---:   | :---:   | :---:   |
    | 1.    | GPT-3 (Zero-Shot)* | 0.7177 |
    | 2.    | GPT-2 (Zero-Shot)* | 1.2253 |

    ##### [Evaluation Code](https://github.com/EleutherAI/lm_perplexity) {class="center"}

<!-- ## The Pile is now live! ## {class="text-announcement"}
[Download now](https://pile.eleuther.ai/), or you can [read the docs](https://pile.eleuther.ai/paper.pdf) -->

