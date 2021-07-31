---
title: "GPT-J"
date: 2019-04-26T20:18:54+03:00
cover:
    image: ../../images/art50.png
    relative: True
layout: page
hideMeta: True
modality: Text
intended use: Text generation
license: Apache 2.0
training data: The Pile
metrics: GPT-J was trained using validation loss on the Pile, as well as LAMBADA, Winogrande, Hellaswag, and PIQA
limitations and biases: GPT-J was trained as an autoregressive language model. This means that its core functionality is taking a string of text and predicting the next token. While language models are widely used for tasks other than this, there are a lot of unknowns with this work. When prompting GPT-J with er that the statistically most likely next token is often not the same thing as the most accurate,  GPT-J was trained on the Pile, a dataset known to contain profanity, lewd, and otherwise abrasive language. Depending on your usecase GPT-J may produce socially unacceptable text. See Sections 5 and 6 of the Pile paper for a more detailed analysis of the biases in the Pile. As with all language models, it is hard to predict in advance how GPT-J will respond to particular prompts and offensive content may occur without warning. We recommend having a human curate or filter the outputs before releasing them, both to censor undesirable content and to improve the quality of the results. 

links: 
    github: https://github.com/kingoflolz/mesh-transformer-jax
    demo: https://colab.research.google.com/github/kingoflolz/mesh-transformer-jax/blob/master/colab_demo.ipynb
point of contact: Ben Wang (@kindiana#1016)
affiliations: Massive Language Models
---


GPT-J 6B is a transformer model designed using Ben Wang's [Jax implementation of the GPT-3 architecture](https://github.com/kingoflolz/mesh-transformer-jax/). GPT-J refers to the class of models, while 6B represents the number of parameters of this particular pre-trained model.

| Hyperparameter    | Value  | 
|-------------------|--------|
| n_parameters      | 6,053,381,344 |
| n_layers          | 28*    |
| d_model           | 4,096  |
| d_ff              | 16,384 |
| n_heads           | 16     |
| d_head            | 256    |
| n_ctx             | 2,048  |
| n_vocab           | 50,257 (same tokenizer as GPT-2/3)  |
| position encoding | [Rotary position encodings (RoPE)](https://arxiv.org/abs/2104.09864) |
| RoPE dimensions   | [64](https://github.com/kingoflolz/mesh-transformer-jax/blob/f2aa66e0925de6593dcbb70e72399b97b4130482/mesh_transformer/layers.py#L223) |

`*` each layer consists of one feedforward block and one self attention block

The model consists of 28 layers with a model dimension of 4096, and a feedforward dimension of 16384. The model
dimension is split into 16 heads, each with a dimension of 256. Rotary position encodings (RoPE) was applied to 64
dimensions of each head. The model is trained with a tokenization vocabulary of 50257, using the same set of BPEs as
GPT-2/GPT-3.
