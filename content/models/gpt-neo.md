---
title: "GPT-Neo"
date: 2019-04-26T20:18:54+03:00
cover:
    image: ../../images/art50.png
    relative: True
layout: page
hideMeta: True
modality: Text
intended use: Generatin
license: Apache 2.0
training data: The Pile
metrics: GPT-Neo was trained using validation loss on the Pile.
limitations and biases: GPT-Neo was trained as an autoregressive language model. This means that its core functionality is taking a string of text and predicting the next token. While language models are widely used for tasks other than this, there are a lot of unknowns with this work. GPT-Neo was trained on the Pile, a dataset known to contain profanity, lewd, and otherwise abrasive language. Depending on your usecase GPT-Neo may produce socially unacceptable text. See Sections 5 and 6 of the Pile paper for a more detailed analysis of the biases in the Pile. As with all language models, it is hard to predict in advance how GPT-Neo will respond to particular prompts and offensive content may occur without warning. We recommend having a human curate or filter the outputs before releasing them, both to censor undesirable content and to improve the quality of the results. 
links: 
    github: https://github.com/EleutherAI/gpt-neo
    demo: https://colab.research.google.com/github/EleutherAI/GPTNeo/blob/master/GPTNeo_example_notebook.ipynb
point of contact: Sid Black (@Sid#2121)
affiliations: Massive Language Models
---
