---
title: "CLIP Decision Transformer"
date: 2019-04-26T20:18:54+03:00
cover:
    image: ../../images/art50.png
    relative: True
layout: page
hideMeta: True
modality: Text
intended use: Text generation
license: Apache 2.0
training data: Google Conceptual Captions
metrics: CLIP similarity
limitations and biases:
links: 
    github: https://github.com/EleutherAI/gpt-neox
    demo: https://colab.research.google.com/drive/1dFV3GCR5kasYiAl8Bl4fBlLOCdCfjufI
point of contact: Katherine Crowson (@alstroemeria313#1694)
affiliations: Multimodal
---

A 337M parameter autoregressive transformer model intended for sampling sequences of VQGAN tokens conditioned on a CLIP text embedding and desired CLIP similarity. It was trained on Google Conceptual Captions and produces 384x384 output images from a text prompt.
