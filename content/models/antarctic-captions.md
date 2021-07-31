---
title: "Antarctic Captions"
date: 2019-04-26T20:18:54+03:00
cover:
    image: ../../images/art50.png
    relative: True
layout: page
hideMeta: True
modality: Image to Text
intended use: Captioning images
license: MIT
training data: Microsoft COCO dataset
metrics: The final model was chosen by CIDEr score on the COCO dev set.
limitations and biases: Due to the high reliance of CLIP, the model inherits most of its limitations and biases, as described in their model card. COCO captions used for fine-tuning often contain labelling biases, such as annotators attempting to infer unknown attributes from the image context.
links: 
    github: https://github.com/dzryk/antarctic-captions
    demo: https://colab.research.google.com/drive/1FwGEVKXvmpeMvAYqGr4z7Nt3llaZz-F8
point of contact: Jamie Kiros (kirosjamie@gmail)
affiliations: Multimodal
---

A model that inputs an image and generates multiple captions. It combines CLIP, BART and a cache of text to retrieve from. An input image is mapped into CLIP space and scored against the cache to retrieve a collection of n-grams. The n-grams are passed to BART which generates captions. The candidate captions are then re-scored using CLIP. The layernorm parameters of BART's encoder are fine-tuned on COCO, while all other parameters are kept frozen. A key goal for this project is to be able to generate reasonable captions on a wide distribution of images well beyond what is available in standard captioning datasets.


Several extensions are in consideration, proposed by myself as well as other EAI members. This includes (1) building a massive cache and implementing approximate search (2) fine-tuning on other datasets (3) extending the model to other image -> text tasks, such as VQA (4) explore whether there is benefit from harnessing larger LMs.
