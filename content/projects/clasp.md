---
title: "CLASP - Contrastive Language-Amino Acid Sequence Pretraining"
date: 2021-07-08T20:18:54+03:00
layout: page
cover:
    image: ../../images/art4.png
    relative: True
hideMeta: True
status: "Completed"
domain: ["Bio ML"]
deliverables: ["Code"]
description: A CLIP-like model for amino acid sequence prediction.
---

Recently multimodal contrastive generative models have had an explosion in power and popularity. Most of these models follow the high-level approach of [OpenAI's CLIP](https://openai.com/blog/clip/), sometimes replacing the images with data from a different modality. In this project we replace them with amino acid sequences and take our training data from the Universal Protein Resource (UnitProt), an annotated protein database. The goal is to create a generative model that takes natural language descriptions of proteins as an input and returns an amino acid sequence that codes for a protein with the requested properties.

**GitHub Repo:** https://github.com/MicPie/clasp
