---
title: "CLASP - Contrastive Language-Amino Acid Sequence Pretraining"
date: 2021-07-08T20:18:54+03:00
layout: page
cover:
    image: ../../images/art4.png
    relative: True
hideMeta: True
status: "In progress"
domain: ["Bio ML"]
deliverables: ["Code","model"]
description: A CLIP-like model for amino acid sequence prediction.
---

Recently multimodal contrastive models have had an explosion in power and popularity, e.g., [ConVIRT](https://arxiv.org/abs/2010.00747), [CLIP](https://openai.com/blog/clip/), and [ALIGN](https://arxiv.org/abs/2102.05918). In this project we apply a similar setup but use amino acid sequences and their language description as our training data from the Universal Protein Resource ([UniProt](https://www.uniprot.org/)), an annotated protein database. The goal is to create a model that can be used like other CLIP-like models but for amino acid sequences and text.

**GitHub Repo:** https://github.com/MicPie/clasp
