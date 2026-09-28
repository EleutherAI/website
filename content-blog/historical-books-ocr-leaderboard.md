---
title: "FineBooks: are open OCR models good enough to unlock historical knowledge?"
date: 2026-08-10
author: ["Sebastian Majstorovic", "Daniel van Strien"]
description: "Testing open OCR models on historical books with the FineBooks BHL OCR Leaderboard."
draft: false
---

*Originally published on [Hugging Face](https://huggingface.co/blog/finebooks/historical-books-ocr-leaderboard).*

Over the last few decades, libraries have digitized millions of historical books. Many of these are no longer protected by copyright and are now in the public domain. As part of this digitization process libraries used Optical Character Recognition (OCR) to extract the text from the scanned pages. In many cases, the OCR was performed only once, at the time of scanning, with whatever tool or pipeline was available to the library.

As technology has progressed, OCR models have improved dramatically, especially in the last few years with the advent of Visual Language Models (VLMs). Many of this new generation of VLM-based OCR models are published with open weights and under an open license and can therefore be used for free by anyone on any hardware or infrastructure. In most cases, however, these new OCR models have been primarily trained on and optimized for modern documents. Historical books are often considerably more complex and diverse than modern documents: many have unusual and complex layouts, were printed using archaic letterforms such as the [long s (ſ)](https://en.wikipedia.org/wiki/Long_s) and [ligatures](https://en.wikipedia.org/wiki/Ligature_(writing)), or were written in languages no longer in everyday use, e.g. Latin.

### Why do we need better OCR?

<figure>
<div style="display: flex; flex-wrap: wrap; gap: 16px; align-items: start;">
<a href="/images/blog/historical-books-ocr-leaderboard/butler-birds.webp" style="flex: 1 1 180px; min-width: 0;"><img src="/images/blog/historical-books-ocr-leaderboard/butler-birds.webp" alt="A page of printed text from Butler&#x27;s Birds of Great Britain and Ireland" loading="lazy" style="width: 100%;"></a>
<a href="/images/blog/historical-books-ocr-leaderboard/russian-entomological-society.webp" style="flex: 1 1 180px; min-width: 0;"><img src="/images/blog/historical-books-ocr-leaderboard/russian-entomological-society.webp" alt="A multilingual table of contents from the 1881 proceedings of the Russian Entomological Society" loading="lazy" style="width: 100%;"></a>
<a href="/images/blog/historical-books-ocr-leaderboard/engraved-plate.webp" style="flex: 1 1 180px; min-width: 0;"><img src="/images/blog/historical-books-ocr-leaderboard/engraved-plate.webp" alt="An engraved plate whose only text is a line of engraver&#x27;s credits" loading="lazy" style="width: 100%;"></a>
</div>
<figcaption>Three of the 2,165 ground-truth pages: single-column English natural history, a multilingual table of contents, and an engraved plate whose entire transcription is one line of credits.</figcaption>
</figure>

Better OCR is not just a convenience for readers and researchers. Public domain books are one of the largest sources of long-form text that can be freely used for training open AI models — but only if the text quality is good enough. The [Talkie](https://talkie-lm.com/introducing-talkie) project measured what quality is worth: a language model trained on OCR-derived text learned at 30% of the efficiency of the same model trained on human transcriptions of the same books. When EleutherAI and collaborators released [Common Pile](https://huggingface.co/blog/stellaathena/common-pile) last year – the largest openly licensed training corpus to date – it contained around 300,000 public domain books whose text came from earlier OCR pipelines. Re-processing those books with better OCR models is one of the highest-leverage improvements available for open source AI training datasets.

Whether the new OCR models can close the gap on historical documents is therefore not a niche question — and new models keep arriving, sometimes several in a month. This presents libraries and researchers with a hard question: are these new OCR models **accurate** enough, **fast** enough and **cheap** enough to run, so that it would be worth re-running OCR on the scanned images? In other words, are these OCR models ***good enough*** yet?

**FineBooks**, a new collaboration between Hugging Face and EleutherAI, has two clear goals. First, we want to test current open OCR models to determine whether they are good enough to unlock historical knowledge at scale. Second, we aim to re-process public domain book collections using leading models and publish the improved OCR text as open datasets. The first collection we have chosen to work on is the [Biodiversity Heritage Library (BHL)](https://www.biodiversitylibrary.org), a global collection of more than 300,000 digitized historical natural history documents containing more than [64 million pages of scientific knowledge](https://www.theguardian.com/environment/2026/jun/18/natural-world-digital-biodiversity-heritage-library-scientific-knowledge-free-access-aoe).

As a first step, we release today:

- The [**FineBooks BHL OCR Leaderboard**](https://huggingface.co/spaces/finebooks/bhl-ocr-leaderboard): the leaderboard scores 14 open OCR models against 2,165 historical book pages transcribed by a team of expert researchers. This is the main output of the first phase of the FineBooks project.
- [**finebooks/bhl-impact-gt**](https://huggingface.co/datasets/finebooks/bhl-impact-gt): the ground-truth dataset behind the leaderboard.
- [**finebooks/bhl-ocr-eval**](https://github.com/finebooks/bhl-ocr-eval): the evaluation harness, so anyone can verify a score or run the same measurement on their own material.

This post explains how the evaluation works and what it shows so far.

## The ground truth

Measuring OCR accuracy requires pages whose correct transcription is known. Producing such transcriptions requires an expert to type or check every character manually, which is laborious and difficult. Few of these expert transcriptions exist for historical books, but we found one project which had done this work and have built our evaluation around its results.

Between 2011 and 2012, the [IMPACT project and BHL-Europe produced expert-corrected transcriptions](https://web.archive.org/web/20220626135838/https://www.digitisation.eu/impact-bhl-europe-collaboration/) of six BHL volumes: 2,165 pages in English, French, German and Latin, transcribed to roughly one error in two thousand characters. Both projects ended years ago, their websites are no longer online, but the transcriptions were released under an open license and [preserved in a Github repository](https://github.com/impactcentre/groundtruth-bhl) by [IMPACT’s technical manager Clemens Neudecker](https://www.dataspace-culturalheritage.eu/en/news/professionals-in-focus-clemens-neudecker).

This example demonstrates how open datasets can yield long-term benefits, if we make sure to preserve them. When these transcriptions were made, none of the OCR models on our leaderboard existed yet, and most of their architectures hadn't been invented. Fourteen years later, the same 2,165 pages still afford a rigorous evaluation because the work was done carefully and released openly under a CC-BY license.

We also chose the BHL as the first FineBooks collection for a second reason: it publishes its entire collection as [bulk downloads via AWS Open Data](https://registry.opendata.aws/bhl-open-data/). We were therefore able to rebuild the IMPACT transcriptions into a modern ground-truth dataset, [finebooks/bhl-impact-gt](https://huggingface.co/datasets/finebooks/bhl-impact-gt), with every BHL scan image matched to its transcription.

## The evaluation

We evaluated 14 OCR models, all released with open weights under permissive licenses. Anyone can download these models and run them on their own hardware, without API keys, per-page charges, or restrictions on the re-use of the output.

Each model processed all 2,165 pages as a single [Hugging Face Job](https://huggingface.co/docs/hub/en/jobs). Hugging Face Jobs are suited well for this kind of evaluation:

- One command starts a GPU, runs the model over the dataset, and shuts down. There is no infrastructure to set up or maintain.
- Billing is per minute. A full 2,165-page run costs between a few cents and a few dollars per model.
- Every run pins the model revision, container image, and script commit. Any score on the board can be reproduced by re-submitting the same job, and adding a newly released model to the leaderboard is a single job submission.

## How OCR quality is measured

The standard metric for OCR quality is **Character Error Rate (CER)**: the number of character-level mistakes — substitutions, deletions, insertions — divided by the total characters in the correct text. A CER of 0.024 means 24 errors per thousand characters. Lower is better. In the results table of this blog post, we invert the CER into an **accuracy** percentage for better understanding: a CER of 0.024 means 97.6% of characters were recognised accurately.

A single metric hides important differences between models, so the leaderboard reports several additional metrics:

- **Two CER variants.** Historical print documents can contain archaic letterforms like the long-s “ſ”. The *diplomatic* CER variant counts a model that modernises an “ſ” to “s” as an error; the *reading* CER counts this modernisation as correct. Which one matters depends on whether you need a faithful scholarly transcription or a text readable by modern readers. The result table in this post uses the reading variant.
- **Recall**: of the words that should be on the page, how many did the model produce? A model can have a good CER while silently dropping footnotes.
- **Over-extraction**: how much of the output is not on the page. VLM-based models sometimes generate text on blank or illustration-only pages, so the leaderboard scores those pages as a separate category.
- **Loop rate**: the share of pages where a model repeated itself until it hit its token limit. Looped pages are excluded from that model's other scores, so a high loop rate makes the remaining numbers look better than they are. The two must be read together.

## Selected results

The full results are on the [leaderboard](https://huggingface.co/spaces/finebooks/bhl-ocr-leaderboard); this table shows a representative slice, ordered from highest to lowest accuracy based on the CER *reading* variant.

<div role="region" aria-label="OCR model results" tabindex="0" style="overflow-x: auto;">
<table style="min-width: 600px; white-space: nowrap;">
<thead><tr>
<th align="left">Model</th>
<th align="left">Size</th>
<th align="right">Accuracy (<em>reading</em> CER)</th>
<th align="right">Cost / 1,000 pages (using HF Jobs)</th>
</tr>
</thead><tbody><tr>
<td align="left"><a href="https://huggingface.co/dots-studio/dots.mocr">dots.mocr</a></td>
<td align="left">3B</td>
<td align="right">97.6%</td>
<td align="right">$1.94</td>
</tr>
<tr>
<td align="left"><a href="https://huggingface.co/ATH-MaaS/OvisOCR2">OvisOCR2</a></td>
<td align="left">0.9B</td>
<td align="right">96.9%</td>
<td align="right">$0.46</td>
</tr>
<tr>
<td align="left"><a href="https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6">PaddleOCR-VL-1.6</a></td>
<td align="left">1B</td>
<td align="right">96.1%</td>
<td align="right">$0.34</td>
</tr>
<tr>
<td align="left"><a href="https://huggingface.co/allenai/olmOCR-2-7B-1025-FP8">olmOCR-2</a></td>
<td align="left">8.3B</td>
<td align="right">95.7%</td>
<td align="right">$0.45</td>
</tr>
<tr>
<td align="left"><a href="https://huggingface.co/lightonai/LightOnOCR-2-1B">LightOnOCR-2</a></td>
<td align="left">1B</td>
<td align="right">95.1%</td>
<td align="right">$0.37</td>
</tr>
<tr>
<td align="left"><a href="https://huggingface.co/Qwen/Qwen3.5-9B">Qwen3.5-9B</a></td>
<td align="left">9.7B</td>
<td align="right">94.9%</td>
<td align="right">$0.89</td>
</tr>
<tr>
<td align="left"><a href="https://huggingface.co/deepseek-ai/DeepSeek-OCR">DeepSeek-OCR</a></td>
<td align="left">3.3B</td>
<td align="right">93.8%</td>
<td align="right">$0.37</td>
</tr>
</tbody>
</table>
</div>

## Are the open OCR models good enough?

Whether a model is good enough for historical OCR depends on what the text is used for, and our answer differs by use case:

- **Are they good enough for LLM training corpora like Common Pile?** In most cases, yes. What matters here is reading accuracy at scale, and the leading models make far fewer character errors than the earlier pipelines that produced the text contained in the current corpora. At the costs above, re-processing an entire collection the size of BHL's public domain holdings is a realistic project.
- **Are they good enough for libraries replacing legacy OCR?** It depends. One of the challenges with these new OCR models is what they output. Legacy pipelines usually produce [ALTO XML](https://en.wikipedia.org/wiki/Analyzed_Layout_and_Text_Object), a format that library systems rely on for word-level coordinates. The new models output Markdown or plain text, sometimes with text-region coordinates, but never word-level positions, so a library can't simply drop the new text into existing infrastructure built on ALTO XML. There is an open question for libraries about whether they need ALTO XML in all cases, but each institution will need to weigh it up for itself.
- **Are they good enough for scholarly transcription?** Not quite yet. The board scores this separately, and the gap is not that models misread the page, it is that they silently modernise the long-s (ſ), ligatures, and other text features a scholarly transcription must preserve. It is likely some focused fine-tuning could address these limitations in the models quite well.

## Scope and limitations

The scope of the leaderboard follows the scope of the ground truth, and we want to be explicit about both. The IMPACT transcriptions cover six volumes in four languages — English, French, German and Latin — all printed in antiqua-family typefaces. The evaluation therefore says nothing about different typefaces such as German [Fraktur](https://en.wikipedia.org/wiki/Fraktur), about non-Latin writing systems, non-European languages, or about handwriting.

Also, the restriction of the **FineBooks** project to book-like historical documents is deliberate. Books are visually more uniform than newspapers, magazines, or archival materials: they mostly contain single-column running text with consistent layouts and repeating furniture. Multi-column layouts, complex page elements, and other document types present different technical challenges, with different failure modes, and the current leaderboard does not measure them.

## What's next?

- **New models join the board as they are released**. Our pipeline can evaluate a new model for the leaderboard at the cost of a few dollars, so the board tracks the state of the art rather than snapshotting it.
- **Re-processing the entire BHL public domain corpus**. Roughly 200,000 of BHL's 300,000 items carry public domain rights statements. We will re-run OCR across them with one of the leading models and release the resulting text as the first FineBooks dataset. The point of creating the leaderboard first was to build the confidence that we can do this at scale.
- **Working in the open**. As we do this work, we will share the lessons we learn along the way, and openly release any supporting datasets and models we build, e.g. page classifiers, processing pipelines, and the evaluation data itself.

Follow the [Finebooks org](https://huggingface.co/finebooks), [Sebastian](https://huggingface.co/storytracer) and [Daniel](https://huggingface.co/davanstrien) for updates on this work!
