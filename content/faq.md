---
title: "FAQ"
date: 2021-02-26T20:18:54+03:00
layout: page
---

## FAQ  

### General

Q: *How did this all start?*

A: On July 3rd, 2020, [Connor Leahy](https://github.com/ConnorJL) posted in the TPU Podcast Discord:
> https://arxiv.org/abs/2006.16668
> 
> Hey guys lets give OpenAI a run for their money like the good ol' days

To which [Leo Gao](https://github.com/leogao2) replied:
> this but unironically

And so it began.

Q: *Where did the name come from?*

A: In Ancient Greek, [*eleutheria*](https://en.wikipedia.org/wiki/Eleutheria) is a word for "liberty", and was used as a proper noun as a personification of the concept. This same personage became [*Libertas*](https://en.wikipedia.org/wiki/Libertas) to the Romans and [*Lady Liberty*](https://en.wikipedia.org/wiki/Statue_of_Liberty) to Americans.

Q: *How can I get involved?*

A: Join our [Discord](https://www.google.com/url?q=https%3A%2F%2Fdiscord.gg%2FMjSbyKa&sa=D&sntz=1&usg=AFQjCNHLFs5N2ipCqFsV9tnQvcAJ26BSxg) or check out our [Github](https://github.com/EleutherAI)! We're an open community, so you are free to contribute as you wish. We welcome contributors of all backgrounds and experience levels: there's lots to do. Since you're reading this, you've already found our information repo, which should be enough to get you up to speed on our work.

Q: *Are there any other ways to support EleutherAI?*

A: Yes.
* If you're fluent in a language other than English, we would like your input as we build the Pile™ v2. Send a message to `@bmk` on Discord if you would be interested in helping us here.
* Help us on our projects. Our evaluation suite and alignment work are in particular need for more contributors.
* If you or someone you know has access to large quantities of CPU, GPU, or TPU resources, send a message to `@Sid` with more details.

Q: *How does EleutherAI operate?*

A: We currently have a flat structure, with each member responsible for keeping the project control information up to date. Each project has a leader and set of core contributors, so please speak to them first in the Discord if you want to participate. [Connor](https://github.com/ConnorJL) has set the group's alignment vision and general direction. [Stella](https://github.com/StellaAthena) has taken on a leadership and coordination role, and has overall responsibility for the Documentation and Project Control systems. [Leo](https://github.com/leogao2) and [Sid](https://github.com/sdtblck) have led development of the Pile™ and model training, respectively.


Q: *So . . . what's the deal with your logo?*

A: In keeping with the theme, our logo and font are also AI-generated.

Q: *Where can I go if I have more questions?*

A: The Discord is the best place for that. The community has always been helpful to new, curious members. Additionally, some of our core contributors' usernames will appear in purple or green in the chat.

Q: *I'm new to deep learning - what is a transformer?*
   *How do i get into AI? Tell me how everything works!*
   
A: We are a research-focused discord server and not an educational one. We welcome beginners to lurk and talk about topics they’re knowledgeable of, but this is not the place to get intro-level resources or answers to basic questions. We have links to several excellent beginner-friendly servers in our #communities channel.


### GPT-Neo

Q: *What is GPT-Neo?*

A: GPT-Neo is our codebase for training massive language models, which we plan to release as open source. The models themselves are *unnamed*, as of yet.

Q: *How are you going to train such big models?*

A: We have built a [framework](https://github.com/EleutherAI/gpt-neo) using Mesh Tensorflow that lets us train models at super-large scales, including on GPUs and TPUs. At the moment, we have limited access to preemptible TPUs through the [TensorFlow Research Cloud (TFRC) program](https://www.tensorflow.org/tfrc). In the future, our plan is to ask "*please, sir, may I have some more?*". In the event such a plan does not work, we will consider other options.

Q: *What about distributed computing, like [Folding@Home](https://foldingathome.org/) or [hivemind](https://github.com/learning-at-home/hivemind)?*

A: We've considered the possibility of pooling GPUs for training models. The main problems with current approaches are: a) they are unlikely to work given how extremely dense and sensitive backprop is, and MoE-based models significantly underperform regular models, b) even just considering theoretical performance, getting enough contributors to reach more compute power than we currently have is unrealistic, let alone after all the overhead of distributing, and c) current approaches are not attacker-resistant at all, or would cause enormous amounts of overhead. In short, doing it well, and at this scale is an unsolved problem that would take a lot of work to make happen. If you have expertise in this area, though, folks would be happy to hear you out.

Q: *How big is the largest model you've trained?*

A: At the time of writing---October 27th, 2020---we've trained many models under many configurations. The largest we have trained at all clocked in at 100B parameters. For full training, our largest yet is 1.3B parameters, approximately the same size as GPT-2 XL, and used OpenWebText as its corpus. In the near future, we will be training a set of smaller models on The Pile™ and Common Crawl.

Q: *How's the model doing?*

A: Well! (I think) If you're curious about how our models are doing, you can watch them train on the lovely [Foomboard](https://kevinwatkins.github.io/foomboard/).

Q: *Have you considered more efficient architectures?*

A: Yes, we are exploring the full design space, including various linear-scaling attention mechanisms, mixture-of-experts, and other designs. In general, we have found that a mix of global and local attention is important for robust performance.


Q: *Is GPT-Neo free software?*

A: GPT-Neo is MIT-licensed, it is open source.

Q: *Are the models free software?*

A: We have not determined the licensing situation for our models yet.

### The Pile

Q: *What's in the Pile?*

A: The Pile is a 1.25 Terabyte dataset constructed from a curated conglomeration of diverse, high-quality text datasets. It covers a wide gamut, from academic writing, to legal texts, to online literature, video subtitles, and more. This abundance means that saying precisely what is in this meta-dataset is difficult. If you are interested in exploring this, send a message to `#the-pile` on Discord.

Q: *What's the format of the Pile?*

A: We use a simple, compressed JSON format of our own design called [lm_dataformat (LMD)](https://github.com/leogao2/lm_dataformat). It's designed to make writing, storing, and reading text simple and performant. Every logical document maps to a JSON object with `text` and `meta` fields, and batches of these objects are compressed using `zstd` or `gz`. Any kind of corpus that goes into the Pile™---whether HTML, ePUB, PDF extraction, etc.---will be converted into LMD.

Q: *Who can use the Pile?*

A: The Pile was primarily designed for researchers training large-scale langauge models. It also may be of interest to other researchers interested in topics such as bias, online discourse, and text compression.

Q: *Is the Pile released yet?*

A: Yes!

Q: *Where can I get the Pile?*

A: We provide all of the code necessary to replicate the Pile yourself. Additionally, the community of data afficionados at [The-Eye](https://the-eye.eu/) are distributing [pre-built versions](https://the-eye.eu/public/AI/pile/) as well.

Q: *Can I add something to the Pile?*

A: Yes! All contributions should be sent to the [`version2` branch](https://github.com/EleutherAI/the-pile/tree/version2). Pile v1 is finalized an is no longer accepting contributions.

Q: *Have you considered adding Discord logs?*

A: Yes. We decided against it, as there are good privacy reasons Discord users may not expect or want their conversations unwittingly added to a public dataset like this. Collecting such dataset would most likely also violate [Discord's ToS](https://discord.com/terms). In general, more trouble than they're worth.

Q: *Can I make my own version of the Pile?*

A: Of course! For just this reason, all of the components and the Pile creation process are reproducible. Look for a repo labeled as `pile-[COMPONENT]` or `pile_[COMPONENT]` if you want to reproduce a component. [This repo](https://github.com/EleutherAI/the-pile) is where you should go if you want to build your own pile out of the same base datasets. We may also provide links to pre-processed components to allow you to mix, match, and re-sample to derive your own.


### The Radioactive Lab

Q: *How do you plan on making your models "radioactive"?*

A: The Rad Lab is pursuing a number of directions to stamp our models. One of particular interest is through the use of "radioactive data", which was an approach introduced by [this paper](https://arxiv.org/abs/2002.00937).


### Other Activities

#### HUMONGOUS

Status: on hold until we figure out proper multilingual

Q: *Why are you building HUMONGOUS?*

A: Because none of the other large datasets met our needs. For Pile™ v2, we will need tens of terabytes of clean, multilingual text, which is quite hard to come by! For that, we felt that building our own pipeline to extract LM-ready text from raw WARC (Web ARChive) files was the best strategy.


Q: *Doesn't Common Crawl already provide processed plaintext in WET files?*

A: Yes, and they are *horrendous* in quality (no slight intended to the good folks at CC). For LM training, in their raw form, these WET files are basically unusable for LM training. Also, filtering text files is really hard. 

Q: *How is HUMONGOUS different from ([OSCAR](https://oscar-corpus.com/)/[C4](https://www.tensorflow.org/datasets/catalog/c4)/[CC100](http://data.statmt.org/cc-100/))?*

A: From a filtering standpoint, our goal is to eliminate *most* of the cruft so researchers can filter---however they want to---a 35TB dataset (for example), rather than a 3500TB one. Compared to other Common-Crawl derived datasets, HUMONGOUS aims to do a couple of things:
* Start from the raw WARC files, rather than WET files, which allows us to extract better text.
* Be fully multi-lingual, covering as many languages as we can identify.
* Process all Common Crawl dumps, past and future, not just one month's snapshot.


Q: *How big will the resulting text be?*

A: We don’t know for sure yet, but probably a few dozen TB. 

Q: *But won't this take __ages__?*

A: It could, but we don’t need to do it all at once. CC can be split into chunks and processed gradually, only taking as much as is needed for the various stages of the Pile™, and handling the rest over a longer period.
