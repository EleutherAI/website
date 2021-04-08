---
title: "FAQ"
date: 2021-02-26T20:18:54+03:00
layout: page
---

## FAQ  

### General

#### Q: *How did this all start?*

A: On July 3rd, 2020, [Connor Leahy](https://github.com/ConnorJL) ({{<discord/handle drole="O5" name="@Daj">}}) posted in the TPU Podcast Discord:
> https://arxiv.org/abs/2006.16668  
> Hey guys lets give OpenAI a run for their money like the good ol' days

To which [Leo Gao](https://github.com/leogao2) ({{<discord/handle drole="O5" name="@bmk">}}) replied:
> this but unironically

And so it began.

---

#### Q: *Where did the name come from?*

A: In Ancient Greek, [*eleutheria*](https://en.wikipedia.org/wiki/Eleutheria) is a word for "liberty", and was used as a proper noun as a personification of the concept. This same personage became [*Libertas*](https://en.wikipedia.org/wiki/Libertas) to the Romans and [*Lady Liberty*](https://en.wikipedia.org/wiki/Statue_of_Liberty) to Americans.

---

#### Q: *How can I get involved?*

A: Join our [Discord](https://discord.gg/avxKQUv2fW) or check us out on [GitHub](https://github.com/EleutherAI)! We're an open community, so you are free to contribute as you wish. However, we expect newcomers either to be fairly knowledgeable or to sit on the sidelines until they understand the internal structure and culture of our operations.  
If you are interested, check out our page on [getting involved](/get-involved).

---

#### Q: *Are there any other ways to support EleutherAI?*

A: Yes. If you or someone you know has access to large quantities of CPU, GPU, or TPU resources, send a message to Sid Black ({{<discord/handle drole="O5" name="@Sid">}}) on [Discord](https://discord.gg/avxKQUv2fW) with more details.

---

#### Q: *So . . . what's the deal with your logo?*

A: Keeping with the theme, our logotype and all images on this website are all generated with machine learning techniques.

---

#### Q: *Where can I go if I have more questions?*

A: [Discord](https://discord.gg/avxKQUv2fW) is the best place for that. Some of our core contributors' usernames will appear in {{<discord/handle drole="O5" name="purple">}} or {{<discord/handle drole="level5" name="blue">}} in the chat, and our regulars familiar with our operations will appear in {{<discord/handle drole="regular" name="green">}}. They should be able to provide helpful guidance or answer questions.  
However, we ask that you do not expect us to be your tech support; those who contribute to EleutherAI do so in their free time and tend to prefer contributing to projects rather than debugging your problems. We recommend consulting the corresponding documentation before asking us for help. If you think you have found a bug, please consider opening an issue on [GitHub](https://github.com/EleutherAI).

---

#### Q: *I'm new to deep learning---How do I get into AI? What is a transformer? Tell me how everything works!*

A: We are a research-focused Discord server and not an educational one. We welcome beginners to lurk and talk about topics they are knowledgeable of, but this is not the place to get intro-level resources or answers to basic questions. We have links to several excellent beginner-friendly servers on [Discord](https://discord.gg/avxKQUv2fW) in the {{<discord/channel "#communities">}} channel.

---

### GPT&#8288;-&#8288;Neo and GPT&#8288;-&#8288;NeoX

#### Q: *What are GPT&#8288;-&#8288;Neo and GPT&#8288;-&#8288;NeoX?*

A: [GPT&#8288;-&#8288;Neo](https://github.com/EleutherAI/gpt-neo) and [GPT&#8288;-&#8288;NeoX](https://github.com/EleutherAI/gpt-neox) are our codebases for training massive language models, which we plan to release as open-source. The models themselves are referred to by their size in billions of parameters.

---

#### Q: *What differentiates GPT&#8288;-&#8288;NeoX from GPT&#8288;-&#8288;Neo?*

A: GPT&#8288;-&#8288;Neo is a codebase built from the ground up upon [Mesh Tensorflow](https://github.com/tensorflow/mesh), designed to lets us train models at super-large scales on GPUs or TPUs.
Apart from appending the 24th letter of the ISO basic Latin alphabet, GPT&#8288;-&#8288;NeoX is an entirely separate, in-development codebase based upon Megatron&#8288;-&#8288;LM and DeepSpeed.

---

#### Q: *Why do you need GPT&#8288;-&#8288;NeoX when you have GPT&#8288;-&#8288;Neo? Why maintain two codebases?*

A: GPT&#8288;-&#8288;NeoX is designed to succeed GPT&#8288;-&#8288;Neo as our primary codebase, which provides several benefits over the older code:
- We can utilize PyTorch, which we find far more flexible and maintainable than TensorFlow. 
- It is not feasible to use the TPUs we have access to through TFRC to train larger models.
- [We have access to sufficient GPU resources to train larger models.](#compute)

---

#### Q: *How big is the largest publically available model you have trained?*

A: On March 21th 2021 we released a 2.7 billion (2.7B) parameter model trained upon the Pile, comparable in size and performance to GPT&#8288;-&#8288;3 Ada.

---

#### Q: *Are you serious when you say you are going to train a model comparable to GPT&#8288;-&#8288;3 DaVinci (175B parameters)?*

A: Yes, that is the plan. We expect our final model to be somewhere between 150B and 200B parameters.

---

#### Q: *When do you plan to have a model of that scale trained? Wouldn't that take a long time?*

We asked some of our GPT&#8288;-&#8288;Neo and GPT&#8288;-&#8288;NeoX contributors about their predictions, and we got the following responses:

> Soon™  
**Leo Gao** ({{<discord/handle drole="O5" name="@bmk">}})  

> Before we all become paperclips—if we're lucky.  
**Connor Leahy** ({{<discord/handle drole="O5" name="@Daj">}})  

> Before the next Millennium Prize Problem is solved.  
**Stella Biderman** ({{<discord/handle drole="mathemagician" name="@StellaAthena">}})  

> Exactly 1.21 gigaseconds after you read this.  
**Shivanshu Purohit** ({{<discord/handle drole="level5" name="@triggerhappygandi">}})  

> In less time than it took *Voyager I* to reach interstellar space.  
**Eric Hallahan** ({{<discord/handle drole="level5" name="@EricHallahan">}})  

A: As a collective of volunteer developers, engineers, and researchers who contribute in our free time, we are unable to commit to a timeline as to when larger models will become available in the future. Our original estimation for a model somewhere between 150B and 200B parameters was sometime in Q3 2021, most likely the August-September timeframe.

To be more specific in our confidence,
- No earlier than August 2021.
- Ideally by the end of 2021.
- "Before the heat death of the universe." ---&nbsp;**Sid&nbsp;Black**&nbsp;({{<discord/handle drole="O5" name="@Sid">}}) 

Our estimates for how long a model similar to GPT&#8288;-&#8288;3 DaVinci will take to train lies somewhere around the four-to-five-month range with optimization and the right hardware.

---

#### Q: *How are you training such large models?* {#compute}

A: For our "small" 1.3B and 2.7B parameter models trained with GPT&#8288;-&#8288;Neo, we utilized our limited access to preemptible TPUs through the [TensorFlow Research Cloud (TFRC) program](https://www.tensorflow.org/tfrc).
For our larger future models to be trained with GPT&#8288;-&#8288;NeoX, we have been graciously been offered high-performance GPU compute by [CoreWeave](https://www.coreweave.com/), an NVIDIA Preferred Cloud Services Provider. CoreWeave is excited by the open nature of the project and is very keen in helping us to break the OpenAI-Microsoft monopoly on massive autoregressive language models.

---

#### Q: *What about distributed computing, like [Folding@Home](https://foldingathome.org/) or [hivemind](https://github.com/learning-at-home/hivemind)?*

A: We have considered the possibility of pooling GPUs for training models, but upon thorough review, we have concluded that such approaches are not a viable option today. There are numerous problems with current distributed approaches for us:
- Backpropagation is extremely dense and sensitive to precision, therefore requiring high-bandwidth communication.
- MoE-based models tend to significantly underperform regular models for the same number of parameters.
- Having enough contributors to outweigh the high overhead is infeasible.
- Resistance to outside attack is not currently possible without significant additional overhead.

In short, doing distributed compute well at this scale is an unsolved problem. If you have expertise in this area, drop us a line and we will be happy to hear you out.

---

#### Q: *Have you considered more efficient architectures or methods?*

A: Yes, we are exploring the full design space, including various linear-scaling attention mechanisms, mixture-of-experts, and other designs. In general, we have found that a mix of global and local attention is important for robust performance.

---

#### Q: *Are the codebases free software?*

A: GPT&#8288;-&#8288;Neo is MIT-licensed, while GPT&#8288;-&#8288;NeoX is licensed under Apache 2.0. These are the most freely-termed licenses that we can provide for each codebase respectively.

---

#### Q: *Are the models free software?*

A: Models are licensed as under Apache 2.0.

---

### The Pile

#### Q: *What's in the Pile?*

A: The Pile is a 1.25 Terabyte dataset constructed from a curated conglomeration of diverse, high-quality text datasets. It covers a wide gamut, from academic writing to legal texts, to online literature, video subtitles, and more. This abundance means that saying precisely what is in this meta-dataset is difficult. If you are interested in exploring this, send a message to {{<discord/channel "#the-pile">}} on Discord.

---

#### Q: *What's the format of the Pile?*

A: We use a simple, compressed JSON format of our design called [`lm_dataformat` (LMD)](https://github.com/leogao2/lm_dataformat). It's designed to make writing, storing, and reading text simple and performant. Every logical document maps to a JSON object with `text` and `meta` fields, and batches of these objects are compressed using `zstd` or `gz`. Any kind of corpus that goes into the Pile---whether HTML, ePUB, PDF extraction, etc.---will be converted into LMD.

---

#### Q: *Who can use the Pile?*

A: The Pile was primarily designed for researchers training large-scale language models. It also may be of interest to other researchers interested in topics such as bias, online discourse, and text compression.

---

#### Q: *Is the Pile released yet?*

A: Yes! [Read the preprint on arXiv here.](https://arxiv.org/abs/2101.00027)

---

#### Q: *Where can I get the Pile?*

A: We provide all of the code necessary to replicate the Pile yourself. Additionally, the community of data aficionados at [The-Eye](https://the-eye.eu/) are distributing [pre-built versions](https://the-eye.eu/public/AI/pile/) as well.

---

#### Q: *Can I add something to the Pile?*

A: Yes! All contributions should be sent to the [`version2` branch](https://github.com/EleutherAI/the-pile/tree/version2). Pile v1 is finalized and is no longer accepting contributions.

---

#### Q: *Have you considered adding Discord logs?*

A: Yes. We decided against it, as there are good privacy reasons Discord users may not expect or want their conversations unwittingly added to a public dataset like this. Collecting such a dataset would most likely also violate [Discord's ToS](https://discord.com/terms). In general, more trouble than they're worth.

---

#### Q: *Can I make my own version of the Pile?*

A: Of course! For just this reason, all of the components and the Pile creation process are reproducible. Look for a repo labeled as `pile-[COMPONENT]` or `pile_[COMPONENT]` if you want to reproduce a component. [This repo](https://github.com/EleutherAI/the-pile) is where you should go if you want to build your own Pile out of the same base datasets. We may also provide links to pre-processed components to allow you to mix, match, and re-sample to derive your own.
