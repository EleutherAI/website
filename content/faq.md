---
title: "FAQ"
date: 2021-05-23T16:45:00-04:00
layout: page
---

# FAQ  

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

A: Keeping with the theme, our logotype and all images on this website were generated with deep learning techniques.

---

#### Q: *Where can I go if I have more questions?*

A: [Discord](https://discord.gg/avxKQUv2fW) is the best place for that. Our founding members appear in {{<discord/handle drole="O5" name="purple">}} and our core contributors appear in {{<discord/handle drole="level5" name="blue">}}. They will be able to provide helpful guidance or answer questions.

However, we ask that you do not expect us to be your tech support; those who contribute to EleutherAI do so in their free time and tend to prefer contributing to projects rather than debugging your problems. We recommend consulting the corresponding documentation before asking us for help. If you think you have found a bug, please consider opening an issue on [GitHub](https://github.com/EleutherAI).

---

#### Q: *I'm new to deep learning---How do I get into AI? What is a transformer? Tell me how everything works!*

A: We are a research-focused Discord server and not an educational one. We welcome beginners to lurk and talk about topics they are knowledgeable of, but this is not the place to get intro-level resources or answers to basic questions. We have links to several excellent beginner-friendly servers on [Discord](https://discord.gg/avxKQUv2fW) in the {{<discord/channel "#communities">}} channel.

---

### GPT&#8288;-&#8288;Neo and GPT&#8288;-&#8288;NeoX

#### Q: *What are GPT&#8288;-&#8288;Neo and GPT&#8288;-&#8288;NeoX?*

A: [GPT&#8288;-&#8288;Neo](https://github.com/EleutherAI/gpt-neo) and [GPT&#8288;-&#8288;NeoX](https://github.com/EleutherAI/gpt-neox) are our codebases for training massive language models, for which we plan to release under open licenses. The models themselves are referred to by their size (in millions or billions of parameters).

---

#### Q: *How big is the largest publically available model you have trained?*

A: On March 21st, 2021 we released a 2.7 billion parameter model trained upon the Pile.

---

#### Q: *Are you serious when you say you are going to train a model comparable to the biggest GPT&#8288;-&#8288;3 (175 billion parameters)?*

A: Yes, that is the plan. We expect our final model to be somewhere between 150 and 200 billion parameters.

---

#### Q: *Have you considered the possible risks of creating models like these?*

A: Yes, we have considered the risks of creating such models at length. Although EleutherAI contributors have nuanced opinions, there is a consensus with the following arguments:
- Given the continuing advancement of technology, it is impossible to prevent these kinds of models from becoming widespread. We cannot put the genie back in the bottle.
- Any sufficiently funded actor (including but not limited to large corporations and foreign intelligence services) could already have built such models outside of the public eye. There is good reason to believe multiple already have, or are at least in the process of doing so. [*Understanding&nbsp;the&nbsp;Capabilities, Limitations, and&nbsp;Societal&nbsp;Impact of Large Language Models*](https://arxiv.org/abs/2102.02503) estimates that such models could be completed within a year after [*Language&nbsp;Models&nbsp;are&nbsp;Few&#8288;-&#8288;Shot&nbsp;Learners*](https://arxiv.org/abs/2005.14165).
- Without open access to such models to study, [performing critical safety research](https://arxiv.org/abs/2103.14659) is difficult. We intend to make these models accessible to assist academics in such research.
- To entrust the assessments of for-profit corporations on the risks of new technologies is difficult, even if they have the best of intentions. This is especially true when a clear financial incentive to exclusivity exists for those afformentioned new technologies.

---

#### Q: *When do you plan to have more models available?*  

A: As a collective of volunteer researchers and engineers who contribute in our free time, we are unable to commit to either a timeline or a roadmap for future models.

---

#### Q: *How are you training such large models?*

A: For GPT&#8288;-&#8288;Neo, we utilize our limited access to preemptible TPUs through the [TPU Research Cloud (TRC)](https://sites.research.google/trc/). For our future models to be trained with GPT&#8288;-&#8288;NeoX, we have been graciously offered high-performance GPU compute by [CoreWeave](https://www.coreweave.com/). CoreWeave is excited by the open nature of the project and is very keen in helping us to break the OpenAI-Microsoft monopoly on massive autoregressive language models.

---

#### Q: *What differentiates GPT&#8288;-&#8288;NeoX from GPT&#8288;-&#8288;Neo?*

A: GPT&#8288;-&#8288;Neo is a codebase built from the ground up upon [Mesh Tensorflow](https://github.com/tensorflow/mesh), designed for training on TPUs.  
Apart from appending the 24th letter of the ISO basic Latin alphabet, GPT&#8288;-&#8288;NeoX is an entirely separate, in-development codebase based upon Megatron&#8288;-&#8288;LM and [DeepSpeed](https://www.deepspeed.ai/) and is designed for GPUs.

---

#### Q: *Why do you need GPT&#8288;-&#8288;NeoX when you have GPT&#8288;-&#8288;Neo? Why maintain two codebases?*

A: Our motivation for the development of GPT&#8288;-&#8288;NeoX is our access to compute resources: It is not realistic for us to use TRC TPUs to train models larger than around 20 billion parameters. Although TRC can potentially provide enough TPUs to train such large models, the compute is unavailable for the time we would need due to the pre-empting of instances. Even with a v3&#8288;-&#8288;2048, a model between 150 and 175 billion parameters would require months to train. CoreWeave provides us a path to train models at the scales we would like, but we need to utilize their GPUs for training instead of TPUs.

We, therefore, have two reasons to retire the GPT-Neo codebase in favor of developing GPT-NeoX:
- Mesh TensorFlow handles TPUs and GPUs differently, and code designed for use with TPUs has no guarantee to work well on GPUs.
- It makes sense to build a new codebase to take full advantage of GPU hardware---even tiny performance improvements can add up to substantial time and resource savings.

---

#### Q: *What about volunteer-driven distributed computing, like&nbsp;[BOINC](https://boinc.berkeley.edu/), [Folding@Home](https://foldingathome.org/), or&nbsp;[hivemind](https://github.com/learning-at-home/hivemind)?*

A: We have considered the possibility of pooling volunteer resources for training models, but upon thorough review, we have concluded that such approaches are not a viable option today. There are numerous problems with current distributed approaches for us:
- Backpropagation is dense and sensitive to precision, therefore requiring high-bandwidth communication.
- Mixture-of-experts-based models tend to significantly underperform monolithic (regular) models for the same number of parameters.
- Having enough contributors to outweigh the high overhead is infeasible.
- Verifiability and resistance to outside attack are not currently possible without significant additional overhead.

In short, doing volunteer-driven distributed compute well for this use case is an unsolved problem. If you have expertise in this area, drop us a line and we will be happy to hear you out.

---

#### Q: *Have you considered more efficient architectures or methods? Have you considered distillation?*

A: Our intention is not to perfectly replicate the architecture used by GPT&#8288;-&#8288;3 but to instead build models comparable to what OpenAI has built. We are committed to exploring the entire space of architectures and methods, including various linear-scaling attention mechanisms, mixture-of-experts, and other designs. However, in our experience, these designs are not always well suited to language modeling: Attention mechanisms that scale with linear complexity with respect to sequence length are often strictly incompatible with the autoregressive objective used for text generation; the remaining methods have faired poorly in our testing. Engineering is full of trade-offs, and silver-bullet research breakthroughs are uncommon occurences. [If and when new methodologies surpass what we have already, we will integrate and use them.](https://blog.eleuther.ai/rotary-embeddings)

Our agreement with CoreWeave includes a stipulation that we attempt distillation on the final model to make it easier to deploy. It is unknown if distillation is advantageous at these scales, but we intend to find out.

---

#### Q: *Will I be able to run models on my computer locally, offline?*

A: The answer is highly dependent on hardware and configuration.

No, you will not be able to run a model the size of full-scale GPT&#8288;-&#8288;3 on your first-generation Macbook Air. 175 billion parameters at single-precision (binary32) take up 700 Gigabytes, and realistically the entire model needs to be loaded into memory for inference. It is unlikely that consumer hardware will be able to run anything of that scale for years to come, even on CPU. To run large models beyond a few billion parameters there is an expectation to utilize systems with large amounts of compute and memory.

Smaller models can be run on more pedestrian hardware: 125 million parameters take up only 500 Megabytes and should run on a basic laptop without a hitch, while 1.3 billion parameters take up 5 Gigabytes and should run on capable personal computers without issue.

If you are interested in inferencing and fine-tuning models, we highly recommend using [the implementation in Hugging Face Transformers](https://huggingface.co/transformers/model_doc/gpt_neo.html), which is often far easier to both install and use than our research code. We do not support or maintain the Hugging Face implementation beyond [our organization in Model Hub](https://huggingface.co/eleutherai), and issues with Transformers or its usage should be directed elsewhere such as the [Hugging Face community forums](https://discuss.huggingface.co/).

---

#### Q: *Are the codebases free software?*

A: GPT&#8288;-&#8288;Neo is MIT-licensed, while GPT&#8288;-&#8288;NeoX is licensed under Apache 2.0. These are the most freely-termed licenses that we can provide for each codebase respectively.

---

#### Q: *Are the models free software?*

A: EleutherAI is licensing models under Apache 2.0. If you use our models, we would highly appreciate you citing or displaying your usage of them. 

---

#### Q: *How should I cite your models?*

A: We ask that you cite both the codebase and the dataset together when citing models. Our recommended citation method is as follows.

*In the document body:*
```latex
X.XB GPT-Neo \citep{gpt-neo} model trained on the Pile \citep{pile}
```

*BibTeX entries:*
```bibtex
@article{pile,
  title={The {P}ile: An 800GB Dataset of Diverse Text for Language Modeling},
  author={Gao, Leo and Biderman, Stella and Black, Sid and Golding, Laurence and Hoppe, Travis and Foster, Charles and Phang, Jason and He, Horace and Thite, Anish and Nabeshima, Noa and Presser, Shawn and Leahy, Connor},
  journal={arXiv preprint arXiv:2101.00027},
  year={2020}
}
@software{gpt-neo,
  author = {Black, Sid and Gao, Leo and Wang, Phil and Leahy, Connor and Biderman, Stella},
  title = {{GPT-Neo}: Large Scale Autoregressive Language Modeling with Mesh-Tensorflow},
  url = {http://github.com/eleutherai/gpt-neo},
  version = {1.0},
  year = {2021}
}
```

---

### The Pile

#### Q: *What is the Pile? What is in it?*

A: The Pile is a 825 GiB diverse, open source language modeling data set that consists of 22 smaller, high-quality datasets combined together. For more information, please read the [paper](https://arxiv.org/abs/2101.00027).

---

#### Q: *Who can use the Pile?*

A: The Pile was primarily designed for researchers training large-scale language models. It also may be of interest to other researchers interested in topics such as bias, online discourse, and text compression.

---

#### Q: *Where can I get the Pile?*

A: The data can be downloaded [here](https://the-eye.eu/public/AI/pile/).

---

#### Q: *Can I add something to the Pile?*

A: Pile v1 is finalized and is no longer accepting contributions. All contributions for a Pile v2 should be sent to the [`version2` branch](https://github.com/EleutherAI/the-pile/tree/version2). 

---

#### Q: *Have you considered adding Discord logs?*

A: Yes. We decided against it, as there are good privacy reasons Discord users may not expect or want their conversations unwittingly added to a public dataset like this. Collecting such a dataset would most likely also violate [Discord's ToS](https://discord.com/terms). In general, more trouble than they're worth.

---

#### Q: *Can I make my own version of the Pile?*

A: Of course! For just this reason, all of the components and the Pile creation process are reproducible. The code used to create the PIle can be found [here](https://github.com/EleutherAI/the-pile). Links to the code for reproducing each component are also available at that repo.