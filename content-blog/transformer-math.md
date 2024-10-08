---
title: "Transformer Math 101"
date: 2023-04-18T00:00:00+01:00
draft: False
author:
  ["Quentin Anthony", "Stella Biderman", "Hailey Schoelkopf"]
description: "We present basic math related to computation and memory usage for transformers"
categories: ["Investigations"]
mathjax: True
---

# Introduction

A lot of basic, important information about transformer language models can be computed quite simply. Unfortunately, the equations for this are not widely known in the NLP community. The purpose of this document is to collect these equations along with related knowledge about where they come from and why they matter.

**Note:** This post is primarily concerned with training costs, which are dominated by VRAM considerations. For an analogous discussion of inference costs with a focus on latency, check out [this excellent blog post](https://kipp.ly/blog/transformer-inference-arithmetic/) by Kipply.

# Compute Requirements

The basic equation giving the cost to train a transformer model is given by: 

$$
C\approx\tau T = 6PD
$$

where:

 - $C$ is the compute required to train the transformer model, in total floating point operations
 - $C=C_{\text{forward}}+C_{\text{backward}}$
 - $C_{\text{forward}}\approx2PD$
 - $C_{\text{backward}}\approx4PD$
 - $\tau$ is the aggregate throughput of your hardware setup ($\tau=(\text{No. GPUs}) \times (\text{Actual FLOPs}/\text{GPU})$), in FLOPs
 - $T$ is the time spent training the model, in seconds
 - $P$ is the number of parameters in the transformer model
 - $D$ is the dataset size, in tokens

These equations are proposed and experimentally validated in [OpenAI’s scaling laws paper](https://arxiv.org/abs/2001.08361) and [DeepMind’s scaling laws paper](https://arxiv.org/abs/2203.15556). Please see each paper for more information.

It’s worth taking an aside and discussing the units of $C$. $C$ is a measure of total compute, but can be measured by many units such as:

- FLOP-seconds, which is in units of $[\frac{\text{Floating Point Operations}}{\text{Second}}] \times [\text{Seconds}]$
- GPU-hours, which is in units of $[\text{No. GPUs}]\times[\text{Hours}]$
- Scaling laws papers tend to report values in PetaFLOP-days, or $10^{15}\times24\times3600$ total floating point operations

One useful distinction to keep in mind is the concept of $\text{Actual FLOPs}$. While GPU accelerator whitepapers usually advertise their theoretical FLOPs, these are never met in practice (especially in a distributed setting!). Some common reported values of $\text{Actual FLOPs}$ in a distributed training setting are reported below in the Computing Costs section.

Note that we use the throughput-time version of the cost equation as used in [this wonderful blog post on LLM training costs](https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4).

## Parameter vs Dataset Tradeoffs

Although strictly speaking you can train a transformer for as many tokens as you like, the number of tokens trained can highly impact both the computing costs and the final model performance making striking the right balance important.

**Let’s start with the elephant in the room: “compute optimal” language models.** Often  referred to as “Chinchilla scaling laws” after the model series in the paper that gave rise to current beliefs about the number of parameters, a compute optimal language model has a **number of parameters** and a **dataset size** that satisfies the approximation $D=20P$. This is optimal in one very specific sense: in a resource regime where using 1,000 GPUs for 1 hour and 1 GPU for 1,000 hours cost you the same amount, if your goal is to maximize performance while minimizing the cost in GPU-hours to train a model you should use the above equation.  

**We do not recommend training a LLM for less than 200B tokens.** Although this is “chinchilla optimal” for many models, the resulting models are typically quite poor. For almost all applications, we recommend determining what inference cost is acceptable for your usecase and training the largest model you can to stay under that inference cost for as many tokens as you can. 

## Engineering Takeaways for Compute Costs

Computing costs for transformers are typically listed in GPU-hours or FLOP-seconds.

- GPT-NeoX achieves 150 TFLOP/s/A100 with normal attention and 180 TFLOP/s/A100 with Flash Attention. This is in line with other highly optimized libraries at scale, for example Megatron-DS reports between 137 and 163 TFLOP/s/A100.
- As a general rule of thumb, you should always be able to achieve approximately 120 TFLOP/s/A100. If you are seeing below 115 TFLOP/s/A100 there is probably something wrong with your model or hardware configuration.
- With high-quality interconnect such as InfiniBand, you can achieve linear or sublinear scaling across the data parallel dimension (i.e. increasing the data parallel degree should increase the overall throughput nearly linearly). Shown below is a plot from testing the GPT-NeoX library on Oak Ridge National Lab’s Summit supercomputer. Note that V100s are on the x-axis, while most of the numerical examples in the post are for A100s.


{{<figure src="/images/blog/transformer-math/neox-scaling.png" alt="GPT-NeoX scaling" align="center"/>}}

# Memory Requirements

Transformers are typically described in terms of their *size in parameters*. However, when determining what models can fit on a given set of computing resources you need to know **how much space in bytes** the model will take up. This can tell you how large a model will fit on your local GPU for inference, or how large a model you can train across your cluster with a certain amount of total accelerator memory.

## Inference

### Model Weights

{{<figure src="/images/blog/transformer-math/dl-precisions.png" alt="Different floating point formats and their relative bits for precision and range" align="center"/>}}

Most transformers are trained in **mixed precision**, either fp16 + fp32 or bf16 + fp32. This cuts down on the amount of memory required to train the models, and also the amount of memory required to run inference. We can cast language models from fp32 to fp16 or even int8 without suffering a substantial performance hit. These numbers refer to the size *in bits* a single parameter requires. Since there are 8 bits in a Byte, we divide this number by 8 to see how many Bytes each parameter requires  

- In int8, $\text{memory}_{\text{model}}=(1 \text{ byte} /\text{param})\cdot ( \text{No. params})$
- In fp16 and bf16, $\text{memory}_{\text{model}}=(2 \text{ bytes} /\text{param})\cdot ( \text{No. params})$
- In fp32, $\text{memory}_{\text{model}}=(4 \text{ bytes} /\text{param})\cdot (\text{No. params})$

There is also a small amount of additional overhead, which is typically irrelevant to determining the largest model that will fit on your GPU. In our experience this overhead is ≤ 20%.

### Total Inference Memory

In addition to the memory needed to store the model weights, there is also a small amount of additional overhead during the actual forward pass. In our experience this overhead is ≤ 20% and is typically irrelevant to determining the largest model that will fit on your GPU. 

In total, a good heuristic answer for “will this model fit for inference” is:

$\text{Total Memory}_{\text{Inference}}\approx(1.2) \times \text{Model Memory}$

We will not investigate the sources of this overhead in this blog post and leave it to other posts or locations for now, instead focusing on memory for model training in the rest of this post. If you’re interested in learning more about the calculations required for inference, check out [this fantastic blog post covering inference in depth](https://kipp.ly/blog/transformer-inference-arithmetic/). Now, on to training!

## Training

In addition to the model parameters, training requires the storage of optimizer states and gradients in device memory. This is why asking “how much memory do I need to fit model X” immediately leads to the answer “this depends on training or inference.” Training always requires more memory than inference, often very much more!

### Model Parameters

First off, models can be trained in pure fp32 or fp16:

- Pure fp32, $\text{memory}_{\text{model}}=(4 \text{ bytes} /\text{param})\cdot (\text{No. params})$
- Pure fp16, $\text{memory}_{\text{model}}=(2 \text{ bytes} /\text{param})\cdot (\text{No. params})$

In addition to the common model weight datatypes discussed in Inference, training introduces **mixed-precision** training such as [AMP](https://developer.nvidia.com/automatic-mixed-precision). This technique seeks to maximize the throughput of GPU tensor cores while maintaining convergence. The modern DL training landscape frequently uses mixed-precision training because: 1) fp32 training is stable, but has a high memory overhead and doesn’t exploit NVIDIA GPU tensor cores, and 2) fp16 training is stable but difficult to converge. For more information on mixed-precision training, we recommend reading [this notebook by tunib-ai](https://nbviewer.org/github/tunib-ai/large-scale-lm-tutorials/blob/main/notebooks/08_zero_redundancy_optimization.ipynb). Note that mixed-precision requires an fp16/bf16 and fp32 version of the model to be stored in memory, requiring:

- Mixed-precision (fp16/bf16 and fp32), $\text{memory}_{\text{model}}=(2 \text{ bytes} /\text{param})\cdot (\text{No. params})$

plus an additional size $(4\text{ bytes/param}) \cdot (\text{\#params})$ copy of the model **that we count within our optimizer states**.

### Optimizer States

Adam is magic, but it’s highly memory inefficient. In addition to requiring you to have a copy of the model parameters and the gradient parameters, you also need to keep an additional three copies of the gradient parameters. Therefore, 

- For vanilla AdamW, $\text{memory}_{\text{optimizer}}=(12 \text{ bytes}/\text{param})\cdot (\text{No. params})$
    - fp32 copy of parameters: 4 bytes/param
    - Momentum: 4 bytes/param
    - Variance: 4 bytes/param
- For 8-bit optimizers like [bitsandbytes](https://github.com/TimDettmers/bitsandbytes), $\text{memory}_{\text{optimizer}}=(6 \text{ bytes} /\text{param})\cdot (\text{No. params})$
    - fp32 copy of parameters: 4 bytes/param
    - Momentum: 1 byte/param
    - Variance: 1 byte/param
- For SGD-like optimizers with momentum, $\text{memory}_{\text{optimizer}}=(8 \text{ bytes} /\text{param})\cdot (\text{No. params})$
    - fp32 copy of parameters: 4 bytes/param
    - Momentum: 4 bytes/param

### Gradients

Gradients can be stored in fp32 or fp16 (Note that the gradient datatype often matches the model datatype. We see that it therefore is stored in fp16 for fp16 mixed-precision training), so their contribution to memory overhead is given by:

- In fp32, $\text{memory}_{\text{gradients}}=(4 \text{ bytes} /\text{param})\cdot (\text{No. params})$
- In fp16, $\text{memory}_{\text{gradients}}=(2 \text{ bytes} /\text{param})\cdot (\text{No. params})$

### Activations and Batch Size

Modern GPUs are typically bottlenecked by memory, not FLOPs, for LLM training. Therefore activation recomputation/checkpointing is an extremely popular method of trading reduced memory costs for extra compute costs. Activation recomputation/checkpointing works by recomputing activations of certain layers instead of storing them in GPU memory. The reduction in memory depends on how selective we are when deciding which layers to clear, but Megatron’s selective recomputation scheme is depicted in the figure below:

{{<figure src="/images/blog/transformer-math/activations.png" alt="activation memory" align="center"/>}}

Where the dashed red line indicates the memory capacity of an A100-80GB GPU, and “present work” indicates the memory requirements after applying selective activation recomputation. See  [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198) for further details and the derivation of the equations below

The basic equation giving the memory required to store activations for a transformer model is given by: 

$$
\begin{align*}\text{memory}^{\text{No Recomputation}}_{\text{activations}}=sbhL(10+\frac{24}{t}+5\frac{a \cdot s}{h\cdot t}) \text{ bytes}\end{align*}
$$

$$
\begin{align*}\text{memory}^{\text{Selective Recomputation}}_{\text{activations}}=sbhL(10+\frac{24}{t}) \text{ bytes}\end{align*}
$$

$$
\begin{align*}\text{memory}^{\text{Full Recomputation}}_{\text{activations}}=2 \cdot sbhL \text{ bytes}\end{align*}
$$

where:

- $s$ is the sequence length, in tokens
- $b$ is the batch size per GPU
- $h$ is the dimension of the hidden size within each transformer layer
- $L$ is the number of layers in the transformer model
- $a$ is the number of attention heads in the transformer model
- $t$ is the degree of tensor parallelism being used (1 if not)
- We assume no sequence parallelism is being used
- We assume that activations are stored in fp16

The additional recomputation necessary also depends on the selectivity of the method, but it’s bounded above by a full additional forward pass. Hence the updated cost of the forward pass is given by: 

$$
2PD\leq C_{\text{forward}}\leq4PD
$$

### Total Training Memory

Therefore, a good heuristic answer for “will this model fit for training” is:

$$
\begin{align*}\text{Total Memory}_{\text{Training}} = \text{Model Memory}+\text{Optimiser Memory}+\text{Activation Memory}+\text{Gradient Memory}\end{align*}
$$

## Distributed Training

### Sharded Optimizers

The massive memory overheads for optimizers is the primary motivation for sharded optimizers such as [ZeRO](https://arxiv.org/abs/1910.02054) and [FSDP](https://engineering.fb.com/2021/07/15/open-source/fsdp/). Such sharding strategies reduce the optimizer overhead by a factor of $\text{No. GPUs}$, which is why a given model configuration may fit at large scale but OOM at small scales. If you’re looking to calculate the memory overhead required by training using a sharded optimizer, you will need to include the equations from the figure below. For some sample calculations of sharded optimization, see the following figure from the [ZeRO](https://arxiv.org/abs/1910.02054) paper (Note that $P_{os}$ $P_{os+g}$ and $P_{os+g+p}$ are commonly denoted as ZeRO-1, ZeRO-2, ZeRO-3, respectively. ZeRO-0 commonly means “ZeRO disabled”):


{{<figure src="/images/blog/transformer-math/zero_fig.png" alt="ZeRO illustration" align="center"/>}}
{{<figure src="/images/blog/transformer-math/zero_legend.png" alt="ZeRO legend" align="center"/>}}

In the language of this blog post (assuming mixed-precision and the Adam optimizer):

- For ZeRO-1,

$$
\begin{align*}\text{Total Memory}_{\text{Training}}\approx\text{Model Memory}+\frac{\text{Optimizer memory}}{(\text{No. GPUs})}+\text{Activation Memory}+\text{Gradient Memory}\end{align*}
$$

- For ZeRO-2,

$$
\begin{align*}\text{Total Memory}_{\text{Training}}\approx\text{Model Memory}+\text{Activation Memory}+\frac{\text{Optimizer Memory}+\text{Gradient Memory}}{(\text{No. GPUs})}\end{align*}
$$

- For ZeRO-3,

$$
\begin{align*}\text{Total Memory}_{\text{Training}}\approx \text{Activation Memory}+\frac{\text{Model Memory}+\text{Optimizer Memory}+\text{Gradient Memory}}{(\text{No. GPUs})} + \text{(ZeRO-3 Live Params)}\end{align*}
$$

Where $(\text{DP Degree})$ is just $(\text{No. GPUs})$ unless pipeline and/or tensor parallelism are applied. See [Sharded Optimizers + 3D Parallelism](https://www.notion.so/Sharded-Optimizers-3D-Parallelism-9c476d020d7641a299fb6be6ae82e9f8) for details.

Note that ZeRO-3 introduces a set of live parameters. This is because ZeRO-3 introduces a set of config options (***stage3_max_live_parameters, stage3_max_reuse_distance, stage3_prefetch_bucket_size, stage3_param_persistence_threshold***) that control how many parameters are within GPU memory at a time (larger values take more memory but require less communication). Such parameters can have a significant effect on total GPU memory.

Note that ZeRO can also partition activations over data parallel ranks via **ZeRO-R**. This would also bring the $\text{memory}_\text{activations}$ above the tensor parallelism degree $t$. For more details, read the associated [ZeRO paper](https://arxiv.org/abs/1910.02054) and [config options](https://www.deepspeed.ai/docs/config-json/#activation-checkpointing) (note in GPT-NeoX, this is the `partition_activations` flag). If you are training a huge model, you would like to trade some memory overhead for additional communication cost, and activations become a bottleneck. As an example of using ZeRO-R along with ZeRO-1:

$$
\begin{align*}\text{Total Memory}_{\text{Training}}\approx\text{Model Memory}+\frac{\text{Optimizer Memory}}{(\text{No. GPUs})}+\text{Activation Memory}+\text{Gradient Memory}\end{align*}
$$

### 3D Parallelism

Parallelism for LLMs comes in 3 primary forms:

**Data parallelism:** Split the data among (possibly model-parallel) replicas of the model

**Pipeline or Tensor/Model parallelism:** These parallelism schemes split the parameters of the model across GPUs. Such schemes require significant communication overhead, but their memory reduction is approximately: 

$$
\begin{align*}\text{memory}^{\text{w/ parallelism}}_{\text{model}}\approx\frac{\text{Model Memory}}{\text{(Pipe-Parallel-Size})\times\text{(Tensor-Parallel-Size)}}\end{align*}
$$

$$
\begin{align*}\text{memory}^{\text{w/ parallelism}}_{\text{gradients}}\approx\frac{\text{Gradient Memory}}{\text{(Pipe-Parallel-Size})}\end{align*}
$$

Note that this equation is approximate due to the facts that (1) pipeline parallelism doesn’t reduce the memory footprint of activations, (2) pipeline parallelism requires that all GPUs store the activations for all micro-batches in-flight, which becomes significant for large models, and (3) GPUs need to temporarily store the additional communication buffers required by parallelism schemes. 

### Sharded Optimizers + 3D Parallelism

When ZeRO is combined with tensor and/or pipeline parallelism, the resulting parallelism strategy forms a mesh like the following:

{{<figure src="https://i.imgur.com/xMgptTN.png" alt="3D parallelism" align="center"/>}}

As an important aside, the DP degree is vital for use in calculating the global batch size of training. The data-parallel degree depends on the number of complete model replicas:

$$
\begin{align*}\text{DP Degree = }\frac{\text{No. GPUs}}{\text{(Pipe-Parallel-Size})\times\text{(Tensor-Parallel-Size)}}\end{align*}
$$

Pipeline parallelism and tensor parallelism are compatible with all stages of ZeRO. However, it's difficult to maintain efficiency when combining pipeline parallelism with ZeRO-2/3's gradient sharding (Because ZeRO-2 shards the gradients, but pipeline parallelism accumulates them. It's possible to carefully define a pipeline schedule and overlap communication to maintain efficiency, but it's difficult to the point that DeepSpeed currently forbids it: https://github.com/microsoft/DeepSpeed/blob/v0.10.1/deepspeed/runtime/pipe/engine.py#L71). Tensor parallelism, however, is complementary to all stages of ZeRO because on each rank:

- ZeRO-3 gathers the full layer **parameters** from other ranks, processes a **full** input on the now-local full layer, then frees the memory that was allocated to hold the remote ranks' parameters.
- Tensor Parallelism gathers the remote **activations** for the local input from other ranks, processes a **partition** of the input using the local layer partition, then sends the next layer's activations to remote ranks

For the majority of Eleuther's work, we train with pipeline and tensor parallelism along with ZeRO-1. This is because we find ZeRO-3 to be too communication-heavy for our hardware at large scales, and instead use pipeline parallelism across nodes along with tensor parallelism within nodes.

Putting everything together for a typical 3D-parallel ZeRO-1 run with activation partitioning:

$$
\begin{align*}\text{Total Memory}_{\text{Training}}\approx\frac{\text{Model Memory}}{\text{(Pipe-Parallel-Size})\times\text{(Tensor-Parallel-Size)}}+\frac{\text{Optimizer Memory}}{(\text{No. GPUs})}+\frac{\text{Activation Memory}}{\text{(Tensor-Parallel-Size)}}+\frac{\text{Gradient Memory}}{\text{(Pipe-Parallel-Size})}\end{align*}
$$

# Conclusion

EleutherAI engineers frequently use heuristics like the above to plan efficient model training and to debug distributed runs. We hope to provide some clarity on these often-overlooked implementation details, and would love to hear your feedback at contact@eleuther.ai if you would like to discuss or think we’ve missed anything!


To cite this blog post, please use:

```bibtex
@misc{transformer-math-eleutherai,
  title = {Transformer Math 101},
  author = {Anthony, Quentin and Biderman, Stella and Schoelkopf, Hailey},
  howpublished = \url{blog.eleuther.ai/},
  year = {2023}
}
```

