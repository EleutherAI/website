---
title: "Attention Probes"
date: 2025-06-28T22:00:00-00:00
description: "Adding attention to linear probes"
author: ["Stepan Shabalin", "Nora Belrose"]
ShowToc: true
mathjax: true
draft: false
---

Linear probes are a simple way to classify internal states of langauge models. They are trained either on a per-token basis or on a compressed representation of latent vectors from multiple tokens. This reprsentation can be gathered with mean pooling, or the last token could be used.

We propose *attention probes*, a way to avoid pooling by collecting hidden states with an attention layer. The pseudocode is as follows:

```python
def attention_probe(
    hidden_states: Float[Tensor, "seq_len d_model"],
    query_proj: Float[Tensor, "d_model n_heads"],
    value_proj: Float[Tensor, "d_model n_heads n_outputs"],
    position_weights: Float[Tensor, "n_heads"],
) -> Float[Tensor, "n_outputs"]:
    # seq_len, n_heads
    attention_logits = hidden_states @ query_proj
    # position embedding. this is a version of ALiBi with a learned position bias.
    # it is relative to the last or first token.
    position_bias = position_weights[None, :] * torch.arange(seq_len)[:, None]
    attention_logits += position_bias
    # seq_len, n_heads
    attention_weights = softmax(attention_logits, dim=-2)
    # seq_len, n_heads, n_outputs
    values = hidden_states @ value_proj  # won't actually work in torch
    # n_outputs
    output = (attention_weights[..., None] * values).sum(dim=-(-2, -3))

    return output
```

As you can see, the attention probe has multiple heads. Each head finds a single attention logit for a token instead of a logit for each pair of tokens. We add a learnable position bias and take softmax to find attention probabilities. Again, there is only one probability per token and head. This can be thought of as cross-attention with one learned query token.

We then perform the value projection. Because the output dimension of a probe is often very small, we do not need to factorize the projection into value and output as MHA does. There is a version of the output for each token and head, and we sum them up after weighting by attention probabilities to get the final output.

## Results

We trained a probe on the [OpenWebText](https://huggingface.co/datasets/openwebtext) dataset. We used a 125M parameter model with 128 hidden dimensions and 12 attention heads. We used a 128-dimensional output.

## Usage

