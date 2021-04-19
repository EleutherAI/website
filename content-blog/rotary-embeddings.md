---
title: "Rotary Embeddings: A Relative Revolution"
date: 2021-04-19T11:07:40+01:00
draft: False
description: "This is a short description of the page"
mathjax: True
---

Rotary position embedding (RoPE) is a new type of position encoding that unifies absolute and relative attention. Developed by Su Jianlin in a series of blog posts earlier this year, it has already garnered widespread interest in some Chinese NLP circles. However, this development is not widely known to the global community, in large part due to the lack of English-language resources. This post walks through the method as we understand it. We have found that, across a large suite of setups including regular, local, and linear attention, it either matches or beats all other methods currently available for injecting positional information into transformers.

**Disclaimer: Rotary embeddings *are not* a development of EleutherAI or associated contributors. Please cite the original authors by way of their original blog post and upcoming papers.**

### Motivation

Since "Attention is All You Need", there have been many schemes introduced for encoding positional information in transformers. When applying self-attention to a given domain, the choice of positional embeddings involves tradeoffs between flexibility, efficiency, and performance. For example, learned absolute position embeddings may not generalize; the network must separately learn that the relationship between positions $n$ and $n+1$ in a sequence is similar to that between positions $n+1$ and $n+2$. Often, relative position is more salient than absolute. 

Even methods for relative positional embeddings are not silver bullets. Methods like T5's relative positional bias require us to construct the attention matrix between positions, which is not possible when using some efficient alternatives to softmax attention [cite linear transformers and efficient attention].

Therefore, a principled, easy to implement, and generally-applicable method for $1D$ relative position embeddings---one that works for both vanilla and “efficient” attention---is of great interest. Rotary position embeddings (called RoPE by the original author) are designed to address this need.

### Intuition

We would like to find a positional encoding function $f(x, t)$ such that the inner product between $f(q, n)$ and $f(k, m)$ is sensitive only to the values of $q$, $k$, and their relative position $n-m$. This is conceptually similar to the kernel trick: we are searching for a feature map such that its kernel has certain properties.

A key piece of information is the dot product between complex numbers:

$$q \cdot k = \lVert q\rVert \lVert k\rVert \cos\left(\theta_{qk}\right)$$

In plain English, the dot product between two vectors is a function of the individual vectors and the angle between them.

With this in mind, the intuition behind rotary embeddings is that we can represent the token embeddings as complex numbers and their positions as pure rotations that we apply to them. If we shift both the query and key by the same amount, changing absolute position but not relative position, this will lead both representations to be additionally rotated in the same manner---as we will see in the derivation---thus the angle between them will remain unchanged. By exploiting of the nature of rotations, the dot product used in self-attention will have the property we are looking for, preserving relative positional information while discarding absolute position.

### Derivation

As a reminder, the formula for "vanilla" attention is

$$\mathcal{A}(\mathbf{X},\mathbf{W}_Q,\mathbf{W}_K,\mathbf{W}_V,d) := \mathrm{Softmax}\left(\frac{\mathbf{Q}\cdot\mathbf{K}}{\sqrt{d}}\right)\mathbf{V}$$

where $\mathbf{Q}$ is the query, $\mathbf{K}$ is the key, $\mathbf{V}$ is the value, $\mathbf{W}$'s are the corresponding weights, $\mathbf{X}$ is the input, and $d$ is the dimension of the embedding space. Throughout this derivation we will talk about single-headed attention, though it generalizes immediately to multi-headed attention.

We begin with absolute positional information: for each token we know where it is in the sequence. However dot products (and therefore attention) doesn't preserve absolute positional information, so if we encode that positional information in the absolute position of the embeddings we will lose a significant amount of information. Dot products do preserve relative position however, so if we can encode the absolute positional information into the token embeddings in a way that only leverages relative positional information, that will be preserved by the attention function.

Let $\mathbf{q}$ and $\mathbf{k}$ be a query and key vectors respectively and let $n$ and $m$ be the absolute position of the corresponding tokens. Let $f(\mathbf{x}, i)$ be the function that takes the token embedding $\mathbf{x}$ for a token in position $i$ and outputs a new embedding that contains (in some fashion) the positional information. Let $g(\mathbf{q}, k, d)$ take a query embedding, key embedding, and the distance between the corresponding tokens and output the dot product of the query and key, with the additional relative positional information encoded. Our goal is to find "nice" functions $f$ and $g$ such that

$$\langle f(\mathbf{q}, m),f(\mathbf{k},n) \rangle = g(\mathbf{q}, \mathbf{k}, m - n)$$

While it is common in machine learning to restrict our attention to the real numbers, for rotary embeddings it is mathematically more convenient to use the complex numbers as the base field for our embedding space. Formally we actually have two embedding spaces, $V=\mathbb{R}^d$ being the space that $\mathbf{q}$ and $\mathbf{k}$ inhabit and $W=\mathbb{C}^d$ being the space that we map the token embeddings to when we encode positional information. In particular, $f:V\times\mathbb{N}\to W$ and $g:V\times V\times\mathbb{Z}\to\mathbb{C}$. It is important to note that we are considering both $V$ and $W$ as $d$-dimensional vector spaces, with the first being a vector space over $\mathbb{R}$ and the second a vector space over $\mathbb{C}$.

Using the exponential form of complex numbers, we have

$$\begin{align}
    f(\mathbf{q}, m) &= R_f(\mathbf{q}, m)e^{i\Theta_f(\mathbf{q}, m)}\\\\
    f(\mathbf{k}, n) &= R_f(\mathbf{k}, n)e^{i\Theta_f(\mathbf{k}, n)}\\\\
    g(\mathbf{q}, \mathbf{k}, m - n) &= R_g(\mathbf{q}, \mathbf{k}, m - n)e^{i\Theta_g(\mathbf{q}, \mathbf{k}, m - n)}
\end{align}$$

Computing the inner product and equating corresponding components yields

$$\begin{align*}
    R_f(\mathbf{q}, m) R_f(\mathbf{k}, n) &= R_g(\mathbf{q}, \mathbf{k}, m - n)\\\\
    \Theta_f(\mathbf{q}, m) - \Theta_f(\mathbf{k}, n) &= \Theta_g(\mathbf{q}, \mathbf{k}, m - n)\\\\
\end{align*}$$

Substituting $m=n$ and applying the initial condition $f(\mathbf{x}, 0) = \mathbf{x}$ gives
$$R_f(\mathbf{q}, m) R_f(\mathbf{k}, m) = R_g(\mathbf{q}, \mathbf{k}, 0) = R_f(\mathbf{q}, 0) R_f(\mathbf{k}, 0) = ||\mathbf{q}||\cdot||\mathbf{k}||$$ This means that $R_f$ is independent of the value of $m$, so we can set $R_f(\mathbf{x}, y) = ||\mathbf{x}||$. Similarly, if we denote $\Theta(\mathbf{x}) = \Theta_f(\mathbf{x}, 0)$ we obtain $$\Theta_f(\mathbf{q}, m) - \Theta_f(\mathbf{k}, m) = \Theta_g(\mathbf{q}, \mathbf{k}, 0) = \Theta_f(\mathbf{q}, 0) - \Theta_f(\mathbf{k}, 0) = \Theta(\mathbf{q}) - \Theta(\mathbf{k})$$ which implies that $\Theta_f(\mathbf{q}, m) - \Theta(\mathbf{q}) = \Theta_f(\mathbf{k}, m) - \Theta(\mathbf{k})$ for all $\mathbf{q},\mathbf{k},m$. This allows us to decompose $\Theta_f$ as $\Theta_f(\mathbf{x}, y) = \Theta(\mathbf{x}) + \varphi(y)$. Examining the case of $m = n + 1$ reveals that $$\varphi(m) - \varphi(m-1) = \Theta_g(\mathbf{q}, \mathbf{k}, 1) + \Theta(\mathbf{q}) - \Theta(\mathbf{k})$$ Since the right hand side does not depend on $m$, the left hand side must not either and so $\varphi$ is an arithmetic progression. Setting the initial values $\varphi(0)=0$ and $\varphi(1)=\theta$, we have $\varphi(m)=m\theta$.

Putting all of these pieces together, we get the final formula for the rotary positional encoding:
$$f(\mathbf{q}, m) = R_f(\mathbf{q}, m)e^{i\Theta_f(\mathbf{q}, m)}=||\mathbf{q}||e^{i(\Theta(\mathbf{q})+m\theta)} = qe^{im\theta}$$