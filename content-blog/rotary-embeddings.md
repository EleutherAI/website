---
title: "Rotary Embeddings: A Relative Revolution"
date: 2021-04-19T11:07:40+01:00
draft: False
description: "This is a short description of the page"
mathjax: True
---

Stella Biderman, Sid Black, Charles Foster, Leo Gao, Eric Hallahan, Horace He, Ben Wang, Phil Wang

## TL;DR:
Rotary Position Embedding (RoPE) is a new type of position encoding that unifies absolute and relative approaches. Developed by Jianlin Su in a series of blog posts earlier this year [12, 13], it has already garnered widespread interest in some Chinese NLP circles. However this development is not widely known to the global community, in large part due to the lack of English-language resources. This post walks through the method as we understand it, with the goal of bringing it to the attention of the wider academic community. In general we have found that, across a large suite of setups including regular, linear, and local self-attention, it **either matches or beats all other methods currently available for injecting positional information into transformers.**

## What's the Problem?

Since Vaswani et al., 2017 [16] there have been many schemes introduced for encoding positional information in transformers. When applying self-attention to a given domain, the choice of position encoding typically involves tradeoffs between simplicity, flexibility, and efficiency. For example, learned absolute positional encoding is very simple, but may not generalize while sinusoidal embeddings.

Another major limitation of existing methods is that they do not work with efficient transformers. Methods like T5's relative positional bias [10] require constructing the full $N \times N$ attention matrix between positions, which is not possible when using many of the efficient alternatives to softmax attention, including kernelized variants like FAVOR+ [2].

A principled, easy to implement, and generally-applicable method for relative position encoding---one that works for both vanilla and “efficient” attention---is of great interest. Rotary Position Embedding (RoPE) is designed to address this need.

## What's the Solution?

In this section we introduce and derive the rotary positional embedding. We begin with discussing the intuition, before presenting a full derivation.

#### Intuition

We would like to find a positional encoding function $f(\mathbf{x}, \ell)$ for an item $\mathbf{x}$ and its position $\ell$ such that, for two items $\mathbf{q}$ and $\mathbf{k}$ at positions $m$ and $n$, the inner product between $f(\mathbf{q}, m)$ and $f(\mathbf{k}, n)$ is sensitive only to the values of $\mathbf{q}$, $\mathbf{k}$, and their relative position $m-n$. This is related in spirit to the kernel trick: we are searching for a feature map such that its kernel has certain properties.
A key piece of information is the geometric definition of the dot product between Euclidean vectors:

\begin{equation}
    \mathbf{q} \cdot \mathbf{k} = \lVert \mathbf{q} \rVert \lVert \mathbf{k} \rVert \cos(\theta_{qk})
\end{equation}

In plain English, the dot product between two vectors is a function of the magnitude of individual vectors and the angle between them.
With this in mind, the intuition behind rotary embeddings is that we can represent the token embeddings as complex numbers and their positions as pure rotations that we apply to them. If we shift both the query and key by the same amount, changing absolute position but not relative position, this will lead both representations to be additionally rotated in the same manner---as we will see in the derivation---thus the angle between them will remain unchanged and thus the dot product will also remain unchanged. By exploiting of the nature of rotations, the dot product used in self-attention will have the property we are looking for, preserving relative positional information while discarding absolute position.

The following is an example illustrating the core idea of rotary embeddings—a more rigorous derivation is presented in a subsequent section. Some arbitrary $0 < \varepsilon \leq \frac \pi {2N}$ is chosen, where $N$ is the maximum sequence length. When viewed elementwise on $\mathbf{q}$ and $\mathbf{k}$, with $j$ as the element index, the rotary embedding can be viewed as follows:

\begin{align}
    \mathrm{RoPE}(x, m) &= xe^{mi\varepsilon} \\\\
    \langle \mathrm{RoPE}(q_j, m), \mathrm{RoPE}(k_j, n)\rangle &= \langle q_j e^{mi\varepsilon}, k_j e^{ni\varepsilon} \rangle \\\\
    &= q_j k_j e^{mi\varepsilon} \overline{e^{ni\varepsilon}} \\\\
    &= q_j k_j e^{(m - n)i\varepsilon} \\\\
    &= \mathrm{RoPE}(q_j k_j, m - n)
\end{align}

### Visualization and an Analogy from Physics

Sinusoidal embeddings  

Interferometry is a common technique for measuring small changes in distance through relative phase. Imagining a double-slit interferometer, 

However the difference in phase cannot provide absolute distance: as a periodic function, it is impo

### Derivation

We begin with absolute positional information: for each token, we know where it is in the sequence. However dot products (and therefore attention) do not preserve absolute positional information, so if we encode that positional information in the absolute position of the embeddings, we will lose a significant amount of information. Additionally, absolute position is not particularly meaningful due to the common practice [1, 3, 9, 15] of packing short sentences and phrases together in a single context and breaking up sentences across contexts. Dot products do preserve relative position however, so if we can encode the absolute positional information into the token embeddings in a way that only leverages relative positional information, that will be preserved by the attention function.

While it is common in machine learning to restrict our attention to the real numbers, for rotary embeddings it is mathematically more convenient to use the complex numbers as the base field for our embedding space. Instead of working in the usual $\mathbb{R}^d$, we will work in $\mathbb{C}^{d/2}$ by considering consecutive pairs of elements of the query and key vectors to be a single complex number. Specifically, instead of viewing $\mathbf{q}=(q_0,q_1,q_2,q_3,\ldots,q_{d-1})$ as a $d$-dimensional real vector we view it as $\mathbf{q}=(q_0+iq_1, q_2+iq_3,\ldots q_{d-2} + iq_{d-1})\in\mathbb{C}^{d/2}$. As we will see, casting it in this fashion will make discussing the rotary embeddings easier. If $d$ is odd, we can pad it with a dummy coordinate to ensure things line up correctly. Alternatively, we can simply increase $d$ by one.

Let $\mathbf{q}$ and $\mathbf{k}$ be query and key vectors respectively and let $m$ and $n$ be the absolute positions of the corresponding tokens. Let $f(\mathbf{x}, \ell)$ be the function that takes the token embedding $\mathbf{x}$ for a token in position $\ell$ and outputs a new embedding that contains (in some fashion) the relative positional information. Our goal is to find a "nice" function $f$ that does this. Once the positional information is encoded, we need to compute the inner product like so:

\begin{equation}\label{fg}
    \langle f(\mathbf{q}, m),f(\mathbf{k},n) \rangle = g(\mathbf{q}, \mathbf{k}, m - n)
\end{equation}

where $g(\mathbf{q},\mathbf{k},m-n)$ now represents the pre-softmax logit of the usual attention equation. Writing these three functions in exponential form gives
\begin{align*}
    f(\mathbf{q}, m) &= R_f(\mathbf{q}, m)e^{i\Theta_f(\mathbf{q}, m)}\\\\
    f(\mathbf{k}, n) &= R_f(\mathbf{k}, n)e^{i\Theta_f(\mathbf{k}, n)}\\\\
    g(\mathbf{q}, \mathbf{k}, m - n) &= R_g(\mathbf{q}, \mathbf{k}, m - n)e^{i\Theta_g(\mathbf{q}, \mathbf{k}, m - n)}
\end{align*}

Computing the inner product and equating corresponding components yields

\begin{align*}
    R_f(\mathbf{q}, m) R_f(\mathbf{k}, n) &= R_g(\mathbf{q}, \mathbf{k}, m - n)\\\\
    \Theta_f(\mathbf{q}, m) - \Theta_f(\mathbf{k}, n) &= \Theta_g(\mathbf{q}, \mathbf{k}, m - n)\\\\
\end{align*}

Substituting $m=n$ and applying the initial condition $f(\mathbf{x}, 0) = \mathbf{x}$ gives
$$R_f(\mathbf{q}, m) R_f(\mathbf{k}, m) = R_g(\mathbf{q}, \mathbf{k}, 0) = R_f(\mathbf{q}, 0) R_f(\mathbf{k}, 0) = ||\mathbf{q}||\;||\mathbf{k}||$$ As the prior equation is valid for all $m$, it means that $R_f$ is independent of the value of $m$, so we can set $R_f(\mathbf{x}, y) = ||\mathbf{x}||$. Similarly, if we denote $\Theta(\mathbf{x}) = \Theta_f(\mathbf{x}, 0)$ we obtain $$\Theta_f(\mathbf{q}, m) - \Theta_f(\mathbf{k}, m) = \Theta_g(\mathbf{q}, \mathbf{k}, 0) = \Theta_f(\mathbf{q}, 0) - \Theta_f(\mathbf{k}, 0) = \Theta(\mathbf{q}) - \Theta(\mathbf{k})$$ which implies that $\Theta_f(\mathbf{q}, m) - \Theta(\mathbf{q}) = \Theta_f(\mathbf{k}, m) - \Theta(\mathbf{k})$ for all $\mathbf{q},\mathbf{k},m$. This allows us to decompose $\Theta_f$ as $\Theta_f(\mathbf{x}, y) = \Theta(\mathbf{x}) + \varphi(y)$. Examining the case of $m = n + 1$ reveals that $$\varphi(m) - \varphi(m-1) = \Theta_g(\mathbf{q}, \mathbf{k}, 1) + \Theta(\mathbf{q}) - \Theta(\mathbf{k})$$ Since the right hand side does not depend on $m$, the left hand side must not either and so $\varphi$ is an arithmetic progression. Setting the initial values $\varphi(0)=0$ and $\varphi(1)=\theta$, we have $\varphi(m)=m\theta$.

Putting all of these pieces together, we get the final formula for the rotary positional encoding:
\begin{equation}
    f(\mathbf{q}, m) = R_f(\mathbf{q}, m)e^{i\Theta_f(\mathbf{q}, m)}=||\mathbf{q}||e^{i(\Theta(\mathbf{q})+m\theta)} = \sum_{j=0}^{d/2} ||q_j||e^{im\theta} \vec{e_j}
\end{equation}
where $q_j$ is the $j^{th}$ coordinate of $\mathbf{q}\in\mathbb{C}^{d/2}$ and $\vec{e_j}$ is the $j^{th}$ unit vector of $\mathbb{C}^{d/2}$.

### How is this different from the sinusoidal embeddings used in "Attention is All You Need"?

A response many of us at EleutherAI had when first coming across this was "how does this differ from sinusoidal embeddings," so we feel it is worth discussing this comparison. There are two ways that rotary embeddings are different from sinusoidal embeddings:
1. Sinusoidal embeddings apply to each coordinate individually, while rotary embeddings mix pairs of coordinates
2. Sinusoidal embeddings add a $\cos(m\theta)$ or $\sin(m\theta)$ term, while rotary embeddings use a multiplicative factor.

## Okay, what About in Practice?

After reading  Jianlin Su's original blog posts [12, 13], we were curious how well such a first-principles approach to positional encoding would stack up against existing methods. Despite a tremendous amount of papers that have come out claiming to improve the transformer architecture, very few approaches generalize well across codebases and tasks. However, we have found that rotary positional embeddings perform as well or better than other positional embedding techniques in every architecture we have tried.

### Implementation

A naive implementation of rotary position embeddings would left multiply every query and key at position $m$ by the following block diagonal matrix:
\begin{equation*}
R_{m} = 
\begin{pmatrix}
cos(m\theta_{0}) & -sin(m\theta_{0}) & 0 & 0 & \cdots & 0 & 0 \\\\
sin(m\theta_{0}) & cos(m\theta_{0}) & 0 & 0 & \cdots & 0 & 0 \\\\
0 & 0 & cos(m\theta_{1}) & -sin(m\theta_{1}) & \cdots & 0 & 0 \\\\
0 & 0 & sin(m\theta_{1}) & cos(m\theta_{1}) & \cdots & 0 & 0 \\\\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\\\
0 & 0 & 0 & 0 & \cdots & cos(m\theta_{d/2-1}) & -sin(m\theta_{d/2-1}) \\\\
0 & 0 & 0 & 0 & \cdots & sin(m\theta_{d/2-1}) & cos(m\theta_{d/2-1}) 
\end{pmatrix}
\end{equation*}
This matrix $R_{m}$ consists of a set of rotations that are applied $m$ times to each pair of dimensions by an angle $\theta_{k}$. The $\theta_{k}$ are shared across all positions $m$, and are intended to provide multiple scales of resolution for positional information. 
In practice, implementing rotary positional embeddings this way is highly inefficient and more optimized forms are readily available. The original implementations of RoPE are available in the [roformer](https://github.com/ZhuiyiTechnology/roformer) and [bert4keras](https://github.com/bojone/bert4keras) libraries. Additionally, we have implemented rotary positional embeddings in the [x-transformers](https://github.com/lucidrains/x-transformers) library and the [GPT-Neo](https://github.com/EleutherAI/gpt-neo) and [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) codebases. An example implementation from GPT-NeoX is shown below for reference: 

```python
class RotaryEmbedding(torch.nn.Module):
    
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[:, None, None, :]
            self.sin_cached = emb.sin()[:, None, None, :]
        return self.cos_cached, self.sin_cached
        
# rotary pos emb helpers:
def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=x1.ndim - 1) # dim=-1 triggers a bug in torch < 1.8.0

@torch.jit.script
def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
```
**N.B:** The layout of the queries and keys in GPT-NeoX, following Megatron \cite{megatron}, is `[seq, batch, heads, hdim]`, in order to avoid memory-intensive transpose operations. The code will need to be modified to work with the conventional layout of `[batch, seq, heads, hdim]`.

### Experiments

We have found rotary embeddings to be effective for many varieties of attention:

**Comparison against other PEs for Global attention:** We conducted [comparisons](https://wandb.ai/eleutherai/neox/reports/Rotary-Test-3--Vmlldzo2MTIwMDM) of rotary embeddings with learned absolute positional embeddings, used in GPT-3 [1], and the learned relative positional embeddings (henceforth RPE) used in T5 [10] using our GPT-Neox codebase. Comparisons were done using 125M parameter models with the same hyperparameters as the equally-sized model from [1]. Models were trained on [OpenWebText2]({https://www.eleuther.ai/projects/open-web-text2/), a large and diverse dataset of online text. We see faster convergence of training and validation curves and a lower overall validation loss with a minimal decrease in throughput. 

\begin{figure}[ht]
    \centering
    \includegraphics[width=0.9\textwidth]{Rotary Embedding/rope-learned-rpe.png}
    \caption{OWT2 validation loss with 150M parameter models}
    \label{fig:rope-learned}
\end{figure}

Final validation loss / ppl scores on OWT2 validation set at 55k steps (~30B tokens):

\begin{center}
 \begin{tabular}{c c c} 
 \toprule
 Embedding Type & OWT2 Loss & OWT2 Ppl. \\ [0.5ex] 
 \midrule
 Learned Absolute & 2.809 & 16.59 \\ 
 T5 RPE & 2.801 & 16.46 \\
 Rotary & 2.759 & 15.78 \\
 \bottomrule
\end{tabular}
\end{center}

**Billion+ parameter models:** We additionally conducted additional larger scale experiments with the [mesh-transformer-jax](https://github.com/kingoflolz/mesh-transformer-jax) codebase and 1.4B parameter models, against baselines of learned absolute position embeddings and T5 RPE. Hyperparameters similar to GPT3's 1.3B model were used, with the dataset being the Pile [3]. A similar increase in convergence speed was observed as seen over learned absolute (~30\%), and a smaller improvement (10-20\%) was still seen over the T5 relative position encoding, demonstrating scalability into the billion parameter regimen. For full details, see [here](https://wandb.ai/eleutherai/mesh-transformer-jax/reports/Position-encoding-shootout--Vmlldzo2MTg2MzY).

\begin{figure}[ht]
    \centering
    \includegraphics[width=0.9\textwidth]{Rotary Embedding/jax-experiments.png}
    \caption{Pile validation loss with 1.5B parameter models}
    \label{fig:rope-learned}
\end{figure}

Final validation loss / ppl scores on Pile validation set at 8k steps (~5B tokens):

\begin{center}
 \begin{tabular}{c c c} 
 \toprule
 Embedding Type & Pile Loss & Pile Ppl. \\ [0.5ex] 
 \midrule
 Learned Absolute & 2.24 & 9.393 \\ 
 T5 RPE & 2.223 & 9.234 \\
 Rotary & 2.173 & 8.784 \\
 \bottomrule
\end{tabular}
\end{center}

**Comparison against learned absolute for Performer:** Performer [2] is an example of an alternative attention mechanism designed to avoid quadratic bottlenecks with respect to sequence lengths. We ran small scale tests of Performer on enwiki8, for 8 layer char-based transformers with 512 dimensions and 8 heads. [These tests indicated](https://wandb.ai/lucidrains/eleuther-blogpost/reports/performer-rotary--Vmlldzo2MTgyNDg) that substituting rotary embeddings into the Performer leads to stark decreases in validation loss and to rapid convergence. Though these improvements do not close the gap between efficient and quadratic attention mechanisms, such a significant improvement makes mechanisms like Performer more attractive.

In smaller scale tests, we have also put RoPE head to head against other alternatives including the relative position embedding method of Shaw et al. [11], TUPE [5], and position-infused attention [8], seeing positive results across the board. 
\begin{figure}[ht]
    \centering
    \includegraphics[width=0.9\textwidth]{Rotary Embedding/preformer.png}
    \caption{a nice plot}
    \label{fig:preformer}
\end{figure}

#### Runtime
In general, we find that the runtime cost of rotary embeddings is fairly negligible. With the above implementation, we find that applying the rotary embeddings is naively about 4-5x the cost of applying additive positional embeddings. With the addition of a fusing optimizer like Torchscript, the runtime can be reduced to about 2-2.5x the runtime of additive positional embeddings. Concretely, for query and key tensors of shape $[2048, 16, 12, 64]$, applying rotary embeddings take 5.3 milliseconds, while applying additive positional embeddings takes 2.1 milliseconds.

Unlike standard positional embeddings, however,  rotary embeddings must be applied at every layer. As large transformer models are typically dominated by matrix multiplies, we find that the overall overhead remains negligible. With fusion, we find that rotary embeddings imposes a 1-3\% overhead across a range of transformer sizes.


## Conclusion
Rotary embeddings make it possible to implement relative attention in a straightforward and efficient manner. We are excited to read the upcoming rotary positional embeddings paper from the original authors and the work it inspires. Simple improvements to the transformer architecture that carry over robustly between different types of self-attention are few and far between [6].

With relative ease RoPE can be extended into the multidimensional case. To represent two dimensions, two independent 1-dimensional rotary embeddings can be used. To implement this, we can split each of $\mathbf{q}$ and $\mathbf{k}$ in half and apply rotary piece-wise as follows:

`\begin{align}
    \langle f(\mathbf{q}, m, i),f(\mathbf{k}, n, j) \rangle &= \langle f_1(\mathbf{q}_{:d/2}, m),f_1(\mathbf{k}_{:d/2}, n) \rangle + \langle f_2(\mathbf{q}_{d/2:}, i),f_2(\mathbf{k}_{d/2:}, j) \rangle \\\\
    &= g_1(\mathbf{q}_{:d/2}, \mathbf{k}_{:d/2}, m - n) + g_2(\mathbf{q}_{d/2:}, \mathbf{k}_{d/2:}, i - j) \\\\
    &= g(\mathbf{q}, \mathbf{k}, m - n, i - j)
\end{align}`

This formulation can also be further extended to data of an arbitrary number of dimensions. This sort of multi-dimensional relative coding would let us, for example, implement relative timing and relative pitch embeddings similar to Music Transformer [4] in a drastically simpler manner. More generally, we believe there is potentially a large class of invariances that first-principles positional codes like RoPE may enable us to capture. 

### Citation Information

To cite the RoPE methodology, please use:
```bibtex
@article{rope-paper,
  title={RoFormer: Enhanced Transformer with Rotary Position Embedding},
  author={Su, Jianlin and Lu, Yu and Pan, Shengfeng and Wen, Bo and Liu, Yunfeng},
  journal={arXiv preprint arXiv:2012.15832},
  year={2021}
}
```

To cite this blog post, please use:

```bibtex
@misc{rope-eleutherai,
  title = {Rotary Embeddings: A Relative Revolution},
  author = {Biderman, Stella and Black, Sid and Foster, Charles and Gao, Leo and Hallahan, Eric and He, Horace and Wang, Ben and Wang, Phil},
  howpublished = \url{blog.eleuther.ai/},
  note = {[Online; accessed ]},
  year = {2021}
}
```

## References

[1] Tom B Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. *arXiv preprint arXiv:2005.14165*, 2020.

[2] Krzysztof Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Davis, Afroz Mohiuddin, Lukasz Kaiser, et al. Rethinking attention with performers. *arXiv preprint arXiv:2009.14794*, 2020.

[3] Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, et al. The Pile: An 800GB dataset of diverse text for language modeling. *arXiv preprint arXiv:2101.00027*, 2021.

[4] Cheng-Zhi Anna Huang, Ashish Vaswani, Jakob Uszkoreit, Noam Shazeer, Ian Simon, Curtis Hawthorne, Andrew M Dai, Matthew D Hoffman, Monica Dinculescu, and Douglas Eck. Music transformer. *arXiv preprint arXiv:1809.04281*, 2018.

[5] Guolin Ke, Di He, and Tie-Yan Liu. Rethinking the positional encoding in language pre-training.arXiv preprint arXiv:2006.15595, 2020.

[6] Sharan Narang, Hyung Won Chung, Yi Tay, William Fedus, Thibault Fevry, Michael Matena, Karishma Malkan, Noah Fiedel, Noam Shazeer, Zhenzhong Lan, et al. Do transformer modifications transfer across implementations and applications? *arXiv preprint arXiv:2102.11972*, 2021.

[7] Deepak Narayanan, Mohammad Shoeybi, Jared Casper, Patrick LeGresley, Mostofa Patwary, Vijay Korthikanti, Dmitri Vainbrand, Prethvi Kashinkunti, Julie Bernauer, Bryan Catanzaro, et al. Efficient large-scale language model training on GPU clusters, 2021.

[8] Ofir Press, Noah A Smith, and Mike Lewis. Shortformer:  Better language modeling usingshorter inputs. *arXiv preprint arXiv:2012.15832*, 2020.

[9] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. *arXiv preprint arXiv:2103.00020*, 2021.

[10] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu.  Exploring the limits of transfer learning with a unified text-to-text transformer. *arXiv preprint arXiv:1910.10683*, 2019.

[11] Peter Shaw, Jakob Uszkoreit, and Ashish Vaswani. Self-attention with relative position representations. *arXiv preprint arXiv:1803.02155*, 2018.

[12] Su Jianlin. 让研究人员绞尽脑汁的Transformer位置编码. https://kexue.fm/archives/8130, 2021. [Online; accessed 18-April-2021].

[13] Su Jianlin. Transformer升级之路：2、博采众长的旋转式位置编码. https://kexue.fm/archives/8265, 2021. [Online; accessed 18-April-2021].

[14] Su Jianlin, Yu Lu, Shengfeng Pan, Bo Wen, and Yunfeng Liu. RoFormer: Enhanced Transformer with Rotary Position Embedding. https://kexue.fm/archives/8265, 2021. [Online; accessed 18-April-2021].

[15] Hao Tan and Mohit Bansal. Vokenization: improving language understanding with contextual-ized, visual-grounded supervision. *arXiv preprint arXiv:2010.06775*, 2020.

[16] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. *arXiv preprint arXiv:1706.03762*, 2017.
