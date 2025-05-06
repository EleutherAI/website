---
title: "Studying inductive biases of random networks via local volumes"
date: 2024-12-12T16:00:00-00:00
description: 
author: ["Louis Jaburi", "Nora Belrose"]
ShowToc: true
mathjax: true
draft: false
---

In this post, we will study **inductive biases** of the **parameter-to-function map** of **random neural networks** using **star domain volume estimates**. 
This builds on the ideas introduced in the [Estimating the Probability of Sampling a Trained Neural Network at Random](https://arxiv.org/abs/2501.18812) 
(henceforth [NNstar](https://arxiv.org/abs/2501.18812))
and [Neural Redshift: Random Networks are not Random Functions](https://arxiv.org/abs/2403.02241) (henceforth [NRS](https://arxiv.org/abs/2403.02241)).

## Inductive biases

To understand generalization in deep neural networks, inductive biases are unavoidable. Theoretical results such as universal approximator theorems are NOT the reason for the great success of deep learning:
- Polynomial functions are unable to learn, let alone generalize from, many tasks we care about. Although there are approximations that would work arbitray well (on the training distribuition), gradient based techniques fail to converge towards such solutions in practice.
- Within a family of architechtures, changing a few properties such as the activation function (e.g. from Tanh to ReLU) or the number of layers can substantially change the training dynamic.
- Given a fixed architecture, some tasks will be easily learnable, while others would need exponentially long (see [here](https://www.lesswrong.com/posts/Mcrfi3DBJBzfoLctA/the-subset-parity-learning-problem-much-more-than-you-wanted) and [here](https://arxiv.org/abs/2004.00557))

We would like to understand what training tasks different architectures are (un)able to solve, what kind of solutions they find, and whether they have some shared properties e.g. of geometric kind. More precisely, we would hope that properties of neural networks with a fixed architectures **at initialization** already determine properties of the network **over training**. To do this, we built on [NRS](https://arxiv.org/abs/2403.02241). We summarize a few hypotheses based on their observations

1. **The Neural redshift hypothesis**: Inductive bias complexity (i) exists, (ii) can be read off at initialization and (iii) needs to be close to the complexity of the task.
2. Training increases complexity.
3.  Extreme simplicity bias leads to shortcut learning: The phenomenon where models learn simpler, but more suprious, features of the training distribution. Thus models which are victim to shortcut learning learn worse or not at all.


## Parameter-to-function maps/Geometry of loss landscape

Say we have a supervised task $f:X\to Y$ (where $X$ and $Y$ are finite) and a neural network with weights $w\in W$. Importantly, the weights $w$ induce a function $f_w :\mathbb{R}^{|X|} \to \mathbb{R}^{|Y|}$ and this association $w\mapsto f_w$ is very much non-identfiable, i.e. there are usually many different $w,w'\in W$ that induce the same function $f_w=f_{w'}$ (see e.g. [here](https://arxiv.org/pdf/1905.09803)). This map $w\mapsto f_w$, the parameter-to-function map, has been extensively studied and its properties attributed to the generalization behaviour of neural networks (e.g. [here](https://arxiv.org/abs/1909.11522), [here](https://arxiv.org/abs/2405.10927) and references therein). 
Thus we can think of the loss landscape as a composition of the parameter-to-function map and the induced loss function
$$ L:W\xrightarrow{p} \mathcal{F} \xrightarrow{L_{\mathbb{F}}} \mathbb{R}$$
where  $\mathcal{F}=\\\{ f_w:\mathbb{R}^{|X|} \to \mathbb{R}^{|Y|} \mid w\in W \\\}$ is the function space of all possible $f_w$.
We stress that the parameter-to-function map is completely task independent! It does not take into account the training distribution. 

As explained in the previous section, we are interested in the general inductive bias of neural networks without a specific training task. [TODO: Make this more clear: when does data come in and how]
Therefore, we chose to study the geometry of the parameter-to-function map rather than that of the loss. 
In practice, this means that our cost function $C:W\to \mathbb{R}$ is not the loss function but the KL-divergence $KL(w_0)(w)=\mathbb{E}\_{x\sim \mathcal{D}}[D\_{KL}(f_\{w_0\}(x) \mid\mid f_{w}(x))]$. This serves as a measure of how local change around $w$ affects the induced function $f_w$.


## Star domain volume/Complexity measures

In the first section we mentioned "the" "complexity" of a neural network. 
There is an abundant variety of such measures (see [here](https://arxiv.org/abs/1912.02178), [here](https://arxiv.org/abs/1806.08734), and [here](https://arxiv.org/abs/2308.12108)) with no clear consensus on which one is preferred. 
In the original [NRS](https://arxiv.org/abs/2403.02241) paper, the authors used spectral analysis and LZ-complexity. 
We were excited to replicate and extend the results using the volume measure introduced in [NNstar](https://arxiv.org/abs/2501.18812). 
Our underlying motivation for a volume based measure are the volume basin hypotheses (see [NNstar](https://arxiv.org/abs/2501.18812) for details):

Let $A,B\subset W$ be two different regions - basins - of the parameter space. Intuitively, we think of them as regions corresponding
to different kinds of solutions/behaviour. Then the odds of converging to a solution in $A$ compared to $B$ is roughly determined 
by the ratio of the volumes of the two basins.

In practice, these basins are of the form $\\\{ w\in W \mid C(w)< \epsilon \\\}$, where $C$ is the cost function.
In our case $C(\cdot)=KL(w_0)(\cdot)$, but one could also consider the loss function here. 
Estimating such volumes is difficult as naive approaches fall victim to the curse of dimensionality, either becoming intractable or inaccurate.
To overcome this, we use the star domain: [picture of star domain?]
1. We sample random unit directions $u_i$ at $w_0$.
2. We compute the radii $r_i$ for which $C(w_0+ r\cdot u_i)<\epsilon$ for all $r<r_i$.
3. We compute a **Gaussian integral** along the direction $u_i$ using the the radii $r_i$. 
4. Finally, we normalize by taking the average of samples.

The reason for taking a Gaussian integral (rather than the $r_i ^n$) is that the $r_i$ could be infinite: A direction might not change the cost function at all
and therefore the volume would be infinite along this direction. 
We therefore use the Gaussian integral centered at $0$, as this is the distribution of the weights at initialization.
In a Bayesian setting, this would be the prior distribution of the weights.
(Alternative choices (e.g. centering at $w_0$ as done for [learning coefficients](https://arxiv.org/abs/2308.12108)) exist as well.)

## Experiments

Our setup is as follows:
We take a random neural network, where we vary
- the number of layers from $2$ to $6$
- the activation function $\sigma$ from ReLU, GELU, Tanh, Sigmoid, Gaussian, and a custom activation function Complex multiplication
- The weight scale from from $10 ^{-0.5}$ to $\\sqrt 10 ^\{0.5 \}$ in logarithmic steps
We initialized the network using a uniform distribution scaled by $\frac{1}{\\sqrt(fan_\{in\})}$, which is the [default initialization in PyTorch](https://github.com/pytorch/pytorch/issues/57109).
We tried other initializations, but they did not change the results significantly.

We ran two types of experiments:
1. **Initialization**: We compute the volume of the star domain around the initialization point $w_0$ for different values of $\epsilon$ across the different architectures.
2. **Training**: We train the networks on a simple task (modular addition) and compute the volume of the star domain along the training checkpoints. (say sth about NRS 2)

### 1. Initialization
<iframe src="/images/blog/inductive-bias/image.png" style="width: 100%; height: 600px; border: none;"></iframe>

Over all were not able to replicate the findings in [NRS](https://arxiv.org/abs/2403.02241).
Specifically, we did not observe that higher weight amplitude and additional layers lead to a lower volume of the star domain (as those correspond to more complex solutions according to the volume basin hypothesis).


### 2. Training
<iframe src="/images/blog/inductive-bias/multi_heatmaps.html" style="width: 100%; height: 800px; border: none;"></iframe>

Similary, we did not find that the volume of the star domain is a good predictor for learning behaviour.
While training does generally lead to lower volumes, we observe architectures with similar basin volumes (e.g. ReLU and GELU) but different learning behaviour (more GELUs grokked).
Additionally, complex multiplication had a variety of basin volumes, but consistenly learned the task well and much faster than the other architectures. 
This was partially observed in [NRS](https://arxiv.org/abs/2403.02241) as well.


## Conclusion

Inductive biases are important and play an important role in the generalization behaviour of neural networks.
But it seems unlikely that one single measure can faithfully capture the inductive bias of a neural network via a one-dimensional notion of complexity:
- Specifically, could we say that language is less complex than parity as ReLU networks can learn language but not parity?
Over all, we could not provide further evidence for the neural redshift hypothesis, as we did not observe a correlation between the volume of the star domain and the learning behaviour of the networks. We remain interested in geometric descriptions of the parameter-to-function map and the loss landscape, but ...?

