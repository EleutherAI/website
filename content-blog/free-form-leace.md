---
title: "Free Form Least-Squares Concept Erasure Without Oracle Concept Labels"
date: 2024-05-24T16:00:00-00:00
description: "Achieving even more surgical edits than LEACE without concept labels at inference time."
author: ["Brennan Dury"]
ShowToc: true
mathjax: true
draft: false
---

_This post assumes some familiarity with the idea of concept erasure and our LEACE concept erasure method. We encourage the reader to consult our [arXiv paper](https://arxiv.org/abs/2306.03819) for background._

In our paper [LEACE: Perfect linear concept erasure in closed form](https://arxiv.org/abs/2306.03819), we derived a concept erasure method that is least squares optimal within the class of affine transformations. We now extend this result by deriving the least squares optimal edit, only requiring that the edited representation is a function of the unedited representation.

In a previous [blog post](https://blog.eleuther.ai/oracle-leace/), we solved this problem in the case where the transformation may depend both on the representation $\mathbf x$ and the label $\mathbf z$. This assumption allows a more surgical edit than the edit found here, but requires access to labels at inference time and comes at the cost of possibly injecting non-linearly represented information into the representation, as discussed in the blog post. By not assuming oracle access, we solve both issues.

In Theorem 1 below, we derive **Free Form LEACE** ("FF-LEACE"), a closed-form formula for the function $r : \R^n \rightarrow \R^n$ nearest to the identity function such that no linear classifier can do better than chance at predicting $\mathrm Z$ from $\mathrm r(X)$, or equivalently $\mathrm{Cov}(\mathrm r(X), \mathrm Z) = \textbf{0}$.

We prove a more general result, where we are interested in the case $\Omega = \R^n$ and $h(x) = x$.

**Theorem 1.**

Let $X$ be a random object taking values in the set $\Omega$, and let $Z$ be a random vector in $\R^k$. Let $h: \Omega \rightarrow \R^n$ be arbitrary and define $f : \Omega \rightarrow \R^k$ such that $f(x) = \mathbb{E}[Z | X=x]$. Assume $h(X)$ and $f(X)$ have finite second moments. Then for every p.s.d. inner product $\langle \mathbf a, \mathbf b \rangle\_{\mathbf M} = \mathbf a^T \mathbf M \mathbf b$ on $\mathbb{R}^d$, the objective

$$
    \mathop{\mathrm{inf \hspace{0.5em}}}\_{\substack{\mathrm r : \Omega \rightarrow R^n}} \mathbb{E} \big\| \mathrm r(X) - \mathrm h(X) \big\|^2\_{\mathbf M} \quad \mathrm{s.t.} \hspace{0.5em} \mathrm{Cov}(\mathrm r(X), \mathrm Z) = \mathbf{0}
$$

is minimized by:

$$
    r^*(x) = h(x) - \mathbf{\Sigma}\_{h(X)Z} \mathbf{\Sigma}\_{f(X)f(X)}^+ \big(\mathrm f(x) - \mathbb{E}[\mathrm Z]\big)
$$

where $\mathbf{A}^{+}$ denotes the Moore-Penrose pseudoinverse of a matrix $\mathbf{A}$.

**Proof.**

Observe that $\mathrm{Cov}(r(X), Z) = \mathrm{Cov}(r(X), f(X))$, so we rewrite the objective as:

$$
    P\_1 = \mathop{\mathrm{inf \hspace{0.5em}}}\_{\substack{\mathrm r : \Omega \rightarrow R^n}} \mathbb{E} \big\| \mathrm r(X) - \mathrm h(X) \big\|^2\_{\mathbf M} \quad \mathrm{s.t.} \hspace{0.5em} \mathrm{Cov}(\mathrm r(X), \mathrm f(X)) = \mathbf{0}
$$

We first consider a relaxed objective over random variables instead of functions:

$$
    P\_2 = \mathop{\mathrm{inf \hspace{0.5em}}}\_{\substack{\mathrm Y \in \mathcal H^d}} \mathbb{E} \big\| \mathrm Y - \mathrm h(X) \big\|^2\_{\mathbf M} \quad \mathrm{s.t.} \hspace{0.5em} \mathrm{Cov}(\mathrm Y, \mathrm f(X)) = \mathbf{0}
$$

Any function $r : \Omega \rightarrow R^n$ that is feasible for $P\_1$ corresponds to a random variable $r(X)$ that is feasible for $P\_2$. So $P\_2 \leq P\_1$.

We showed in the previous [blog post](https://blog.eleuther.ai/oracle-leace/) that $P\_2$ is minimized by the (appropriately shifted) ordinary least squares residuals from regressing $\mathrm h(X)$ on $\mathrm f(X)$:

$$
    \mathrm Y\_{\mathrm{LEACE}} = \mathrm h(X) - \mathbf{\Sigma}\_{h(X)f(X)} \mathbf{\Sigma}\_{f(X)f(X)}^+ \big(\mathrm f(X) - \mathbb{E}[\mathrm f(X)]\big)
$$

Notice that $Y\_{\mathrm{LEACE}} = r^*(X)$, where $r^*(x) = h(x) - \mathbf{\Sigma}\_{h(X)f(X)} \mathbf{\Sigma}\_{f(X)f(X)}^+ \big(\mathrm f(x) - \mathbb{E}[\mathrm f(X)]\big)$. Since $r^*$ is feasible for $P\_1$ and $Y\_{\mathrm{LEACE}}$ is optimal for $P\_2$, $P\_1 \leq P\_2$.

So $P\_1 = P\_2$, with minimizer $r^*$. Since $E[f(X)] = \mathbb{E}[Z]$ and $\mathbf{\Sigma}\_{h(X)f(X)} = \mathbf{\Sigma}\_{h(X)Z}$, we rewrite $r^*(x) = h(x) - \mathbf{\Sigma}\_{h(X)Z} \mathbf{\Sigma}\_{f(X)f(X)}^+ \big(\mathrm f(x) - \mathbb{E}[\mathrm Z]\big)$.