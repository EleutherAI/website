---
title: "Diff-in-Means Concept Editing is Provably Optimal"
date: 2023-11-13T22:00:00-00:00
description: "Explaining a result by Sam Marks and Max Tegmark"
author: ["Nora Belrose"]
ShowToc: true
mathjax: true
---

# Introduction

Our recent paper [LEACE: Perfect linear concept erasure in closed form](https://arxiv.org/abs/2306.03819)

In a recent paper, [Sam Marks and Max Tegmark](https://arxiv.org/abs/2310.06824) show that the difference-in-means direction in activation space is highly effective at steering models' behavior when intervened on, _and_ is robust to distribution shifts when used for probing. Below, we offer a theoretical explanation for these results by showing that interventions on the difference-in-means direction $\boldsymbol \delta$ are _worst-case optimal_ in the following sense. For any latent concept in the model-- operationalized as an unknown predictor that is admissible (Def. 2) for the concept in question-- edits to the activations along $\mathrm{span}(\boldsymbol \delta)$ will maximize the worst-case magnitude of the change to the latent concept.

# Definitions

**Notation.** In what follows, capital letters $\mathrm{X}$ denote random variables, bold lowercase letters $\boldsymbol{x}$ are deterministic vectors, and non-bold lowercase letters $x$ are deterministic scalars. We write $f_x$ to denote the partial derivative of a function $f$ with respect to a variable $x$. 

**Definition 1.**
The **trivially attainable loss** for labels $\mathrm Z$ and loss $\mathcal{L}$ is the lowest possible expected loss available to a constant predictor $\eta(\mathbf{x}) = \alpha$:
$$
    \mathcal{L}_{\tau} = \inf_{\alpha \in \mathbb R} \mathbb{E} [\mathcal{L}(\mathbf \alpha, \mathrm Z)]
$$

**Definition 2.**
An **admissible predictor** for labels $\mathrm Z$ and loss $\mathcal{L}$ is a linear predictor whose loss is strictly less than the trivially attainable loss $\mathcal{L}_{\tau}$.

**Definition 3.**
A loss function $\mathcal{L}(\eta, z) : \mathbb{R} \times \mathcal \{0, 1\} \rightarrow \mathbb{R}$ is **monotonic** if it monotonically decreases in $\eta$ when $z = 1$, and monotonically increases in $\eta$ when $z = 0$. Equivalently, its derivative wrt $\eta$ satisfies
$$
    \forall \eta \in \mathbb{R} : \mathcal{L}_{\eta}(\eta, 1) \le 0 \le \mathcal{L}_{\eta}(\eta, 0).
$$

Nearly all classification loss functions used in practice meet this criterion, including the categorical cross-entropy loss and the support vector machine hinge loss.

**Definition 4.**
A loss function is **balanced** if its derivative wrt $\eta$ satisfies
$$
    \forall \eta \in \mathbb{R} : \mathbb{P}(\mathrm Z = 0) \cdot \mathcal L_{\eta}(\eta, 0) = -\mathbb{P}(\mathrm Z = 1) \cdot \mathcal L_{\eta}(\eta, 1).
$$

Intuitively, this means that $\mathcal L$ does not treat false positives and false negatives differently. The binary cross-entropy loss is balanced in this sense when either the labels are balanced, i.e. $\mathbb{P}(\mathrm Z = 0) = \mathbb{P}(\mathrm Z = 1) = \frac{1}{2}$, or class weights are used to compensate for class imbalance.

# Theorems

We will now show that the coefficient vectors of **all** admissible predictors must have positive inner product with the difference-in-means direction.

**Theorem 1.**
    Let $\boldsymbol{\delta} = \mathbb{E}[\mathrm X | \mathrm Z = 1] - \mathbb{E}[\mathrm X | \mathrm Z = 0]$ be the difference in class centroids. Suppose $\eta(\mathbf{x}) = \boldsymbol{\beta}^T \mathbf{x} + \alpha$ is admissible for $(\mathrm X, \mathrm Z)$ and convex monotonic loss $\mathcal{L}$. Then $\langle \boldsymbol{\beta}, \boldsymbol{\delta} \rangle > 0$.

**Proof.**
Suppose for the sake of contradiction that $\langle \boldsymbol{\beta}, \boldsymbol{\delta} \rangle \le 0$ and hence
$$
\begin{align}
    0 &\ge \boldsymbol{\beta}^T \big ( \mathbb{E}[\mathrm X | \mathrm Z = 1] - \mathbb{E}[\mathrm X | \mathrm Z = 0] \big ) \\
    &= \mathbb{E}_{\boldsymbol{x}} [ \boldsymbol{\beta}^T \boldsymbol{x} | \mathrm Z = 1] - \mathbb{E}_{\boldsymbol{x}} [ \boldsymbol{\beta}^T \boldsymbol{x} | \mathrm Z = 0] \\
    &= \mathbb{E}_{\boldsymbol{x}} [ \eta(\boldsymbol{x}) | \mathrm Z = 1] - \mathbb{E}_{\boldsymbol{x}} [ \eta(\boldsymbol{x}) | \mathrm Z = 0]
\end{align}
$$
and therefore
$$
\begin{equation}
    \mathbb{E}_{\boldsymbol{x}} [\eta(\boldsymbol{x}) | \mathrm Z = 1] \le \mathbb{E}_{\boldsymbol{x}} [\eta(\boldsymbol{x})] \le \mathbb{E}_{\boldsymbol{x}} [\eta(\boldsymbol{x}) | \mathrm Z = 0].
\end{equation}
$$
We can now show that the expected loss is lower bounded by the trivially attainable loss $\mathcal{L}_{\tau}$:
$$
\begin{align}
    \mathbb{E}_{(\boldsymbol{x}, z)} \big[\mathcal{L}(\eta(\boldsymbol{x}), z)\big]
    &= \mathbb{E}_z \big[  \mathbb{E}_{\boldsymbol{x}} \big[\mathcal{L}(\eta(\boldsymbol{x}), z) \big| z \big] \big] \tag{law of total expectation} \\
    &\ge \mathbb{E}_z \big[ \mathcal{L}\Big( \mathbb{E}_{\boldsymbol{x}} \big[ \eta(\boldsymbol{x}) \big| z \big], z \Big) \big] \tag{Jensen's inequality} \\
    &\ge \mathbb{E}_z \big[ \mathcal{L}\Big( \mathbb{E}_{\boldsymbol{x}} \big[ \eta(\boldsymbol{x}) \big], z \Big) \big] \tag{Eq. 1 and Monotonicity of $\mathcal{L}$} \\
    &\ge \mathcal{L}_{\tau}. \tag{Def.}
\end{align}
$$

The penultimate step is justified because, by Eq. 1 and the monotonicity of $\mathcal L$, replacing $\mathbb{E}_{\boldsymbol{x}} [\eta(\boldsymbol{x}) | \mathrm Z = 0]$ with $\mathbb{E}_{\boldsymbol{x}} [\eta(\boldsymbol{x})]$ can only decrease the loss on examples where $\mathrm Z = 0$, and replacing $\mathbb{E}[\eta | \mathrm Z = 1]$ with $\mathbb{E}_{\boldsymbol{x}} [\eta(\boldsymbol{x})]$ can only decrease the loss on examples where $\mathrm Z = 1$.

If $\mathbb{E}_{(\boldsymbol{x}, z)} \big[\mathcal{L}(\eta(\boldsymbol{x}), z)\big] \ge \mathcal{L}_{\tau}$, the classifier cannot be admissible (Def.~\ref{def:admissible-predictor}), contradicting our earlier assumption. Therefore the admissibility of $\eta$ implies $\langle \boldsymbol{\beta}, \boldsymbol{\delta} \rangle > 0$.

**Theorem 2.** Suppose $\mathcal L$ is a convex, monotonic, and balanced loss function. Then if $\langle \boldsymbol{\beta}, \boldsymbol{\delta} \rangle > 0$, there exist constants $\alpha, c$ with $c > 0$ such that $\eta(\mathbf{x}) = c \boldsymbol{\beta}^T \mathbf{x} + \alpha$ is admissible for $\mathcal L$.

**Proof.** If there are any $(\alpha, c)$ that make $\eta$ admissible for $\mathcal L$, the values that _minimize_ $\mathbb E[\mathcal L]$ would be among them. Hence we may assume the first-order optimality condition
$$
\mathbb E[\mathcal{L}_{\alpha}(\boldsymbol{x}, z)] = \mathbb E [\mathcal{L}_{c}(\boldsymbol{x}, z)] = 0.
$$
If $c = 0$, $\eta$ is an optimal constant predictor and it achieves the trivially attainable loss $\mathcal{L}_{\tau}$. If $c < 0$, this would imply $\langle c \boldsymbol{\beta}, \boldsymbol{\delta} \rangle < 0$ and hence the expected loss would be no better than $\mathcal{L}_{\tau}$ by Theorem 1. Therefore we may assume $c \ge 0$.

We will now show $c \neq 0$. Suppose for the sake of contradiction that $c = 0$. Since $\eta$ is a constant function, $\mathcal{L}_{\eta}$ is also constant for each $z \in \{0, 1\}$. This allows us to rewrite the optimality condition as
$$
\begin{align*}
    0 &= \mathbb E[\mathcal{L}_{c}(\boldsymbol{x}, z)] \\
    &= \mathbb P(\mathrm{Z} = 1) \cdot \mathbb E[\mathcal{L}_{\eta}(\boldsymbol{x}, 1) \cdot \eta_{c}(\boldsymbol{x}) | \mathrm{Z} = 1] + \mathbb P(\mathrm{Z} = 0) \cdot \mathbb E[\mathcal{L}_{\eta}(\boldsymbol{x}, 0) \cdot \eta_{c}(\boldsymbol{x}) | \mathrm{Z} = 0] \\
    &= \mathbb P(\mathrm{Z} = 1) \cdot \mathcal{L}_{\eta}(\boldsymbol{x}, 1) \cdot \mathbb E[ \boldsymbol{\beta}^T \boldsymbol{x} | \mathrm{Z} = 1] + \mathbb P(\mathrm{Z} = 0) \cdot \mathcal{L}_{\eta}(\boldsymbol{x}, 0) \cdot \mathbb E[ \boldsymbol{\beta}^T \boldsymbol{x} | \mathrm{Z} = 0] \\
    &= \mathbb P(\mathrm{Z} = 1) \cdot \mathcal{L}_{\eta}(\boldsymbol{x}, 1) \cdot \boldsymbol{\beta}^T \big ( \mathbb E[ \boldsymbol{x} | \mathrm{Z} = 1] - \mathbb E[ \boldsymbol{x} | \mathrm{Z} = 0] \big ) \tag{Def. 4} \\
    &= \boldsymbol{\beta}^T \boldsymbol{\delta},
\end{align*}
$$
which contradicts our assumption that $\langle \boldsymbol{\beta}, \boldsymbol{\delta} \rangle > 0$. Therefore $c > 0$.

# Additive concept edits

We define an **additive edit** as a transformation which replaces a feature vector $\boldsymbol{x}$ with $\boldsymbol{x}' = \boldsymbol{x} + a \boldsymbol{u}$ for some unit vector $\boldsymbol{u} \notin \mathrm{span}(\boldsymbol{x})$ and some scalar ``intensity'' $a \neq 0$. The goal of such an edit is to move the latent prediction in a specified _direction_; that is, to ensure $\tau = \eta(\boldsymbol{x}') - \eta(\boldsymbol{x})$ has the desired sign. We also expect that the magnitude of $\tau$ should monotonically increase with $|a|$.

Since we assume $\eta$ is linear, an additive edit will be successful in this minimal sense if and only if the edit direction is in the half-space of the latent coefficients $\boldsymbol{b}$; that is
$$
    \nabla_{\boldsymbol{u}} \eta(\boldsymbol{x}) = \boldsymbol{u}^T \Big(\frac{\partial \eta}{\partial \boldsymbol{x}}\Big) = \boldsymbol{u}^T \boldsymbol{b} > 0,
$$
where $\nabla_{\boldsymbol{u}} \eta(\boldsymbol{x})$ denotes the directional derivative of $\eta(\boldsymbol{x})$ along $\boldsymbol{u}$. In general we would like $\nabla_{\boldsymbol{u}} \eta(\boldsymbol{x})$ to be as large as possible, since this achieves a larger effect size $|\tau|$ with a constant-norm change to $\boldsymbol{x}$:
$$
    |\tau| = |\eta(\boldsymbol{x}') - \eta(\boldsymbol{x})| = |\boldsymbol{b} (\boldsymbol{x} + a \boldsymbol{u}) - \boldsymbol{b} \boldsymbol{x}| = |a \boldsymbol{b}^T \boldsymbol{u}| = | a \nabla_{\boldsymbol{u}}\eta(\boldsymbol{x}) |.
$$
By the Cauchy-Schwartz inequality, Eq.~\ref{eq:effect-size} is maximized when $\boldsymbol{u} \in \mathrm{span}(\boldsymbol{b})$, so optimal additive editing is trivial when $\boldsymbol{b}$ is known.

\paragraph{Maximin additive edits.} When $\boldsymbol{b}$ is unknown, we can successfully perform additive edits by selecting $\boldsymbol{u}$ to maximize the **{worst-case} directional derivative.

**Theorem 2.**
Let $\mathrm X$ and $\mathrm Z$ be random vectors taking values in $\mathbb{R}^d$ and $\{0, 1\}$ respectively. Let $H$ denote the set of all admissible predictors $\eta : \mathbb{R}^d \rightarrow \mathbb{R}$ for $(\mathrm X, \mathrm Z)$ of the form $\eta(\boldsymbol{x}) = \boldsymbol{b} \boldsymbol{x} + \alpha$. Then the maximin directional derivative objective
$$
    \underset{\| \boldsymbol{u} \| = 1}{\mathrm{argmax}} \inf_{\eta \in H} \nabla_{\boldsymbol{u}} \eta(\boldsymbol{x})
$$
is maximized by the difference-in-means direction $\boldsymbol{u}^* = \frac{\boldsymbol{\delta}}{\| \boldsymbol{\delta} \|}$.


**Proof.** Consider the orthogonal decomposition of $\boldsymbol{b}$ into $S = \mathrm{span}(\boldsymbol{\delta})$ and $S^\perp$:
$$
    \nabla_{\boldsymbol{u}} \eta(\boldsymbol{x}) = \boldsymbol{u}^T \boldsymbol{b} = \boldsymbol{u}^T (\boldsymbol{b}_S + \boldsymbol{b}_{S^\perp} ) = b \boldsymbol{u}^T \boldsymbol{\delta} + \boldsymbol{u}^T \boldsymbol{b}_{S^\perp},
$$
where $b$ is a positive scalar by Theorem~\ref{thm:admissible-halfspace}. By Cauchy-Schwartz, the first term is maximized when $\boldsymbol{u} = c \boldsymbol{\delta}$ for some $c > 0$, and hence $\boldsymbol{u} \in S$, no matter the value of $\boldsymbol{b}$.

Since $\boldsymbol{b}_{S^\perp}$ is free to take any value in $S^\perp$, we cannot lower bound $\boldsymbol{u}^T \boldsymbol{b}_{S^\perp}$ unless $\boldsymbol{u}^T \boldsymbol{v} = 0$ for every $\boldsymbol{v}$ in $S^\perp$. To see this, suppose $\boldsymbol{u}^T \boldsymbol{v} \neq 0$ for some $\boldsymbol{v} \in S^\perp$. Then we can select $\boldsymbol{b}$ such that $\boldsymbol{b}_{S^\perp} = \lambda \boldsymbol{v}$ for any arbitrarily large $\lambda \in \mathbb{R}$, with the appropriate sign so that $\boldsymbol{u}^T \boldsymbol{b}_{S^\perp}$ is an arbitrarily large negative value. Hence the second term is maximized when $\boldsymbol{u} \in (S^\perp)^\perp = S$.

Since the optima for the first term are also optima for the second term, **a fortiori** they are optimal for the original objective. Since we are imposing a unit norm constraint on $\boldsymbol{u}$, we have $\boldsymbol{u}^* = \frac{\boldsymbol{\delta}}{\| \boldsymbol{\delta} \|}$.
