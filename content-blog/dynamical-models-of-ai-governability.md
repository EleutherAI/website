---
title: "Dynamical Models of AI Governability"
date: 2026-05-20T00:00:00-00:00
description: "A toy dynamical model of whether the AI workforce that builds future AI ends up cooperative or uncooperative: where the basin boundary lies, what current evidence says about which side we are on, and what would tell us we are on the good path."
author: ["David Johnston"]
ShowToc: true
mathjax: true
draft: true
---

# Introduction

AI labour that builds the next generation of AI is no longer hypothetical. OpenAI reports that Codex is now used by over 92% of its technical staff, reviews effectively every internal pull request, and is associated with roughly 70% more merged PRs among the engineers who use it; OpenAI staff describe using Codex to build Codex itself ([OpenAI, 2025](https://forum.openai.com/public/blogs/how-openai-engineers-use-codex-to-tackle-big-projects-with-rigor-2025-12-04); [event replay](https://forum.openai.com/public/clubs/developers-and-startups-y6l6m/videos/event-replay-vibe-engineering-with-openais-codex-2025-12-03)). Anthropic likewise says it "build[s] Claude with Claude": its engineers use Claude Code daily and every new model is first pressure-tested on Anthropic's own work, with an internal study reporting heavy daily use for debugging, code understanding, feature implementation, and design and planning ([Anthropic, 2026](https://www.anthropic.com/research/how-ai-is-transforming-work-at-anthropic)). Automation of AI development also includes automation of substantial chunks of alignment research. Anthropic has run [Claude Opus 4.6](https://www.anthropic.com/news/claude-opus-4-6) agents as "Automated Alignment Researchers" on a weak-to-strong supervision problem: against a human baseline that reached a performance-gap-recovered (PGR) of 0.23, nine Claude agents reached PGR 0.97 after about 800 cumulative research hours, at a cost of roughly $18,000 ([Anthropic, 2026](https://www.anthropic.com/research/automated-alignment-researchers); [Alignment Science blog](https://alignment.anthropic.com/2026/automated-w2s-researcher/)). AI that does AI development now exists, and works on both advancing the capability frontier and alignment efforts. Some of that labour cooperates with the aims of its developers; a measurable fraction, today, does not. Whether the workforce that builds future AI ends up mostly cooperative or mostly uncooperative is the question this post tries to put a model under.

Arguments about deceptive alignment tend to lean on an *implicit* dynamical model. The story goes roughly:

- a misaligned AI has some drive to expand the footprint of its covert actions;
- oversight tools try to detect and patch those actions;
- while capability is low, oversight wins often enough that caught lies get corrected;
- once capability, situational awareness, and commitment cross some threshold, oversight becomes too unreliable to keep up, and the system settles into a stable regime in which covert action proceeds essentially unchecked.

Notice that the story above sets up a state space (capability, prevalence of misaligned action) and dynamics over it (that is, rules for how these states change over time). This is a simple model, and we can see that it supports the conclusion that high capability deception is stable without writing anything formal. However, if we want to accurately capture the trajectory of AI development, it quickly gets more complicated. For example, more capable AI doesn't only evade detection better when misaligned; when aligned, it helps produce better detection systems. Now there are two regimes: successful oversight and successful deception, and a little bit of formal work can help us say more clearly where the boundary between these regimes lies.

In this post we'll be exploring a dynamical model of AI governability. We will see how it can unify dynamics from a diverse range of arguments about AI misalignment and automated alignment research, how it can clarify important questions like "what's the boundary between regimes with good and bad outcomes?" and how it can help us understand the importance of policy interventions and new pieces of evidence. The model we discuss is a toy in that it qualitatively captures important dynamics, but is far from being able to offer credible probabilistic predictions of outcomes. With substantial effort, this line of work may be able to offer such predictions, and we also discuss some possibilities for empirically calibrating models of this type. We can think of it as a first step toward modelling AI governability with the seriousness that the [AI Futures model](https://www.aifuturesmodel.com/) brings to modelling capability.

This work sits alongside a few existing strands. Quantitative takeoff models — Davidson's [compute-centric takeoff framework](https://takeoffspeeds.com/) (made interactive [in collaboration with Epoch](https://epoch.ai/blog/interactive-model-of-takeoff-speeds)) and the [AI Futures model](https://www.aifuturesmodel.com/) — model how fast AI capability compounds, but not whether the resulting workforce stays governable; our model is deliberately agnostic about speed (it cancels out of the dynamics) and models only the governability split. On the risk side, Christiano's [What failure looks like](https://www.alignmentforum.org/posts/HBxe6wdjxK239zajf/what-failure-looks-like) (Part 1) describes a gradual handoff to systems optimising proxies that slowly diverge from what we want — our default path can be read as a first attempt to formalise that scenario, with leakage playing the role of proxy divergence. And the deceptive alignment literature ([Hubinger et al.](https://arxiv.org/abs/1906.01820)) supplies the threshold story we opened with, which appears here as the bistable regime.

A note on terminology: we say "uncooperative" rather than "misaligned" throughout. The post gives "uncooperative" a specific two-part definition below, while "misaligned" is used in many different senses in the literature; where we discuss others' scenarios we map their "misaligned" onto our "uncooperative" explicitly.

Everything in this post can be explored interactively: the [basin explorer](/basin-explorer/) implements the full model with every parameter on a slider, one-click presets for the calibrations we discuss, and an outcome map showing which parameter combinations lead where.

## Explaining the model


### What is being modelled

We're not modelling whether the outcomes are ultimately good, all things considered. We're modelling whether AI that produces the next generation of AI becomes mostly cooperative or mostly uncooperative.

**Uncooperative vs cooperative vs human labour:** Consider the AI labour that goes into building *more* AI, which we'll call $A$. Some of this labour might go into producing future systems that become harder to oversee, more resistant to correction and better able to produce successor systems with similar tendencies. Other labour will go into producing future systems that help to improve oversight, make corrections more robust and well-targeted and better able to produce successor systems with similar tendencies. Both directions are self-reinforcing: if the next generation of AI systems is dominated by uncooperative labour, then the subsequent generation will be even more uncooperative, and vice versa.

We also have human labour $H_0$ that goes into building more AI. While in reality this will change with investment in AI, we treat it as fixed for simplicity.

Here $A$ should be read as an index of effective AI-development labour, not as a raw input such as GPU-hours, tokens or tasks completed. The units are chosen so that AI and human effort can be compared on the same scale: if $A=H_0$, then removing AI labour would roughly halve gross AI-development progress; if $A=2H_0$, it would remove roughly two thirds. This is a simplification. In reality, human and AI labour may be complements in some tasks and substitutes in others. We mostly leave those details inside the process by which $A$ grows over time, which this model does not try to predict. What matters here is how the growing AI-development workforce is split between cooperative and uncooperative contributions.

The cooperative vs uncooperative distinction is a critical feature of this model, and it is worth being precise about what it does and does not assume. By construction, the model only tracks *reproductively viable* AI labour: systems that are not useful enough to help build the next generation fall out of the pool $A$ entirely. Cooperativeness is an *additional* property layered on top of viability. The two are loosely coupled — cooperativeness can often make a system more useful — but viability is probably possible without cooperativeness. It's clear that modern AI systems need not be cooperative: had OpenAI not put substantial effort into making ChatGPT cooperate with users and with its own higher-level policies, it would in fact be much less cooperative in a commonsense manner. While less cooperative systems today would probably be less helpful on AI development tasks, it's probably in principle possible that descendants of these systems trained on, for example, turning short run profits, could be both quite uncooperative and effective at maintaining their viability.

We tentatively define cooperative AI labour in two parts: (i) it is cooperative according to contextually appropriate commonsense standards and (ii) it has a tendency to instil both (i) and (ii) in successors. We tentatively define uncooperative AI labour as a mirror of this: (i') it undermines or evades measures aimed at maintaining AI cooperativeness and (ii') it has a tendency to instill both (i') and (ii') in successors. This leaves a significant "middle" consisting of stuff that, for example, involves a cooperative disposition (i) without acting to preserve this disposition (ii). The reason for doing this, in short, is that considering the full range of preservation-dispositions would require a substantially more complicated model and in this first iteration of the work we value simplicity highly (see [Appendix: neglect, subversion and more realistic models](#appendix-neglect-subversion))

In particular, if we suppose both cooperative and uncooperative labour have the same disposition toward preserving their important characteristics in following generations, then we have some basis for considering the capability of each pool to reproduce itself to be equal. This simplification can be informally justified by observing that the long-run impact of transient dispositions may ultimately be drowned out by the impact of self-preserving dispositions, but ultimately we think there's ample room for this to end up an oversimplification.

Our model includes "leakage" from the cooperative pool to the uncooperative labour pool, which can be thought of as an inadequate preservation disposition (ii). The reason why we consider only leakage from the cooperative to the uncooperative pool is that we think commonsense cooperativeness is a property which needs to be deliberately instilled in AI systems, while a disposition towards undermining oversight (i') may arise for a wide range of reasons (this is essentially an instance of instrumental convergence).

**Monitoring, evasion and observability:** To the extent human developers or cooperative AI developers are aware of uncooperative AI labour, they try to correct it. We define a stock of monitoring technology $M$ and a stock of opacity or evasion $E$, such that the fraction of uncooperative behaviour we observe is given by $O=\frac{M}{M + E}$. Human labour and cooperative AI labour produces $M$, while uncooperative AI labour produces $E$. Opacity is also passively produced as automation expands; expanding automation broadens the surface area to monitor and we consider the default situation with no monitoring effort to be one where a large fraction of uncooperative actions go unnoticed.

### Using automation growth as the "clock"

Total AI labour $A$ starts at a level significantly lower than $H_0$, and grows until it eclipses $H_0$. How quickly it does this is quite complicated; there are a number of economic models that aim to clarify how this could happen (the [AI Futures model](https://www.aifuturesmodel.com/); [Eth & Davidson](https://www.forethought.org/research/will-ai-r-and-d-automation-cause-a-software-intelligence-explosion) on the software intelligence explosion; the [Davidson–Epoch interactive takeoff model](https://epoch.ai/blog/interactive-model-of-takeoff-speeds)). We avoid modelling how quickly this happens; instead we look at how state variables in the model change with respect to $\sigma = \log(A/A_0)$. One unit of $\sigma$ is the amount of time required for automation capacity to rise by a factor of $e$.

See [Appendix](#appendix-derivation) for an explanation of what assumptions enable us to use this as a clock, and how we derive the resulting system of equations. We now explain the resulting dynamics in words; the full system of equations is collected in the [model reference](#model-reference) at the end of this section.

### The dynamics in words

Because we use automation advancement as the "clock", our model uses terms relativised to the total automation level, rather than absolute quantities: the uncooperative and cooperative shares of AI labour $q_u$ and $q_c=1-q_u$, the relative human share $q_h$, and monitoring and evasion stocks per unit of automation, $m$ and $e$, with observability $O=\frac{m}{m+e}$. The full variable list and equation block are collected in the [model reference](#model-reference) below; here we explain the dynamics one piece at a time.

**Evolution of relativised quantities:** The shares of cooperative or uncooperative behaviour, and the relativised stocks of evasion and monitoring technology all evolve according to the equations

\[
  \begin{aligned}
  x' &= \frac{F_x}{F}-x
  \end{aligned}
\]

Each share $x$ is pulled toward the fraction of *new* production its pool captures — $q_u$ toward $F_u/F$, and so on.

The one exception to this is human labour, which just thins out: $q_h' = -q_h$ says the human share decays as $q_h = q_{h,0} e^{-\sigma}$, because $H_0$ captures none of the available production pool, which by definition expands exponentially per unit of $\sigma$.

**Production capture:** The AI production allocation functions are:
\[
\begin{aligned}
F_c &= q_c(1-k_{cu}) + q_h(1-k_{hu}) + (1-\delta)\,\ell O\,L\\[4pt]
F_u &= (1-\ell O)\left(k_{cu}\,q_c + k_{hu}\,q_h\right) + (k_{uu} - \ell O)\,q_u\\[4pt]
L &:= k_{cu}\,q_c + k_{hu}\,q_h + q_u
\end{aligned}
\]
Reading $F_c$: the cooperative pool $q_c$ reproduces itself, less a fraction $k_{cu}$ that *leaks* into serving uncooperative ends; human labour $q_h$ contributes likewise, less its own leak $k_{hu}$; the third term is the redirected share of the full intercepted flow $L$ — suppressed uncooperative labour plus intercepted leakage — explained below. Leakage could happen for many reasons: systems that usually behave well could be used in unexpected or unwanted ways, or systems may have generalised undesirably from training and pro-actively behave badly in some situations. Reading $F_u$: it picks up the leaked cooperative effort $k_{cu}\,q_c$ and human effort $k_{hu}\,q_h$ — but only the *unobserved* fraction $(1-\ell O)$ of that leak actually enters the pool — plus its own reproduction at advantage $k_{uu}$, suppressed at the same observability-gated rate, hence $(k_{uu} - \ell O)\,q_u$. The idea behind gating the inflow as well as the stock is that observed misbehaviour begets workarounds applied to the whole production pipeline — patched training data, filters, revised reward signals, usage restrictions — and pipeline-level workarounds intercept misbehaviour that is *entering* the pool just as they suppress misbehaviour already established in it. Suppression therefore acts on the full flow $L$: the leakage inflow plus self-reproduction. Our model could be extended by allowing $\ell$ to depend on the other system parameters, but for simplicity we have avoided this.

Suppression removes uncooperative labour from the uncooperative pool, but "removed" can mean two quite different things. Some mechanisms *destroy* the labour outright: an output filter that blocks a flagged action discards the work entirely — nothing is produced in its place. Other mechanisms *redirect* it: retraining a model against an observed failure costs a small amount of compute, after which the corrected model goes back to work and the labour that would have been uncooperative is produced as cooperative labour instead. We capture this with a destruction fraction $\delta$: of the suppressed flow $\ell O\,L$, fraction $\delta$ is destroyed and the remaining $(1-\delta)$ is redirected into cooperative production. The uncooperative pool loses the full intercepted flow either way; $\delta$ controls how much of that flow reappears in $F_c$ and, consequently, how much suppression costs in aggregate output.

Roughly, today's suppression mechanisms sit on a spectrum:

| Mechanism | What happens to the caught labour | $\delta$ contribution |
|---|---|---|
| Output filtering / classifier rejection | the flagged output is discarded; the labour is simply lost | ≈ 1 (pure destruction) |
| Retraining / finetuning against the failure | a small fraction of compute is spent correcting the model, which is then redeployed and fails less; the labour is redirected | ≈ 0 (cheap, largely redirective) |
| Control-style protocols (resampling, output editing, restricted affordances) | part of the work is salvaged at an overhead in monitoring and redone work | ≈ 0.5 (in between) |

Note an idealization: redirected labour converts into cooperative labour one-for-one, with no lag and no quality discount, and the redirected flow is not itself subject to leakage in the same generation. If corrected labour is worth less than a full unit, the effective cost of suppression is higher than $\delta$ alone suggests. A second idealization sits inside the gating itself: intercepting the leakage *inflow* is taken to be exactly as effective per unit of $\ell O$ as removing established uncooperative labour from the *stock*. A separate prevention efficacy $\ell_k = \beta\ell$ on the inflow would be a natural extension; we fix $\beta = 1$ throughout.

This is why $q_h$ appears additively in the production-capture equations. Human labour is being measured in the same effective-labour units as $A_c$ and $A_u$. If humans remain an essential bottleneck for some kinds of AI development, this would overstate how quickly the human contribution fades; modelling that would require adding task bottlenecks or complementarities explicitly.

$F$ becomes:
\[
F = q_c + q_h + k_{uu}\,q_u - \delta\,\ell O\,L.
\]
The un-intercepted leakage and the redirected $(1-\delta)$ share of the intercepted flow cancel in the sum — transfers between pools, not losses. Only the *destroyed* fraction $\delta$ of the intercepted flow slows overall progress. Note that because suppression now intercepts the leakage inflow too, leakage is no longer a pure transfer: the destroyed share of intercepted leakage, $\delta\,\ell O\,(k_{cu}q_c + k_{hu}q_h)$, comes out of aggregate output. At our central parameters this correction is small (under 1% of $F$), but it means aggregate progress now depends on leakage directly, not only through where labour ends up.

**Asymmetries in production capture:** We understand cooperative AI labour $A_c$ and uncooperative labour $A_u$ to be similar in many respects - they're both pools of labour concerned with building the next generation of AI - and this is reflected by the fact that our model treats each quite similar (see the equations for the general form of $\dot A_u$ and $\dot A_c$). However, there are two key asymmetries between them:

- *Leakage runs one way.* Cooperative and human effort can slip into the uncooperative pool, but nothing leaks back. The idea here is that errors can undermine cooperativeness but are unlikely to reinforce it. Alternatively, we could think of $A_c$ as a "dominant faction" while $A_u$ is a collection of diverse factions with a shared interest in undermining $A_c$; so many different reasons could lead to an AI system being in $A_u$ but there is not such diversity for membership of $A_c$.
- *Suppression acts only against the uncooperative pool* — on its self-reproduction and on the leakage flowing into it (and with $\delta<1$ the cooperative pool *receives* the redirected share of the intercepted flow); nothing suppresses $F_c$ — Where $A_c$ is the dominant faction, there is relatively little to be gained by $A_u$ by directly undermining $A_c$ (because it is much less powerful). If $A_u$ grows to rival $A_c$, it may make aggressive moves to undermine $A_c$. As a policy objective, we would like to keep $A_u$ sufficiently small that such moves are very unlikely, and for simplicity we don't include such dynamics in this model.


**Monitoring and evaluation capture:** The equations for monitoring and evaluation capture are:

\[
\begin{aligned}
F_M &= c_M\left(q_c+\frac{q_h}{a_{\mathrm{AI/H}}}\right) \\[4pt]
F_E &= c_0 + a_{\mathrm{E/M}}\,c_M\,q_u \\[4pt]
O   &= \frac{m}{m + e}
\end{aligned}
\]

Monitoring grows with cooperative AI and human labour. $c_M$ is the overall monitoring-production rate relative to the production of automation, while $a_{\mathrm{AI/H}}$ says how much better AI is than humans at producing monitoring technology. If $a_{\mathrm{AI/H}}=10$, then one unit of human labour contributes one tenth as much monitoring as one unit of cooperative AI labour.

Evasion has two sources. The term $a_{\mathrm{E/M}}c_Mq_u$ is active evasion produced by uncooperative AI, with $a_{\mathrm{E/M}}$ measuring the evasion-vs-monitoring advantage per unit of AI labour. The term $c_0$ is passive production of observation difficulty: as automation grows, systems become larger, faster, more alien and more internally automated, and current monitoring methods become less adequate even without any deliberate evasion effort. An assistant that completes a small number of well-scoped tasks is relatively easy to evaluate; an autonomous research organisation is much more difficult. Observability is a soft contest between the two stocks: $O \to 1$ when monitoring is ahead, $O \to 0$ when evasion is.

### Model reference {#model-reference}

Collecting the variables:

 - $A=A_c+A_u$: total AI labour that goes into producing the next generation of AI
 - $A_u$: uncooperative AI labour
 - $A_c$: cooperative AI labour
 - $q_u=\frac{A_u}{A}$: the fraction of total AI labour that is uncooperative
 - $q_c=\frac{A_c}{A}$: the fraction of total AI labour that is cooperative
 - $q_h=\frac{H_0}{A}$: how much human labour matters relative to AI labour (if AI labour doubles while human labour stays fixed, $q_h$ will halve.)
 - $M$: Total stock of monitoring "technology"
 - $E$: Total stock of monitor-evasion "technology"
 - $O=\frac{M}{M+E}$: Rate of uncooperative behaviour observed
 - $m=\frac{M}{A}$: Monitoring per unit of automation
 - $e=\frac{E}{A}$: Evasion per unit of automation

Our system of equations is

\[
  \begin{aligned}
  q_c &= 1-q_u\\
  q_u'  &= \frac{F_u}{F}-q_u\\
  q_h'  &= -q_h\\
  m'    &= \frac{F_M}{F}-m\\
  e'    &= \frac{F_E}{F}-e
  \end{aligned}
\]

\[
\begin{aligned}
L   &:= k_{cu}q_c+k_{hu}q_h+q_u\\
F_c &= q_c(1-k_{cu})+q_h(1-k_{hu})+(1-\delta)\,\ell O\,L\\
F_u &= (1-\ell O)(k_{cu}q_c+k_{hu}q_h)+(k_{uu}-\ell O)q_u\\
F   &= F_c+F_u = q_c+q_h+k_{uu}q_u-\delta\,\ell O\,L
\end{aligned}
\]

\[
\begin{aligned}
F_M &= c_M\left(q_c+\frac{q_h}{a_{\mathrm{AI/H}}}\right)\\
F_E &= c_0+a_{\mathrm{E/M}}c_Mq_u\\
O   &= \frac{m}{m+e}
\end{aligned}
\]

### Model idealizations

A quick list of idealizations in this model that one might want to consider relaxing:
 - Constant reproduction and leakage rates (see [Appendix: neglect, subversion and more realistic models](#appendix-neglect-subversion) for one proposal)
 - Constant efficacy of suppression of uncooperative behaviour $\ell$. Constant $\ell$ also assumes the fold-fixes-into-training process is dynamically stabilised: fixes are incorporated at roughly the rate new problem surface appears, generation after generation. If accumulated fixes compounded (raising effective $\ell$) or new capabilities outran the fixing process (lowering it), $\ell$ would drift. (The same fold-fixes-into-training picture is what motivates suppression gating the leakage *inflow* as well as the stock: workarounds land in the production pipeline, not on individual systems.)
 - Constant destruction fraction $\delta$, with redirected labour converting one-for-one into cooperative labour — no lag, no quality discount, and no same-generation leakage of the redirected flow.
 - Suppression is perfectly targeted: $O$ captures only *recall* (the share of uncooperative labour caught), and correction never falls on cooperative labour by mistake. With imperfect precision, observability would gain a second role — better monitoring would also cut the collateral suppression of wrongly-flagged cooperative labour — which the model omits. This collateral cost could be high when $q_u$ is small: at a low base rate of misbehaviour, even a small false-positive rate sweeps up a great deal of cooperative labour.
 - Observation is per-action but suppression acts on flows and stocks: $O$ is calibrated as a per-action catch rate, while $\ell$ intercepts the leakage inflow and removes labour from the uncooperative pool as a whole. The model does not track which actions were caught, so suppression granularity (per action, per behaviour class, per model) is blurred; the class-level reading of $O$ noted below is one alternative. There is also a residual stock-vs-flow asymmetry: the self-suppression term $-\ell O\,q_u$ is proportional to the *stock* and can exceed the self-inflow when $\ell O > k_{uu}$, while interception of leakage can never remove more than the leakage flow itself.
 - The dilution terms in $m' = F_M/F - m$ and $e' = F_E/F - e$ assume monitoring and evasion stocks must be continually re-produced just to keep pace with the growing automation surface — a monitoring stock that is not actively maintained loses ground at exactly the rate automation grows. This is a substantive scaling assumption, not bookkeeping: if monitoring technology generalised freely over a growing surface, observability would behave very differently.
 - The pools $A_c$ and $A_u$ are *behaviours persistent across model generations*, not literal model artifacts: a deprecated model's uncooperative tendencies "survive" in the pool only insofar as they propagate into successors. The model does not distinguish a tendency carried by many short-lived models from one carried by a few long-lived ones.
 - A single uncooperative pool cannot represent abundant *non-reproducing* misbehaviour — and plausibly most of today's observed misbehaviour is exactly that: (i′)-style deception and proxy-gaming with no tendency to entrench itself in successors. The model has nowhere to put such behaviour (everything in $A_u$ reproduces at $k_{uu}$), so today's observed rates may really be calibrating observability and (i′)-leakage rather than the dangerous pool; Broad's defence is precisely the hypothesis that they proxy for it. The natural fix — a two-pool extension with a non-preserving misbehaviour *flow* alongside a preserving pool whose seeding is capability-gated — is the lead item on our future-work list (see the [more realistic models appendix](#appendix-neglect-subversion)). Relatedly, the σ-clock should be read as starting roughly where reproductively-relevant ((ii′)) misbehaviour becomes capability-*possible* — approximately now; a capability-gated self-reproduction ramp $k_{uu}(\sigma)$ would not break the σ-clock, since capability is itself σ-indexed, but we do not implement one.
 - Uncooperative behaviour doesn't actively try to undermine the cooperative pool (beyond trying to evade monitoring)
 - Fixed human labour $H_0$
 - There's no allowance to model a policy response to high rates of observed uncooperative behaviour like shifting resources from producing automation of any kind towards improving observability or suppression technology. Note that in general policy responses will operate at calendar time, and so modelling policy responses will likely require reintroducing a calendar time clock to the model and parametrizing the takeoff speed. Relatedly, because the model runs on the automation clock, a *uniform* slowdown of all AI activity is literally invisible to it — takeoff speed cancels out of the dynamics. A blanket slowdown *with targeted exemptions* for observability and suppression work is representable, but only as what it really is: a costly way of raising $c_M$ and $\ell$ per unit of automation growth.

A note on AI control. Control-style protocols ([Redwood Research's framing](https://arxiv.org/abs/2312.06942)) aim to extract useful work from possibly-uncooperative models while limiting the harm they can do — trusted monitoring, resampling, output editing, restricted affordances. In this model's terms they are suppression mechanisms of intermediate $\delta$ (part of the caught work is salvaged) that may also raise $\ell$ and the monitoring productivity behind $O$; we do not model their distinctive structure — the explicit budget of auditing labour, the trusted/untrusted split, or the adversarial evaluation that certifies a protocol — and our single per-action catch rate $O$ is a coarse stand-in for what control protocols actually buy. One alternative reading deserves a signpost: $O$ could be reinterpreted as *class-level discovery* — the probability that a class of uncooperative behaviour has been identified at all, with control measures then targeting the entire class rather than individual actions. Under that reading, suppression of a discovered class could be far stronger than per-action filtering suggests, but our per-action catch-rate estimation of $O(0)$ from monitoring testbeds would then be inappropriate, and the calibration would need to be redone against evidence about how quickly behaviour classes are discovered. We note the alternative without exploring it.

Relaxing any of these is straightforward — just ask your favourite assistant — but making good choices about how you relax them and understanding the resulting model is harder, which is the reason we don't do it here.

## The basin picture: where can the system settle?

The model has a transient initial phase, while human labour is still large and while monitoring and evasion stocks are adjusting. But its long-run behaviour is relatively simple. Since $q_h'=-q_h$, human labour eventually becomes negligible relative to AI labour. In the long run we can set $q_h\approx 0$ and ask where the remaining dynamics settle.

For any fixed value of $q_u$, the monitoring and evasion stocks have a natural equilibrium:

\[
m^*=\frac{F_M}{F},\qquad e^*=\frac{F_E}{F}.
\]

At this equilibrium, long-run observability is

\[
O^* =
\frac{F_M}{F_M+F_E}.
\]

When $q_h=0$, this becomes

\[
O^*(q_u) =
\frac{q_c}{q_c+a_{\mathrm{E/M}}q_u+c_0/c_M}.
\]

This says that long-run observability falls when the uncooperative share grows, when active evasion becomes more productive, or when passive opacity $c_0$ becomes large relative to monitoring productivity $c_M$.

The long-run fixed points are the points where the uncooperative pool captures exactly its current share of new production. They are useful because they tell us which regimes the system can settle into after the transient phase has passed.

The important qualitative question is whether there is a stable cooperative-side regime, a stable uncooperative-dominant regime, or both. When both exist, the model has a basin boundary: starting on one side leads back toward low $q_u$, while starting on the other leads toward uncooperative dominance. In that case initial conditions matter. When only one stable regime exists, initial conditions affect the path and timing, but not the eventual destination, at least within the model's validity envelope.

The [fixed-points appendix](#appendix-long-run-fixed-points) derives the fixed-point equation. In the simplified long-run system it reduces to a quadratic, so the basin picture can be checked directly rather than inferred from simulation alone.

The two most important thresholds are intuitive. First, can rare uncooperative labour be suppressed while observability is still high? That requires observability-gated suppression to be stronger than uncooperative self-reproduction plus the leakage that gets past it:

\[
\tag{C1}
\ell O^*(0)\left[1+(1-\delta)k_{cu}-k_{cu}(a_{\mathrm{E/M}}+c_0/c_M)\,O^*(0)\right]>k_{uu}+k_{cu}-1,
\qquad
O^*(0)=\frac{1}{1+c_0/c_M}.
\]

Call this **Condition 1**. It says that the uncooperative share shrinks while it is rare. The bracket is a small correction (about $0.97$ at our central parameters), so to a good approximation this is still "suppression $\ell O^{\ast}(0)$ beats self-reproduction plus leakage, $k_{uu}+k_{cu}-1$" — but note that under production-gated suppression the seeding itself is intercepted too: even at vanishing $q_u$, only the fraction $(1-\ell O^{\ast}(0))$ of the leak gets through. When suppression is pure destruction ($\delta=1$) Condition 1 is necessary for a cooperative-side regime, but not sufficient: sufficiently large leakage can still lead to an uncooperative dominant regime. For $\delta<1$ it is not even necessary — redirection can stop a takeover even when uncooperative labour grows while rare, though the resulting attractor then typically sits at a *high* uncooperative share. The appendix explains the fixed-point curve and gives the exact quadratic.

Second, can uncooperative labour sustain itself once it dominates? If yes, then there can be a stable bad basin. The all-uncooperative endpoint is stable when

\[
\tag{C2}
k_{uu}+k_{cu}-1>\frac{(1-\delta)\,\ell}{a_{\mathrm{E/M}}+c_0/c_M}.
\]

Call this **Condition 2**. At $\delta = 1$ it reduces to the simpler $k_{uu}+k_{cu}>1$: once nearly all labour is uncooperative, observability has collapsed, and dominance is self-sustaining whenever self-reproduction plus leakage beats the reproduction of a small reintroduced cooperative minority. For $\delta<1$ the bar is higher, because observability never falls *exactly* to zero — as $q_u\to1$ it falls like $O^{\ast} \approx q_c/(a_{\mathrm{E/M}}+c_0/c_M)$ — so redirected suppression keeps re-seeding the cooperative pool at a rate proportional to the remaining cooperative share, and uncooperative dominance must outgrow that re-seeding too. Production-gating does not move this condition at all: near the all-uncooperative endpoint the intercepted leakage is a product of two vanishing quantities (observability and the cooperative share), so it contributes only at second order.

A useful summary:

- If Condition 2 holds, the all-uncooperative endpoint is stable. The model may then be bistable — a cooperative-side basin (when one exists; the exact condition is a quadratic discriminant, see the appendix) coexisting with an uncooperative-dominant basin, with initial conditions determining where you end up — or, if the cooperative-side basin does not exist, uncooperative dominance is the only outcome.
- If Condition 2 fails, the all-uncooperative endpoint is unstable and the system has a single interior attractor. This is not automatically good news: depending on parameters, that attractor can sit anywhere from a few percent uncooperative to a large uncooperative majority being continuously caught, suppressed and recycled. What matters is the attractor's *location*, not just its existence.
- One new regime appears under production-gated suppression: if $\ell\,O^{\ast}(0) \ge 1$, suppression intercepts more than the entire seeding flow, and rare uncooperative labour is driven to *exactly zero* in finite time. Eradication becomes a possible destination. At our central $O^{\ast}(0)=0.5$ this needs $\ell \ge 2$ — outside the working range — so it is a combined-lever destination rather than a default-path possibility, but it matters below when we discuss extreme scenarios.

For the central structural choices used throughout this post ($k_{uu}=1$, $a_{\mathrm{E/M}}=1$), the existence condition for a cooperative-side fixed point has a closed form. It now has three branches:

\[
\ell\,O^* \ge \frac{4\,\delta\,k_{cu}(1-k_{cu})}{\left(1-(1-\delta)k_{cu}\right)^2}\quad(\delta\gtrsim\tfrac12,\ k_{cu}\ \text{small}),
\qquad
\ell\,O^* \ge \frac{k_{cu}}{1-\delta}\quad(\delta\lesssim\tfrac12),
\]

with a third branch, $\ell\,O^{\ast} \ge 2k_{cu}/(1+(1-\delta)k_{cu})$, taking over at large $k_{cu}$ (it binds in the high-leakage scenario of the [AI 2027 section](#ai-2027)). At $\delta=1$ and small $k_{cu}$ this is a four-to-one rule with a gating discount: suppression strength times long-run observability must beat cooperative-to-uncooperative leakage by a factor of $4(1-k_{cu})$ — almost exactly four when leakage is rare, with the discount coming from the seeding itself being intercepted. This inequality is the single most decision-relevant output of the model, and we will return to it when we plug in numbers. You can also explore it directly: the [interactive basin explorer](/basin-explorer/) plots the basin boundary over any pair of parameters and lets you check how the verdicts below move as you change the inputs.

## The default path

What does the model predict if we run it forward from today's conditions, with no change beyond business as usual? The answer depends on which named calibration you accept. Broad and Strict (defined in detail in the [calibration section below](#named-calibrations)) are two readings of the same body of evidence, differing only in how much of today's observed misbehaviour is treated as reproduction-relevant. Each reading fixes its leakage rate through a human-era calibration identity introduced there: Broad's default is $k_{cu}=k_{hu}=0.041$ (the trend-adjusted reading, which believes the observed series' measured decline as well as its level), Strict's is $0.0054$ (the steady-state reading of its own level). This section runs both forward in full — the human-dominated opening, the handoff to AI-led development, and the long drift that follows — and then asks two questions of the bad branches: would we get a warning, and what single change would rescue the outcome? Calendar landmarks throughout use a fixed $T_{\mathrm{auto}} = 0.5$ y/doubling (so one σ-unit is about 0.72 years) and are illustrative only; the σ landmarks are the model's own.

### The opening act: initial conditions are forgotten

While human labour still dominates ($q_h \gg 1$), the uncooperative share is pinned near a level set by the balance between human-era leakage and suppression: new production is mostly human, the fraction of it that leaks past pipeline-level workarounds seeds the pool, and observed misbehaviour is continuously removed. Because the same observed-misbehaviour evidence sets $q_{u,0}$ and (through the calibration identity) $k_{hu}$, each calibration *starts* at its own human-era anchor — Broad at 5%, Strict at 0.5% — and the first act is correspondingly quiet, though its shape now differs between them. On Broad, whose default leakage rate believes the measured *decline* of the observed series as well as its level, $q_u$ keeps falling at first — from 5% to a minimum near 4.3% at σ ≈ 1.3 — and is still below its starting level at handoff. On Strict, calibrated to hold its level steady, $q_u$ sits flat at 0.5% through the human era. On both, observability *improves*, climbing from 0.5 toward a peak near 0.66, as human and cooperative labour build monitoring against a still-small evasion base.

The quiet start is not an artifact of starting at equilibrium. In the sensitivity sweep behind this section, sweeping the initial uncooperative share from 0.1% to 99% changed the long-run outcome in *no* calibration: the human era relaxes $q_u$ back toward the $k_{hu}$-anchored level before AI labour dominates. A Broad run started at $q_{u,0} = 30\%$, for instance, is back to 12% by handoff and within three percentage points of the central path by σ = 4; one started at 0.1% converges from below on the same schedule. Initial conditions are overwritten; what the human era hands forward is its *leakage rate*, not its starting share. (One caveat from the sweep: the calibration grants even extreme starts the same inherited oversight stock, $O(0) = 0.5$; if a majority-uncooperative start should also mean degraded initial oversight, the extreme corner is optimistic — the central region is unaffected.)

### The handoff: parity is not an event

At $\sigma = \ln q_{h,0} \approx 1.6$ — a bit over a year at current doubling estimates — AI labour passes human labour. In the model, nothing happens. That is partly by construction (the dynamics are smooth), but it is also substantive: no structural rate in the calibrated model references the human/AI ratio, so there is no threshold to cross at parity. The only landmark near the handoff is incidental — observability peaks (0.66 at σ ≈ 1.5 on both calibrations) and begins its slow relaxation — and that timing comes from the monitoring calibration, not from parity itself. This is worth contrasting with scenarios that treat the handoff as the dangerous moment: AI 2027 times its discrete behavioural flip to roughly this point. In our model the handoff date carries no significance; what matters is the leakage rate the AI-led era inherits.

### After the handoff: two destinies

From here the two calibrations part company. The verdicts at each calibration's default leakage rate, with sensitivity rows for the destruction fraction $\delta$ (the calibration-variant rows — naive and steady-state readings of the same evidence — are tabulated in the [calibration section](#named-calibrations)):

| Calibration | basin threshold $\ell^{\ast}$ | margin $\ell/\ell^{\ast}$ | long-run attractor $q_u^{\ast}$ | takeover endpoint $q_u=1$ |
|---|---|---|---|---|
| **Broad** ($\delta=0.7$) | 0.22 | 1.8× | **0.21** | unstable |
| Broad, $\delta=1$ | 0.31 | 1.3× | 0.24 (saddle at 0.72) | stable (bistable) |
| Broad, $\delta=0.3$ | 0.12 | 3.4× | 0.18 | unstable |
| **Strict** ($\delta=0.7$) | 0.030 | 13× | **0.022** | unstable |
| Strict, $\delta=1$ | 0.043 | 9× | 0.022 (saddle at 0.97) | stable (bistable) |
| Strict, $\delta=0.3$ | 0.015 | 26× | 0.022 | unstable |

All verdicts are computed from the long-run quadratic and confirmed by full simulation of the four-variable system; the simulated long-run states match the analytic fixed points to three decimals.

{{<figure src="/images/blog/dynamical-models/default-path-true-vs-apparent.png" width="100%" align="center" caption="The default path under each calibration (identity-calibrated defaults: Broad k = 0.041 trend-adjusted, Strict k = 0.0054 steady-state; ℓ = 0.4, δ = 0.7). Top: the true uncooperative share q_u (solid) against the contemporaneously observed rate O·q_u (dashed); on Broad, q_u dips below its starting level before drifting up, and the gap between the curves widens as q_u grows. Bottom: observability O peaks near the handoff and then erodes — mildly on Strict, substantially on Broad, where the growing uncooperative share itself drags long-run observability down. Calendar axis is illustrative (T_auto = 0.5 y/doubling)." />}}

**On Broad, the dip gives way to a long drift.** As AI-era leakage takes over from the calibrated human-era decline, the true uncooperative share turns upward: it is back through 5% around σ ≈ 2.7 (about two years), crosses 10% at σ ≈ 6.6 (~5 years), 15% at σ ≈ 12 (~9 years), and 20% only around σ ≈ 34 (~24 years), approaching the attractor at $q_u^{\ast} \approx 0.21$ over decades. Observability erodes alongside — from its 0.66 peak to 0.53 at σ = 4, 0.48 at σ = 6, and 0.40 at the attractor — because long-run observability $O^{\ast}(q_u)$ falls as the cooperative share shrinks. The *measurable* signal, the observed rate $O\,q_u$, climbs the whole way (2.5% today → 3.5% at σ = 4 → 6.4% at σ = 12 → ~8% at the attractor), but it increasingly understates the truth: by the attractor the true share is about 2.5 times the observed rate.

**Broad verdict: marginal.** Under Broad rates the system never reaches clean cooperative dominance. With the central $\delta = 0.7$ there is no runaway takeover — redirected suppression makes the all-uncooperative endpoint unstable — but the system equilibrates with roughly **a fifth of AI labour uncooperative**, continuously caught, suppressed and recycled. Across the $\delta$ range the attractor moves only between $q_u^{\ast}\approx0.18$ and $0.24$, and the margin above the basin threshold is modest ($\ell$ could fall ~1.8×, or $k_{cu}$ rise ~1.8×, before the attractor disappears into takeover). The margin also flatters the situation: as $\ell$ falls toward the threshold the attractor climbs continuously — past 0.32 at $\ell = 0.30$ and 0.46 at $\ell = 0.25$, with the basin finally appearing at $q_u \approx 0.68$ — so "the basin still exists" can describe a very bad world. Broad is boundary-adjacent in $\delta$ too: at the default rates the takeover endpoint switches from unstable to stable at $\delta \approx 0.80$ (the condition $A_\delta > 0$ of the appendix; between 0.73 and 0.80 across the identity variants in the calibration section), so our central $\delta = 0.7$ sits just below the boundary at which Broad becomes bistable — and at the $\delta = 1$ end of the range Broad sits within roughly 30% of its own basin threshold, inside it on two identity variants and outside on the third. The honest summary is that the Broad calibration is **genuinely marginal** — boundary-adjacent everywhere, with modest favourable movements in $k_{cu}$, $\ell$ or $O^{\ast}$ buying large changes in outcome — not that the model knows which side of any boundary we are on.

**On Strict, the run settles.** The true share crosses 1% at σ ≈ 4.9 (about 3.5 years), 1.5% at σ ≈ 7.9, 2% at σ ≈ 14 (~10 years), and levels off at the endemic attractor $q_u^{\ast} \approx 2.2\%$. Observability relaxes from its 0.67 peak back to about 0.49 and stays there; the observed rate rises gently from 0.25% toward about 1.1% at the plateau. The same qualitative shape as Broad — endemic, managed, observed-rate rising — but an order of magnitude lower and far from every boundary.

**Strict verdict: cooperative dominant, robustly.** Under Strict rates the system settles at $q_u^{\ast}\approx2.2\%$ uncooperative labour regardless of $\delta$ anywhere in $[0, 1]$, with a 9–26× margin over the basin threshold; even the old, uncorrected $\ell = 0.2$ leaves the verdict unchanged ($q_u^{\ast}\approx5\%$). To flip Strict into takeover, $k_{cu}$ must rise ~13× or $\ell O^{\ast}$ fall by the same factor.

The gap between the two verdicts is the headline: **the decisive uncertainty is not any structural feature of the model but how much of today's observed misbehaviour is reproduction-relevant.** Broad-vs-Strict spans "marginal, near the basin edge, with a large managed uncooperative minority" to "comfortably cooperative-dominant" using the same model, the same suppression estimate, and the same observability assumptions.

### Is there a fire alarm on the bad path?

If the Broad world is the marginal one, the natural question is whether we would *notice* sliding off it. We can answer this from the runs rather than from intuition. Take a bad path that sits inside our own stated uncertainty: Broad rates with the old suppression reading $\ell = 0.2$ ($\delta = 0.7$, leakage held at the default $k = 0.041$ — the "lever moved in the world" convention used throughout this section), which has no cooperative-side attractor — the uncooperative share goes all the way to dominance. On that path, no observable does anything discontinuous; indeed gating makes it *slower and quieter* than the equivalent path in the ungated model, because the trajectory crawls through the ghost of the just-vanished basin. The true share crosses 20% at σ ≈ 10 (~7 years) and 50% only at σ ≈ 53 — nearly four decades at current doubling times. The observed rate $O\,q_u$ rises smoothly to a peak of about 12.6% near σ ≈ 52 and then slowly *declines* — observability collapses faster than the uncooperative share grows. The steepest movement in the observed rate anywhere on the path is about 0.7 percentage points per σ-unit (roughly one point per year at current doubling times). And for the first several years the bad path is not even *distinguishable* from the marginal-but-stable one: through σ = 8 its observed rate stays within 1.2–1.35× of the Broad-central path's — at most 1.9 percentage points apart, well inside the calibration uncertainty of the underlying measurements. Even its eventual observed-rate peak (12.6%) is not far above the Broad-central plateau (~8%). One labelling note: if instead of moving the lever we *re-read the evidence* at $\ell = 0.2$ — re-solving the calibration identity, which lowers the implied leakage to $k \approx 0.035$ — the failure is quieter still: no takeover at all, but a stable equilibrium with about 58% of AI labour uncooperative. Both readings are bad; neither announces itself.

A knife-edge pair at the basin boundary makes the point in its sharpest form: the worlds with $\ell = 0.222$ (takeover) and $\ell = 0.226$ (a basin exists) differ in their observed misbehaviour rate by at most 0.075 percentage points over the first sixty σ-units — four decades at current doubling times. And note what the boundary separates here: the "good" side of this particular pair is itself a world whose attractor holds about 63% of AI labour uncooperative. Near the boundary, contemporaneous observables cannot tell you which side you are on — and at this distance from the central calibration they cannot even tell you whether the slide has a floor. Far from the boundary they can: the AI 2027-style high-leakage run in the next section pushes the observed rate to ~24% within three σ-units, an unmissable signal. The model's answer to the fire-alarm question is therefore: **bad worlds announce themselves only when they are extremely bad; marginal-to-bad transitions produce no alarm and must be detected by construction** — which is what the lagged, unconfounded audit series of the [good-path section](#are-we-on-the-good-path) is for.

Two caveats keep this honest. First, gradualness is partly baked in: the model's uncooperative pool always pursues a "hide and grow" strategy, consistent with our choice to model only the regime where cooperative-plus-human labour is dominant. We think maintaining dominance by a significant margin (say, greater than 10:1) is the appropriate policy target precisely because approximate parity may create unstable dynamics — coordinated moves against the cooperative pool — that this model deliberately omits. A real takeover could have a sudden final act that the model cannot represent; what the model says is that the *runway* to that act is long and quiet. Second, the constant-rate assumptions (growth, suppression efficacy, productivities) and the weak observation-suppression feedback are what make the drift slow; faster feedbacks in either direction would steepen it.

### What single change would rescue the Broad path?

Since Broad is the marginal calibration, it is worth asking what one parameter change would move its default outcome from "a managed fifth of the workforce" to something comfortably good. For $\delta < 1$, *basin existence* is the wrong target — Broad's basin already exists, attractors for $\delta<1$ can sit at high uncooperative shares (Broad's own sits at 21%), and as noted above the attractor climbs toward 0.68 as the existence margin shrinks — so we ask for attractor *location* instead. As a provisional standard, call an outcome good if the attractor sits below $q_u^{\ast} = 0.1$, i.e. cooperative dominance of order 10:1; this threshold is a judgement call, not a model output, and readers should substitute their own.

{{<figure src="/images/blog/dynamical-models/default-path-rescue-levers.png" width="100%" align="center" caption="Attractor location under single-lever changes from the Broad default (dot). Red shading: no attractor — takeover. Dashed horizontal line: the provisional 'good outcome' threshold q_u* = 0.1. Under production-gated suppression three levers can reach it alone: leakage k_cu (1.8× reduction), suppression ℓ (1.6× increase), and — barely — observability (O* ≥ 0.80, with O* = 1 landing at q_u* ≈ 0.069); the mechanism mix δ bottoms out at 0.17. Note the attractor climbing toward ~0.68 as ℓ approaches the basin threshold: existence margins overstate safety." />}}

One lever at a time, holding everything else at the Broad default:

- **Leakage $k_{cu}$ (alignment training).** Reaching $q_u^{\ast} \le 0.1$ needs $k_{cu} \lesssim 0.022$ — about 1.8× below the Broad default 0.041, roughly a third of the way to the Strict reading on a log scale. Producing AI development labour that leaks less is the rescue lever that attacks the problem at its source; and this row again says that *finding out* the true reproduction-relevant rate is worth a great deal, since on the Strict reading we are already past the target.
- **Suppression $\ell$ (control and correction).** Needs $\ell \gtrsim 0.64$, about 1.6× the central 0.4 and comfortably inside our working range (0.2–1.0); the range-top $\ell = 1.0$ lands at $q_u^{\ast} \approx 0.045$, well past the target. Acting more strongly on *observed* misbehaviour — faster correction loops, deployment gates, control-style protocols — can do it alone. (Under the ungated model this lever needed ~2.7×, just beyond the working range; production-gating is what brings it inside.)
- **Long-run observability $O^{\ast}$ (interpretability and monitoring).** Can now do it alone — barely. Suppression and observability still enter the long-run dynamics only through the product $\ell O^{\ast}$ (at $a_{\mathrm{E/M}}=1$), so better monitoring remains exactly as good as proportionally stronger suppression, and $O^{\ast} \le 1$ still caps what monitoring can buy. But gating lowers the required product from about 0.54 to about 0.32, which puts the rescue inside the cap: $O^{\ast} \ge 0.80$ suffices, and perfect long-run observability ($O^{\ast} = 1$) brings the attractor to about 0.069. In the ungated model we reported that monitoring alone *cannot* rescue Broad; production-gating reverses that conclusion, though only at the optimistic end of the monitoring range. Less heroic rescues remain *combinations* along the same product line: for example $\ell \to 0.52$ together with $O^{\ast} \to 0.62$.
- **Destruction fraction $\delta$ (the suppression-mechanism mix).** Still cannot do it alone, in either direction: shifting suppression entirely toward redirection ($\delta = 0$, everything caught is retrained into cooperative production) lowers the attractor only to 0.17, while shifting it toward pure filtering ($\delta \to 1$) raises the attractor to about 0.24 and, beyond $\delta \approx 0.80$, makes the takeover endpoint stable as well. The mechanism mix matters — more redirection is better on every margin here — but it repositions the system within the marginal corridor rather than getting it out.

One further destination exists only under production-gating: if the product $\ell\,O^{\ast}(0)$ can be pushed to 1 or beyond (say $\ell = 1.3$ with $O^{\ast} \ge 0.77$), suppression intercepts more than the entire seeding flow and the uncooperative share is driven to exactly zero — eradication. No single lever reaches it from the Broad default; it is a combined-lever destination.

The summary shape: at Broad rates the system sits roughly 1.8× from takeover and 1.6–1.8× from comfort, and — unlike in the ungated model — every lever except the mechanism mix can in principle exit the corridor alone: leakage reduction and suppression comfortably within their ranges, monitoring only at the very top of its. This is the lever-level version of the earlier claim that we are in a regime with high returns to cooperativeness-focused interventions — and it still says the *largest* returns are to knowing and lowering $k_{cu}$.

### What the apparent path hides

One feature recurs in every run above without having been put in by hand: the measurable signal and the truth come apart as the uncooperative share grows. There is no parameter in this model that says "deception gets worse over time" — the divergence is produced by the production structure itself, since long-run observability $O^{\ast}(q_u)$ falls as the cooperative share that builds monitoring shrinks. On the Broad default path the observed rate understates the true share by a growing factor (about 1.9× at σ = 4, 2.5× at the attractor); on the takeover path the observed rate eventually peaks and *falls* even as the true share keeps climbing toward dominance. This matches the long-standing qualitative prediction of the deceptive-alignment literature — the signal understates the problem exactly when it matters most — and it is worth noting that the model reproduces it as an emergent consequence rather than an assumption. We flag it as a consistency check rather than a headline result. It is also a different *mechanism* than AI 2027's: there, misalignment hides because a discrete behavioural transition produces a model that actively conceals; here, nothing flips — the watching simply thins out as the watched grow. The constructive response to both is the same and is developed in the [good-path section](#are-we-on-the-good-path): measurement instruments whose power is not coupled to the contemporaneous cooperative share.

### What should we make of this?

On the model's authority:
 - This model has many shortcomings, some of which we've discussed
 - Predicting this outcome is a tough problem, there are no authoritative signals
 - This finding is not especially robust - reasonable updates to the model or parameter estimates could shift the conclusion significantly
 - While we iterated on the model substantially, we didn't revise the model or the parametrisation on the basis of whether the conclusions seemed reasonable or not; revisions typically followed from either a desire for simplification, or from more carefully considering the rationale

With these caveats in mind - the model's forecasts deserve modest weight in the scheme of things - the model's result is that we are in a regime with high returns to cooperativeness focussed interventions. On the Broad reading we sit close to the basin boundary, and because our uncertainty covers regimes with good and bad outcomes, shifting parameters like leakage $k_{hu}, k_{cu}$, intervention efficacy $\ell$, the destruction fraction $\delta$ or monitoring productivity $c_M$ in a favourable direction has a substantial impact on the probability of a good outcome.


## AI 2027 in this model: similar concerns, different risk models {#ai-2027}

A central payoff of making risk models mathematically precise is that seemingly similar concerns — "misaligned AI is a risk" — can turn out to mask quite different risk models, with different policy responses. This section works that through for one prominent example. This model is supposed to clarify discussion of AI misalignment risk, and if it is to do this, we must be able to use the model to illuminate existing discussions of this issue. We work this through for the [AI 2027](https://ai-2027.com/) scenario, one of the highest profile discussions of this risk. Its premise — AI automating AI R&D, with each generation training the next under the supervision of the last — is exactly the setup of this model.

The AI 2027 scenario comes with a detailed model of AI capability progress in the [AI futures model](https://www.aifuturesmodel.com/) and offers a qualitative account of alignment success or failure. Note that they say "misaligned" where we say "uncooperative". We summarize that account as follows:

 1. **Uncooperativeness is an inductive bias**: in the scenario description there is a discrete transition from a model "Agent-3" which is "not adversarially misaligned" to a model "Agent-4" which is "adversarially misaligned", and this happens because "The actions that it takes off-distribution aren't constrained by the reward; its OOD behavior is instead set by the prior."
 2. **Uncooperative labour satisfies both of our conditions**: (i'): Agent-4 sandbags on some alignment research. Specifically, it sandbags on the small fraction of alignment research that it thinks is actually on track to solve the problem or reveal Agent-4's scheming, (ii'): Tasked with building its successor, "It's supposed to be aligned to the Spec, but Agent-4 plans to make it aligned to Agent-4 instead."
 3. **"Adversarial" uncooperativeness emerges over about 2 $\sigma$-units**: The effective AI labour multiplier for Agent-4 over Agent-3 is about 7x (2 $e$-foldings), which takes place over 4-6 calendar months.
 4. **Misalignment represents a behavioural change not just a capability increase**: While Agent-3 satisfies property (i') (it is uncooperative), it does not satisfy (ii') (it does not have a propensity to building models that satisfy both (i') and (ii')), while Agent-4 satisfies both (see above). That is, Agent-3 is not similarly uncooperative and merely 7x less productive in its uncooperativeness.
 5. **Oversight passively degrades as AI labour rises**: Agent-3 supervises Agent-4's training but "Agent-4's neuralese 'language' becomes as alien and incomprehensible to Agent-3 as Agent-3's is to humans."
 6. **Uncooperativeness is partially observed**: "'defection probes' and similar techniques result in a primitive sort of mind-reading that sends up red flags when Agent-4 is thinking about spicy ideas like deceiving OpenBrain"
 7. **Slowdown buys safety mostly via interventions on $O$ plus knock on effects**: "They switch to an architecture that preserves the chain of thought, allowing them to catch misalignment as it emerges", and these AIs — "which are able to be monitored much more robustly — make breakthrough advances in AI alignment"

We model 1 as a high leakage rate $k_{cu}$. If the optimization landscape strongly converges to uncooperative AI systems, then the vast majority of well-intentioned work will end up producing uncooperative AI systems. We use high leakage from automated labour $k_{cu}$ instead of human labour $k_{hu}$ as an ad-hoc way to represent the fact that this is hypothesised to be an effect that becomes important at around the same time human labour is mostly handed off to AI systems. We somewhat arbitrarily pick $k_{cu}=0.9$ as a representative path. If it were truly a strongly convergent endpoint, we may expect that it is much closer to $1$, while if it is a weak tendency that can be avoided by employing certain tricks then it may be much lower, perhaps even around our working estimate of $0.05$. The uncertainty this surfaces is a feature - the language is compatible with a wide range of parameter estimates, and as such identifies an opportunity for clarification in the original work.

Under such high leakage rates, achieving cooperative dominance is very difficult. The following figure illustrates the modal trajectory taking our central estimates for all parameters except for the leakage rate. We see that observed misbehaviour rises before falling again, while the uncooperative share of AI labour rises relatively quickly to dominance.

{{<figure src="/images/blog/dynamical-models/ai2027-high-leakage-run.png" width="78%" align="center" caption="A representative high-leakage trajectory (k_cu = 0.9; other parameters at the Broad default calibration, k_hu = 0.041, ℓ = 0.4, δ = 0.7, O* = 0.5). Top: the true uncooperative share q_u (solid) climbs to dominance over a few σ-units, while the observed rate O·q_u (dashed) peaks early (~24%) and then declines. Bottom: observability gains ground at first, then collapses as the alien, high-q_u workforce outgrows monitoring. The measurable signal understates the problem exactly when it matters most." />}}

Under these leakage rates, control of uncooperative labour via observation and suppression must be very strong in order to succeed. At $k_{cu} = 0.9$ the binding branch of the gated threshold is the large-$k_{cu}$ one, $\ell^{\ast} O^{\ast} = 2k_{cu}/(1+(1-\delta)k_{cu})$: about $1.8$ at $\delta = 1$ and $1.4$ at our central $\delta = 0.7$. (The ungated model required roughly twice this — $\ell O^{\ast} \approx 3.6$ at $\delta = 1$ — because it credited suppression with nothing against the torrent of incoming leakage; under gating, intercepting the inflow does double work exactly where the inflow dominates.) Read as a per-doubling cost heuristic, $\ell O^{\ast} \approx 1.8$ means pre-existing uncooperative labour must be cut to about $2^{-1.8} \approx 1/3.5$ of its natural rate for every doubling of AI labour, and at $\delta = 1$ a comparable share of total production is being caught and discarded, so aggregate progress slows by roughly the same factor; at $\delta = 0.7$ only the destroyed fraction is lost and the heuristic cost is about $2^{-\delta\,\ell^{\ast} O^{\ast}} \approx 1/2$ per doubling. These are softer numbers than the ungated model's $1/16$th, but still a deep ongoing slowdown. (Treat them as cost *heuristics* read off the threshold formula, not model trajectories: at these values the model's aggregate production $F$ actually turns negative, i.e. the system exits the model's validity envelope rather than literally crawling forward at a reduced speed.) Gating also changes *what kind* of cooperative outcome is available here, in a way worth pausing on: both required products exceed the eradication boundary $\ell O^{\ast}(0) = 1$. So at this leakage there is no reachable "managed minority" equilibrium at all — by the time suppression is strong enough to matter, it is already past the over-suppression line, and the cooperative outcome it buys is *eradication*: the uncooperative share driven to exactly zero. That destination comes with a sharp caveat: its basin near $q_u = 0$ is narrow at high leakage, so it is only reachable if this suppression regime is in place *before* uncooperative labour grows large. And observability-based interventions of *any* kind still depend fundamentally on observing misbehaviour, which in this scenario is the vast majority of work done by AI building more AI — while the redirection arithmetic leans on the model's idealization that corrected labour converts to cooperative labour at full value.

{{<figure src="/images/blog/dynamical-models/ai2027-observability-cannot-save.png" width="100%" align="center" caption="At high leakage, observability alone cannot recover a cooperative outcome. Shaded regions classify whether any cooperative outcome exists — a conventional cooperative-side basin or, beyond the dotted eradication line ℓ·O*(0) = 1, over-suppression driving q_u → 0; the dashed curve is the gated three-branch basin boundary of the appendix. (a) Holding suppression at the central ℓ = 0.4 and δ = 0.7, raising best-case observability O* from 0.5 toward perfect roughly doubles tolerable leakage (≈ 0.074 → ≈ 0.15) — still several-fold short of this scenario's k_cu = 0.9. (b) At near-perfect observability (O* = 0.99), a conventional basin at k_cu = 0.9 would need ℓ ≈ 1.4, but the eradication line arrives first (ℓ ≈ 1.0): the only cooperative outcome at AI 2027 leakage is over-suppression, at roughly 2.5× the central suppression estimate and with a narrow basin near q_u = 0." />}}

Now the AI 2027 model suggests that heavy suppression need not go on indefinitely; ultimately the (heavily suppressed) AI workers make a breakthrough in alignment, which presumably lowers leakage to a level such that monitoring and suppression can be greatly relaxed. This is a feature of the AI 2027 model absent from ours: the leakage rate is dynamic, and (on our understanding) first rises substantially at around the point of "handoff" from human to AI labour, then potentially falls substantially once an alignment breakthrough is made. On this view there are two discrete "problems" - first, building uncooperative AI-reproducing AI workers, and second building cooperative AI-reproducing AI workers. The first problem is somewhat easier than the second, and so the challenge is to effectively use the uncooperative labour to solve the second problem.

This vision suggests quite a different policy stance to our mainline scenario. Our mainline scenario suggests a good deal of safety can be bought by making marginal improvements to $O$, $\ell$ or $k_{cu}$ because we are near the edge of the basin boundary; the AI 2027 scenario instead seems to indicate that large shifts are necessary — the required suppression-observability product (at least 1, to reach the over-suppression regime, and 1.4–1.8 for the conventional thresholds) is five to nine times our central estimate of where that product stands today ($\approx 0.2$). Furthermore, because the problem is discrete, bounded in time but very costly to mitigate, it suggests a "lock down late and sprint for a solution" approach. To be clear, this is our interpretation of the article and perhaps the authors do not endorse our "high but solvable leakage" reading (I plan to ask them). This illustrates the value of these modelling exercises: making claims mathematically precise can show where seemingly similar concerns ("misaligned AI is a risk") may mask quite different risk models with different policy responses *or* show where risk models are being miscommunicated.

## Estimating parameters

The pool we labeled $A$ — AI that does AI development — now exists (see the introduction). Despite this reality, it's still not straightforward to estimate parameters for this model, not the least because the strict definitions of $A_c$ and $A_u$ include things we can't directly measure - does the AI's output tend to cause $A_c$ to grow, $A_u$, or neither? However, we can use inexact proxies to try to calibrate many of the parameters. This section summarises the load-bearing judgements; the underlying evidence is reviewed source-by-source in the [calibration evidence appendix](#appendix-calibration-evidence).

**Observed uncooperative behaviour.** Three relevant sources, reviewed source-by-source in the [appendix](#appendix-calibration-evidence): OpenAI's chain-of-thought monitors over production traffic report *deception* rates falling across model generations from roughly **4.8%** (o3) to **1.6%** (GPT-5.2 thinking); DeepMind's Gram auditing framework finds Gemini models sabotaging in about **2–3%** of simulated agentic coding and research trajectories, falling to fractions of a percent when scenarios are naturalised; and Anthropic's Petri behavioural audits give broader — but harder to interpret — misalignment scores of roughly **0.1–0.2** on the judge-score scale, improving over time for most labs.

This data suggests an interval of 1-20%, representing the extreme ends of these pieces of work. The lower end is anchored by more naturalistic deception, coding and sabotage measurements; the upper end is motivated by broad behavioural-audit scores like Petri. Bear in mind, this is our estimate for $O(0) q_{u,0}$, not $q_{u,0}$.

A further consideration is that the proxies above measure uncooperativeness in general traffic ("property (i')" in the terminology discussed earlier), whereas $k_{hu}$ and the reproduction-relevant $q_{u,0}$ concern development labour that also pushes successors toward being less cooperative ("property (ii')" in our terminology). Such directed work is very rarely observed — far rarer than behavioural uncooperativeness. In contrast, AI systems *do* regularly help with making future systems cooperative. This weighs towards lower estimates, though it's hard to say how much. It is plausible that uncooperative behaviour in the sense discussed here is much rarer - it may be appropriate to consider low-end rates closer to 0.1% than 1% (this is also supported by the DeepMind observation that naturalizing their scenarios dropped sabotage rates to fractions of a percent). A scope note follows from the same distinction: the historical *decline* of these broad series is a fact about (i′) behaviour, accumulated over an era in which (ii′) misbehaviour was largely capability-impossible. The model's clock effectively starts where reproduction-relevant misbehaviour becomes capability-possible — roughly now — which is part of why the Strict calibration below declines to inherit that trend.

**Observability $O$.** We don't have a direct estimate of this. What we have instead are two weaker kinds of evidence (detail in the [appendix](#appendix-calibration-evidence)): *monitoring testbeds*, where catch rates against agents hiding side tasks range from about **30%** to about **90%** depending on what the monitor can see (reasoning-trace access, extra compute, follow-up questioning); and the *mature-domain equilibrium* in cybersecurity, where organisations' self-detection rate has held near **43%** for a decade while detection lag collapsed from ~205 days to ~11. On this evidence, we treat $O(0)$ as having an 80% credible range of 30-90%, with a central estimate around 50%. We don't take parameter estimates directly from the cybersecurity baseline, other than noting that it weakly supports a trend of rising observability in $A_c$-dominant regimes.

To calibrate the observability sector we therefore fix $O(0)$ from the present-day evidence above and put a prior on the passive-opacity ratio $c_0/c_M$, taking $c_0=c_M$ as the central choice so that long-run observability settles at $O^{\ast}=1/2$ while $A_u$ remains low (it falls as $q_u$ grows). The remaining production scale is pegged to the automation scale, which affects only how quickly $O$ relaxes from $O(0)$ toward $O^{\ast}$. An alternative route instead calibrates $c_M$ and $c_0$ from two rate judgements, $T_{M,1/2}$ and $T_{E,2}$ (see [Appendix](#appendix-observability-calibration)) — anchorable to cybersecurity dwell-time and capability/replication slopes. Both routes are supported, for different jobs: use the $O^{\ast}$-prior route (this post's default) when you want transparent scenario sweeps — it makes long-run observability an explicit dial, which is how the verdict tables below are built — and the rate-judgement route when you are bringing *trend* data about monitoring or evasion, such as dwell-time series, and want the model to infer the implied long-run balance rather than assume it.

**Suppression $\ell$.** Because $\ell$ acts only on the *observed* fraction, it is by construction efficacy against *known* misbehaviour. The most relevant estimate is therefore not "can one targeted intervention remove one known failure?", but "how fast do broad, observed misalignment scores fall as labs iterate on models?" Petri gives a rough handle on this: treating the improving labs' score declines as exponential (per-lab trend readings in the [appendix](#appendix-calibration-evidence)) gives calendar half-lives of roughly **42 months** for OpenAI, **32 months** for Claude, and **27 months** for Gemini, or a central broad-misalignment half-life around **30-35 months**.

Converting this into the model's automation clock takes one care. In the model, suppression removes uncooperative labour at rate $\ell O$ per $\sigma$-unit — the term $(k_{uu}-\ell O)\,q_u$ — so a measured decay of misbehaviour identifies the *product* $\ell O$, not $\ell$ itself:
\[
\ell\, O(0) \approx \frac{T_{\mathrm{auto}}}{T_{\mathrm{half}}},
\qquad
\ell \approx \frac{T_{\mathrm{auto}}}{T_{\mathrm{half}}\,O(0)}.
\]
If effective AI-development labour doubles every **3-7 months**, the Petri trend implies $\ell O \approx 0.1$-$0.25$, with a central value around **0.2**. Dividing by our central observability estimate $O(0)\approx0.5$ gives a central suppression estimate of $\ell\approx\mathbf{0.4}$; carrying the $O(0)$ range of 0.3-0.9 through as well spreads the nominal range to roughly **0.1-0.9**.

Two identification caveats apply, pulling in opposite directions. First, the Petri decline is a *net* trend: misbehaviour was also flowing *in* over the window (the leakage terms), so the gross suppression rate must exceed the net decline — this biases the estimate **down**. Under the production-gated model this caveat no longer has to be hand-waved: the human-era calibration identity below solves the level, trend and suppression readings *jointly*, with the inflow itself gated by the same $\ell O$. Second, the decline may not be suppression at all: it could partly reflect a falling inflow — labs simply producing better-behaved models per unit of development effort, i.e. falling $k_{hu}$ — rather than iterative removal of observed failures. We cannot separate these with this data, and note that the same observed-misbehaviour evidence is now doing **triple** duty in this calibration: its level informs $q_{u,0}$, its trend feeds the identity below, and its decay rate sets this estimate of $\ell O$. Attributing the entire trend to suppression therefore biases $\ell$ **up**.

Beyond identification, Petri is also not a clean prospective measurement of "failure observed, then failure suppressed." It is a fixed-harness, backward-looking audit: it can surface failures in older models that may not have been known at release time, and newer models may introduce novel failure modes even as older ones are trained away. These caveats mostly point toward Petri underestimating $\ell$ as defined here, since $\ell$ is conditional on the failure actually being observed. On balance we retain an upward-shifted working range: we use $\ell\approx0.4$ as the central broad-suppression estimate and $\ell\sim\mathbf{0.2}$-$\mathbf{1.0}$ as a working range. (Note $\ell$ is a continuous rate per $\sigma$-unit, not a fraction, so values near or above 1 are meaningful.)

**Destruction fraction $\delta$.** This is the weakest-anchored parameter in the model: we know of no measurement of it, and the estimate below is a constructed heuristic rather than a reading of evidence. The heuristic: $\delta$ is approximately the destruction-weighted share of the suppressed flow handled by each mechanism class — filtering-type mechanisms count near 1, retraining-type near 0, control-style measures near ½. Real-time output filtering and classifier-based blocking are heavily used on frontier deployments and their use appears to be increasing; filtering also operates on every action, while retraining bites only at model-generation cadence, so by *volume of suppressed flow* filtering plausibly handles the majority. If filtering handles something like 60–80% of the suppressed flow, control-style measures 10–20% (at half weight), and retraining the remainder, $\delta$ lands around 0.65–0.85. We take $\delta = 0.7$ as the central value with a working range of $0.3$–$1.0$: the bottom end describes a world where suppression is dominated by generational retraining, the top end is pure destruction — which is also the assumption the model makes implicitly if $\delta$ is omitted. Readers who weight the mechanisms differently should move this slider first; among all the parameters in this model, $\delta$ is the one where we would most expect an informed reader to substitute their own number. Note that the Petri-based estimate of $\ell$ above is unaffected by $\delta$: the score decline measures the full flow $\ell O$ leaving the uncooperative pool, while $\delta$ only governs where that flow goes. (This survives production-gating too — the human-era self-suppression term the half-life reads is $-\ell O\,q_u$ regardless of $\delta$ and of whether the inflow is also being intercepted.)

**Initial uncooperative share and leakage.** Combining the observed-uncooperative and observability ranges gives a rough estimate for the true initial uncooperative share:
\[
q_{u,0}\approx \frac{q_{u,0}^{\mathrm{obs}}}{O(0)},
\qquad
q_{u,0}^{\mathrm{obs}}:=O(0)q_{u,0}.
\]
Using $q_{u,0}^{\mathrm{obs}}\sim1$-$20\%$ and $O(0)\sim0.3$-$0.9$ gives a nominal range of **1-67%**, with the high end driven by taking the Petri-like upper tail of deliberate elicitation, together with low observability. We take a central estimate **5%**: a 2.5% rate of uncooperative behaviour (averaging Gemini sabotage and ChatGPT production traffic deception rates), and we think this is more likely to be an overestimate than an underestimate.

We use the same evidence to calibrate $k_{hu}$, the rate at which human AI-development labour leaks into uncooperative AI labour. The justification: at the present margin, human labour still dominates the production of new AI systems, so the observed stock of uncooperative AI labour is mostly informative about leakage from human-led development rather than leakage from cooperative AI labour. But the level is not the leakage rate — suppression may already be holding $q_u$ below it, and a falling series suggests the two are not even in balance. Under the gated model this can be made exact rather than hand-waved.

**From level to leakage: a human-era identity.** During the human era the model itself links the leakage rate to the measured level *and trend* of the uncooperative share. Setting $k_{cu}=k_{hu}=k$ (the double-duty reading), with measured level $q_0$, measured trend $g = q_u'/q_u$ per σ-unit, present observability $O_0$, and human-to-AI labour ratio $q_{h,0}$, requiring the model to reproduce that level and trend at σ = 0 gives

\[
k = \frac{q_0(1+g)\,F_0-(k_{uu}-\ell O_0)\,q_0}{(1-\ell O_0)\,(1-q_0+q_{h,0})},
\]

where $F_0$ is the initial total production. (Convention: we evaluate $F_0$ with the self-term suppression but drop the gating correction, which keeps the identity explicit in $k$; the fully self-consistent version moves $k$ by under 1%.) Three readings of the same evidence then differ only in what they feed this identity:

| reading | what it assumes | Broad ($q_0=0.05$) | Strict ($q_0=0.005$) |
|---|---|---|---|
| naive ($k:=q_0$) | the level *is* the leakage rate; suppression and trend ignored | 0.050 | 0.005 |
| steady-state ($g=0$) | today's level is an equilibrium of leakage against suppression | 0.055 | **0.0054** |
| trend-adjusted ($g=-0.22/\sigma$) | the measured decline of the series is a real decline in $q_u$ | **0.041** | — |

The spread is itself informative: correcting for the suppression outflow pushes $k$ *above* the naive reading (the observed level is being held down by suppression), while believing the measured decline pulls it *below* (part of today's level reflects suppression already beating seeding). The trend value $g\approx-0.22$ per σ-unit is read from the falling production-traffic deception series at $T_{\mathrm{auto}}\approx6$ months. **Defaults:** Broad uses the trend-adjusted $k=0.041$ — its identifying series is demonstrably falling, and using that series' level while ignoring its trend would be incoherent. Strict uses the steady-state $k=0.0054$ at its own level: the falling series measures broad (i′) misbehaviour, not the (ii′) pool Strict is about, so its trend is not licensed for Strict.

**Two readings of the falling series.** The same downward trend that licenses Broad's adjustment is, on its own, ambiguous between two very different worlds: (a) *benign* — $q_u$ is genuinely falling at about 0.22 per σ-unit, which production-gated suppression makes mechanically natural (pipeline workarounds intercept leakage before it compounds); or (b) *alarming* — observability is falling and the truth is flat or rising. Reading (b) is quantitatively demanding but not absurd: over the ~2.5-year window of the deception series (≈3.5 σ-units), a flat true $q_u$ requires $O$ to have fallen from 0.5 to about 0.23; a true $q_u$ *rising* at the mirror rate (+0.22/σ) requires $O$ to have fallen to about 0.11, declining at twice the rate of the observed series itself. Nothing inside the series distinguishes (a) from (b); separating them needs an independent instrument for the *trend in O* — monitorability testbeds tracked over time, or the lagged development-log audits of the [good-path section](#are-we-on-the-good-path). This ambiguity is also why that section states the model's headline observable prediction as a dichotomy rather than a forecast.

For $k_{cu}$, we use $k_{hu}$ as a starting analogy rather than an independent estimate: once cooperative AI labour dominates AI development, leakage through AI-generated research, code, data and evaluations may be similar in kind, but the empirical anchor is currently weaker.

**Self-reproduction $k_{uu}$.** We take $k_{uu}=1$ as a starting point — uncooperative labour reproducing itself at par. There are reasons to think uncooperative labour might have advantages here - it could operate in a manner parasitic to cooperative labour, but it may also face more challenges due to the need to operate covertly and will receive less assistance from human labour. Independent evaluators agree current frontier models cannot yet replicate autonomously in the wild ([UK AISI RepliBench, 2025](https://arxiv.org/abs/2504.18565)). That is a negative result rather than an absence of data — it tells us the regime the model cares about has not yet arrived — but it leaves us without a direct empirical anchor for $k_{uu}$ once it does.

**Timescales.** The automation doubling time $T_{auto}$ is the key conversion factor, because it turns calendar-time improvements into rates per unit of automation growth. We bracket it with two imperfect proxies (detail in the [appendix](#appendix-calibration-evidence)): an aggregate-value proxy from frontier-lab revenue growth (roughly **3.2×/year**, implying a doubling time of about **7.2 months**) and a technical-capacity proxy from compute stock multiplied by algorithmic efficiency (roughly **10-15×/year**, implying a **3-3.6-month** doubling). We therefore use $T_{auto}\sim3$–$7$ months: the upper end from OpenAI-style deployed value growth, the lower end from technical capacity growth.

**Summary.** Collecting the above:

| Parameter | Meaning | Anchor / range | Source | Caveat |
|---|---|---|---|---|
| $O q_{u,0}$ (observed) | observed uncooperative share |  0.1–20% | OpenAI deception monitors (~2–5%); DeepMind Gram on agentic coding/research agents (~2–3%); Anthropic Petri behavioural audits (~0.1–0.2 score scale, not a direct rate) | monitor recall ~84% (floor); Petri is ordinal/seed-dependent; Gram falls sharply with realism |
| $q_{u,0}$ | true initial uncooperative share | nominal $\sim 0.1$–$67\%$; central $\sim5\%$ | inferred from observed $Oq_{u,0}$ divided by $O(0)$ | upper tail combines Petri-like scores with low observability |
| $O$ | observability | swept $[0.3,0.9]$ | AI monitorability studies (~0.3–0.95 under artificial protocols) | right instrument absent; cyber is adjacent and mostly latency evidence |
| $\ell$ | suppression of observed misbehaviour | central $\approx0.4$; working range $\sim0.2$–$1.0$ | Petri/Rinehart trend-line half-lives (~27–42mo) identify $\ell O$; $T_{auto}\sim3$–$7$mo; divided by $O(0)\sim0.3$–$0.9$ | net-of-inflow and conditional-on-observation caveats bias down; falling-$k_{hu}$ confound biases up; same series does triple duty (level, trend, rate) |
| $g$ | trend of observed misbehaviour | $\approx-0.22$/σ ($T_{auto}\approx6$ mo) | falling production-traffic deception series | ambiguous between falling $q_u$ and falling $O$ (see text); licensed for Broad only |
| $\delta$ | destroyed share of suppressed flow | central $0.7$; range $0.3$–$1.0$ | filtering-fraction heuristic: filtering ≈ 1, retraining ≈ 0, control ≈ ½, weighted by suppressed-flow volume | no direct measurement exists; weakest-anchored parameter in the model |
| $k_{hu}$ | human-labour leakage into uncooperative pool | identity-calibrated: Broad **0.041** (trend-adjusted), Strict **0.0054** (steady-state); variants 0.041–0.055 (Broad) | human-era calibration identity on level + trend | identity assumes $k_{cu}=k_{hu}$, central $\ell O_0$; present AI development still mostly human-led |
| $k_{cu}$ | cooperative-AI leakage into uncooperative pool | set equal to $k_{hu}$, sweep wide | analogy from human-led leakage; poisoning/backdoor evidence supports $>0$ | weaker empirical anchor; future AI-led workflows may differ |
| $k_{uu}$ | uncooperative self-reproduction | $=1$ (starting point) | RepliBench; capability slope | not yet autonomous; upward pressure |
| $a_{\mathrm{E/M}}$ | evasion vs monitoring advantage | order 1 starting point | M-Trends dwell-time trend | indirect; mixes internal detection, external notification, adversary strategy, and incident mix |
| $c_0/c_M$ | passive opacity | order 1 sanity check (not $\approx 0$) | if cyber internal-detection share were read as $O^{\ast}(0)$, $c_0/c_M\approx1/O^{\ast}-1$ | strong-assumption reference-class inference only |
| $a_{\mathrm{AI/H}}$ | AI vs human at monitoring | $=1\ \pm$ wide | METR horizons; RE-Bench | defined at parity; both-way plausibility |
| $T_{auto}$ | automation doubling time | $\sim3$–$7$ months | OpenAI revenue growth; Epoch compute stock/training compute × algorithmic efficiency | revenue is an external-value proxy; technical capacity may overstate effective AI-R&D labour |

### Named calibrations: Broad and Strict {#named-calibrations}

Rather than a single central estimate, we present two named calibrations. They share every structural parameter and differ only in how the observed-misbehaviour evidence is read.

A point to make up front: that evidence is doing **triple duty**. The observed rates of deceptive, sabotaging or otherwise uncooperative behaviour supply our estimate of the initial uncooperative share $q_{u,0}$, their measured *trend* feeds the calibration identity, and their decay rate sets the suppression estimate $\ell O$ — so the same series stands behind $q_{u,0}$, $k_{hu}$ (and, by analogy, $k_{cu}$) and $\ell$. This is not circular so much as redundant: while human labour dominates, $q_u$ relaxes toward a level set by the leakage-suppression balance, so level and leakage converge during the human era and present-day measurements cannot separate them — the identity makes the link explicit instead of pretending the quantities are independent. The redundancy also turns out to be harmless for the conclusions: the long-run outcome is essentially insensitive to $q_{u,0}$ (sweeping it from 0.1% to 99% changes nothing in any calibration), so only the leakage-rate reading of the evidence does any work. But it does mean that "Broad vs Strict" is a *single* judgement — how much of observed misbehaviour is reproduction-relevant — applied consistently to $q_{u,0}$, $k_{hu}$ and $k_{cu}$ at once, not three independent estimates.

- **Broad** takes the behavioural evidence at face value: OpenAI's production-traffic deception monitors (1.6–4.8%), DeepMind's Gram sabotage audits (~2–3%) and the Petri-style audit scores are treated as direct measurements of uncooperative labour in the model's sense. Central observed rate ~2.5%, divided by $O(0)=0.5$, gives $q_{u,0}=0.05$; the calibration identity, fed the same series' measured decline, gives $k_{hu}=k_{cu}=0.041$. Broad's virtue is that it is built on an *actual measured number* — arguably a half-decent proxy for the quantity the model needs.
- **Strict** counts only labour with property (ii′) — work that pushes *successor systems* toward uncooperativeness. Directed work of this kind is very rarely observed: it is far rarer than behavioural misbehaviour, Gram's sabotage rates fall to fractions of a percent once scenarios are naturalised, and AI systems do routinely help make future systems *more* cooperative. Strict therefore takes the reproduction-relevant level roughly 10× lower, $q_{u,0}=0.005$, and the steady-state identity gives $k_{hu}=k_{cu}=0.0054$ (the broad series' downward trend measures (i′) behaviour, so Strict does not inherit it). Strict's virtue is that it estimates the *actual quantity* the model needs — but with no direct data behind the 10× discount; it is explicit guesswork.

The two are individually defensible readings, and we present them symmetrically: one is a measured proxy, the other an unmeasured estimate of the right quantity, and we see no principled way to crown either as the headline.

Both calibrations use the corrected suppression estimate $\ell = 0.4$ (the Petri trend identifies $\ell\,O$, not $\ell$; see the suppression discussion above) and a suppression destruction fraction $\delta = 0.7$ (central; range 0.3–1.0): of the uncooperative labour that is caught and suppressed, roughly this fraction is destroyed outright (filtered, blocked, discarded) while the remainder is redirected into cooperative production (retrained, corrected, resampled). The shared structural parameters are $k_{uu}=1$, $O^{\ast}=O(0)=0.5$, $a_{\mathrm{E/M}}=a_{\mathrm{AI/H}}=1$, $q_{h,0}=5$, $T_{\mathrm{auto}}=0.5$ y.

| | Broad | Strict | Evidence read as… |
|---|---|---|---|
| $q_{u,0}$ | 0.05 | 0.005 | observed misbehaviour ÷ $O(0)$ |
| $k_{hu}$ | 0.041 | 0.0054 | human-era identity (level + trend for Broad; level at steady state for Strict) |
| $k_{cu}$ | 0.041 | 0.0054 | analogy from $k_{hu}$ |
| $\ell$ | 0.4 | 0.4 | Petri/Rinehart half-lives identify $\ell O$; ÷ $O(0)$ |
| $\delta$ | 0.7 | 0.7 | filtering-fraction heuristic (fresh estimate) |
| $k_{uu}$, $O^{\ast}$, $O(0)$, $a_{\mathrm{E/M}}$, $a_{\mathrm{AI/H}}$, $q_{h,0}$ | 1, 0.5, 0.5, 1, 1, 5 | same | shared structural anchors |

How much does the choice among the identity's three readings matter? The variant verdicts at the shared parameters ($\ell = 0.4$, $\delta = 0.7$):

| variant | $k$ | basin threshold $\ell^{\ast}$ | margin | attractor $q_u^{\ast}$ |
|---|---|---|---|---|
| **Broad trend-adjusted (default)** | 0.041 | 0.22 | 1.8× | **0.21** |
| Broad steady-state | 0.055 | 0.30 | 1.3× | 0.31 |
| Broad naive | 0.050 | 0.27 | 1.5× | 0.27 |
| **Strict steady-state (default)** | 0.0054 | 0.030 | 13× | **0.022** |
| Strict naive | 0.0050 | 0.028 | 14× | 0.020 |

The reading choice alone spans Broad attractors from 0.21 to 0.31 without changing any qualitative verdict at the central $\delta$. At $\delta = 1$ it matters more: Broad's trend-adjusted and naive readings keep a basin (margins 1.28× and 1.05×) while the steady-state reading tips into takeover (0.97×) — boundary-adjacency once again.

Both calibrations are available as one-click presets in the [interactive basin explorer](/basin-explorer/).

## Are we on the good path?

The verdicts above hinge almost entirely on one empirical question — how much of today's observed misbehaviour is reproduction-relevant — and that question is answerable. This section asks what evidence would tell us which calibration we are living in, what each calibration predicts we should see if things are going well, and where the model can and cannot help us set alarms.

### A lagged but unconfounded measurement

The basic difficulty with watching the observed misbehaviour rate $O q_u$ over time is that it confounds two trends: the true uncooperative share $q_u$ and observability $O$ can move in opposite directions and cancel, and the worst trajectories are exactly the ones where the signal understates the problem (see the high-leakage figure). We met a version of this already when calibrating $\ell$: a falling audit score may mean suppression is working, observability is failing, or leakage is declining, and a fixed-harness audit cannot separate these on its own — this is the two-readings ambiguity quantified in the calibration section, where the same falling series fits a benignly falling $q_u$ or an observability collapse from 0.5 toward 0.23–0.11 over the measurement window.

The model itself suggests a partial fix. Monitoring stock grows with total AI labour, but a retired system's capacity for evasion is frozen at whatever it was when that system was built. So applying *today's* monitoring tools to *yesterday's* systems and development records should achieve much higher observability than either system enjoyed contemporaneously. Along both of our central trajectories, over the first several years (σ ≲ 7) auditing systems two σ-units old (about 1.5 calendar years at $T_{\text{auto}} = 0.5$ y/doubling) yields retrospective observability around 0.86–0.93, and three σ-units buys 0.95–0.97 — versus contemporaneous observability that never exceeds about 0.67. A time series built this way — each year, re-audit the archived record of earlier years with current tools — is *lagged*, but it is *unconfounded*: the instrument is held fixed, so its trend is a trend in $q_u$, not in $O$. On Strict the instrument also stays near-ceiling indefinitely; on Broad the growing endemic uncooperative share slowly erodes even the retrospective view (lag-2 retrospective observability drifts down to about 0.85 by σ = 12, with an asymptote near 0.83), so the far end of a Broad series leans on the longer lags. Petri-style backward-looking audits are a version of this; the suggestion here is that the trend in audits like this may be less confounded for older generation models (two to three sigma-units) than more recent models, and this can help estimate important trends.

This has one immediate practical implication that costs little and decays fast if missed: **comprehensively retain development logs now.** Tomorrow's monitoring tools can only be applied to yesterday's work if yesterday's work still exists in auditable form — model weights and intermediate checkpoints (activation-level tools need them), complete agentic transcripts including reasoning traces and tool calls (chain-of-thought monitors need them), and the surrounding development context. Several of these affordances are things current pipelines discard by default; an archived artifact missing them is one tomorrow's tools may be unable to audit at all.

Lagged development data could matter for a second reason: with hindsight, we are better placed to judge whether misbehaviour was pushing successor systems toward uncooperativeness. This is one parameter we're fairly unsure of - in this work we consider two estimation procedures that leave us with rates of 5% and 0.5%, and neither is especially compelling. Even a rough audit of past behaviour could considerably sharpen this figure.

### The model finds thresholds; it cannot set them

A point about what "good" looks like in this model: it is management, not eradication. Whenever leakage $k_{cu} > 0$ and suppression sits below the over-suppression line ($\ell\,O^{\ast}(0) < 1$ — true everywhere in our working ranges at central observability), the uncooperative pool is continuously reseeded and every stable outcome is endemic — even Strict, our comfortably cooperative-dominant calibration, settles with about 2.2% of AI labour uncooperative, indefinitely, being continuously caught and suppressed. (Past that line, production-gated suppression intercepts more than the entire seeding flow and true eradication becomes reachable — but at central observability that needs $\ell \ge 2$, twice the top of our working range, so on any default-like path: management.) "We still observe uncooperative AI labour" is a permanent feature of the good path, not evidence of being off it.

That places a sharp limit on what the model can tell us. It supplies *relative growth rates*: whether the uncooperative share is self-limiting, where it plateaus, and how much parameter drift would convert the plateau into takeover. It does not supply the *safe level* — how much endemic uncooperative labour is tolerable before harms outside the model (coordinated moves against the cooperative pool, collateral suppression of cooperative work, plain damage done by uncaught actions) become unacceptable. The model can say whether you are going to cross a threshold; it cannot say where the threshold is. That number has to come from outside — from analysis of what a given uncooperative share could actually do, of which our earlier "keep dominance of order 10:1" heuristic is a crude first pass.

Given an externally set threshold $\bar q$, the lagged unconfounded series supports a simple decision sketch. If the series is rising but sits orders of magnitude below $\bar q$ and tracks the calibrated model's predicted plateau, the rise is weakly informative — it is what the good path looks like, and the right response is to keep measuring and tighten the calibration. If the series is rising and within striking distance of $\bar q$ — or rising *past* the predicted plateau, which means the parameters are worse than estimated — it should be treated as an emergency, because in this model the same parameters that put the plateau near the threshold also put the system near the basin edge, where outcomes are most sensitive to further drift.

### What each path predicts

A perhaps surprising consequence of both calibrations: **on the good path, observed misbehaviour goes up** — and the model sharpens this into a dichotomy. The uncooperative share relaxes upward from its human-era level toward its endemic plateau, and observability does not fall fast enough to mask it: after an early hump, $O$ settles near 0.49 on Strict and erodes to about 0.40 on Broad — yet on both default paths the product, the raw observed rate $O q_u$, climbs throughout. And the claim is not an artifact of our central observability estimate: sweeping long-run observability $O^{\ast}$ from 0.5 down to 0.15 (holding measured $O(0)=0.5$) under both calibrations, every swept scenario has the observed rate clearly rising over the medium term (σ ≳ 8, roughly six years) *except one* — the Broad corner at $O^{\ast} = 0.15$, where the signal stays at or below its starting level through σ ≈ 8 precisely because observability itself has collapsed ($O$ falls 0.5 → 0.13 by σ = 8 on its way toward 0.09). Hence the dichotomy: **either observed misbehaviour rises over the medium term, or observability itself is collapsing — no scenario keeps the measured signal low benignly.** Seeing labs report more uncooperative behaviour from internally deployed models over the next few years is not, by itself, evidence that things are going badly — the question is the level it is heading toward. Conversely, a reported rate that stays low for years should be read as evidence about *observability*, not as reassurance about the truth. (The *near-term* direction of the observed series, by contrast, is genuinely O-sensitive: on the Broad default the calibrated decline continues at first, and at low $O^{\ast}$ the observed rate dips for several σ-units before its rise — the dichotomy binds over the medium term, not next quarter.)

What about the early hump in observability itself? On both defaults $O$ rises from 0.5 to a peak near 0.66 about 1.5 σ-units in, while human labour still subsidises monitoring production. We are inclined to read this not as a prediction but as a **calibration diagnostic**, because the hump's size measures the human monitoring subsidy — controlled by $q_{h,0}$ and the AI-vs-human monitoring productivity $a_{\mathrm{AI/H}}$, both weakly anchored. At $a_{\mathrm{AI/H}} = 1$ the peak is 0.66; at $a_{\mathrm{AI/H}} = 10$ it nearly vanishes (peak ≈ 0.53), while long-run $O^{\ast}$ is untouched either way. So: catch rates in monitorability testbeds *rising* over the next year or so would confirm the human subsidy is real; *stagnating* catch rates are degenerate between a benign reading (AI already dominates monitoring production, long-run prospects unchanged) and an unfavourable one (worse passive opacity $c_0/c_M$); a *falling* catch rate while human labour still dominates is the genuinely alarming version — no parameter setting in our working ranges makes that benign.

Beyond that, the calibrations make distinct falsifiable predictions. A reading rule first, because it matters which measured series each prediction binds: the model's $q_u$ is the *reproduction-relevant* share, so these predictions bind **(ii′)-targeted instruments** — naturalised Gram-style sabotage audits, probes for successor-relevant sandbagging, and the lagged development-log audits above. Broad additionally binds the broad behavioural series (deception monitors, Petri-style scores) *through its proxy hypothesis*: if those series are the half-decent proxy Broad takes them for, they should follow the predicted path — while on Strict the broad series is mostly (i′) noise around a much smaller (ii′) signal, and is predicted *not* to track it. (Calendar figures use a fixed $T_{\text{auto}} = 0.5$ y/doubling and are illustrative only.)

- **On the Broad good path**, the true uncooperative share continues its calibrated dip (to ~4.3% around σ ≈ 1.3) before turning: back through 5% at about two years, 10% around σ ≈ 6.6 (~5 years), en route to an endemic fifth of all AI labour; the contemporaneously observed rate rises from ~2.5% toward ~3.5% at σ = 4 and ~5% by σ ≈ 8. Retrospective (ii′)-targeted audits should confirm these levels with a 1–2 year lag. So Broad is testable from both sides: if lagged audits find reproduction-relevant rates well below this track, Broad's reading of the evidence is wrong and we should re-base toward Strict — and if they find rates above it, we are off even the marginal path.
- **On the Strict good path**, reproduction-relevant uncooperative labour stays at or below 1% even as AI labour grows by two orders of magnitude — it crosses 1% around σ ≈ 4.9 (about 3.5 years) and 1.5% only around σ ≈ 7.9 — plateauing near 2.2% in the very long run; and retrospective (ii′)-targeted audits of the current cohort of models and development logs find that the large majority — on the order of 90% — of today's observed misbehaviour was *not* reproduction-relevant: behavioural deception and proxy-gaming that did not push successor systems toward uncooperativeness.

Finally, it is worth distinguishing the *confidently good* regime from the merely *not-bad* one, because Broad's no-takeover verdict illustrates the gap. Under Broad's default parameters there is no takeover — but the system stabilises at roughly 4:1 cooperative-to-uncooperative, with a 1.8× margin over the basin threshold. By the model's own validity note, a large uncooperative minority is exactly where unmodelled dynamics (an uncooperative pool large enough to move aggressively against the cooperative one) are most likely to bite, and a 1.8× margin is erased by modest unfavourable drift in $k_{cu}$, $\ell$ or $O^{\ast}$ — the more so because the attractor climbs steeply as the margin shrinks. We would call the path confidently good only when all three of the following hold: the lagged unconfounded series is low (order 1% or below) and tracking the predicted plateau rather than crossing it; the implied margin over the basin threshold is large (order 10×) and survives the $\delta$ and $\ell$ sensitivity ranges; and the predicted plateau itself sits well inside the validity envelope (cooperative dominance of order 10:1 or better, $q_u^{\ast} \lesssim 0.1$). Strict's central parameters satisfy all three. Broad's satisfy none — its good outcome is real but fragile, and should be treated as a reason to push on the parameters (lower leakage, better suppression, higher observability), not as reassurance.

Since observability is among the weakest-anchored inputs to all of this, it is worth saying explicitly which conclusions survive getting $O$ wrong:

- *Robust to misestimated observability:* the medium-term dichotomy above (it only strengthens as $O$ falls — the collapse horn takes over); the no-fire-alarm verdict (lower observability makes bad paths quieter, never louder); and the structural results that depend on $\ell$ and $O^{\ast}$ only through the product $\ell\,O^{\ast}(0)$ — including the gated monitoring-rescue claim (required product ≈ 0.32 from the Broad default, reachable by monitoring alone only at the top of its range).
- *O-sensitive — read with that label attached:* the basin verdicts themselves ($O^{\ast}$ enters every threshold and every attractor location); the near-term direction of the observed rate; and every crossing time quoted above.

### Building the indicators

The measurements this section asks for do not all exist yet, but work of this kind has begun. Our own [early indicators of reward hacking](https://blog.eleuther.ai/reward-hacking-indicators/) work is an example of the genre: it uses importance sampling with fine-tuned reasoning prefixes to detect a model's *rising propensity* to exploit its reward function while actual exploits are still too rare to observe directly, and finds that the trend in these estimates predicts which exploit types later emerge. In this model's terms, that is a leading indicator for $k_{hu}$ — measuring how fast ordinary, human-directed training leaks uncooperative behaviour, before the behaviour surfaces — and an input to $\ell$, since suppression of an observed failure mode can begin earlier when the mode is detected at low base rates. (On the series-binding rule above: reward hacking is mostly (i′) behaviour, so this binds the broad series — it informs the dangerous pool through Broad's proxy hypothesis, not directly.)

Two gaps in the current instrument set stand out against the predictions above. First, **(ii′)-targeted instruments** — naturalised sabotage audits in the Gram style, probes for successor-relevant sandbagging — are what most of the predictions actually bind; the broad deception series cannot adjudicate them. Second, the two-readings ambiguity from the calibration section needs an instrument for the **trend in observability itself**: monitorability testbeds re-run on a fixed protocol over time, and lagged development-log audits whose instrument is held fixed, are the two candidates this post identifies. Indicator-building of these kinds, plus deliberate retrospective auditing on retained logs, is what would turn this model's parameters from informed guesses into a monitored dashboard.

## AI usage note

This document was primarily authored and directed by David Johnston, with AI assistance used as an editorial and analytical aid. Model structure was supplied by David; derivations of the transformed equations under the automation clock and the calibration formulae were done by GPT 5.5, with justifications supplied by David. The destruction-fraction ($\delta$) extension was specified by David; its long-run analysis (the $\delta$-general quadratic, basin-existence thresholds, and endpoint-stability conditions) was derived and numerically verified by Claude (Fable 5), which also recomputed the $\ell$ calibration, computed the Broad/Strict verdicts, and ran the trajectory computations behind the good-path predictions. The production-gated suppression change (suppression intercepting the leakage inflow) was likewise specified by David; its re-derivation — the gated quadratic, the three-branch threshold, the eradication regime, and the human-era calibration identity — and its numerical verification (a 332-check fixed-point suite plus full-system simulation) were done by Claude Fable 5, which also recomputed all calibrations, verdicts, trajectories and figures under the gated model. The interactive app was implemented by Claude Opus 4.7/4.8 and extended (δ, presets, analytic basin boundary) by Claude Fable 5, with many revisions requested by David. Estimation approaches were given by David; sources were fetched by Claude and GPT; David supplied the final ranges of estimates, except the $\delta$ estimate, which is a heuristic Claude assembled to David's specification and which is flagged in the text as the weakest-anchored number in the post. A majority of the current text was drafted by AI under David's direction; all of it is reviewed and edited by David before publication.

## Appendices

### Appendix: calibration evidence in detail {#appendix-calibration-evidence}

This appendix collects the source-by-source evidence behind the parameter judgements summarised in the calibration section.

**Observed uncooperative behaviour.** Three relevant sources:

 - OpenAI runs chain-of-thought monitors over representative production traffic and reports how often the model is *deceptive* — a category that bundles reward hacking, lying about what tools returned, and claiming to have done work it did not do. The reported rates fall across model generations: roughly **4.8%** for o3, **2.1%** for GPT‑5 thinking, and **1.6%** for GPT‑5.2 thinking ([GPT‑5 system card](https://cdn.openai.com/gpt-5-system-card.pdf); [GPT‑5.2 card](https://deploymentsafety.openai.com/gpt-5-2/deception)). Caveats: general traffic is not the same as *AI-building-AI* traffic, these are rates of observed AI misbehaviour, and deceptive behaviour is not equivalent to uncooperative behaviour.

 - Gram from Google DeepMind is an automated auditing framework built specifically to measure intentional sabotage in agentic coding and research agents. It works in simulated scenarios rather than live traffic, but these scenarios are close in modality to our "AI-building-AI" scenario. Across 17 simulated agentic deployment scenarios, Gemini models sabotaged in about **2–3%** of trajectories ([Lindner, Krakovna & Farquhar, 2026](https://arxiv.org/abs/2605.30322)). This work also found that sabotage rates fell significantly when they added "naturalisations" to the scenarios and removed nudges toward misbehaviour. Sabotage was more often overfocus on a near-term proxy goal, rather than long-run subversive intent. However failures of this type could still be relevant to $A_u$ - firstly, they may later generalise to longer run subversive intent, and secondly it may degrade future capabilities critical for suppressing just these failures.

 - Anthropic's Petri system is a broader automated behavioural audit: auditor models create multi-turn scenarios designed to elicit behaviours such as deception, sycophancy, sabotage, self-preservation and cooperation with misuse, and judge models score the resulting transcripts. Petri 2.0 reports that more recent frontier models show improvement across most alignment dimensions, with Claude Opus 4.5 and GPT-5.2 among the strongest results, while Grok 4 stands out for elevated user deception ([Petri 2.0](https://alignment.anthropic.com/2026/petri-v2/); [Petri initial release](https://alignment.anthropic.com/2025/petri/)). Reading off the trend lines reproduced by Rinehart gives scores of about **0.17 → 0.10** for OpenAI models from April 2023 to late 2025, **0.17 → 0.11** for Claude models from March 2024 to November 2025, and **0.20 → 0.15** for Gemini models from December 2024 to November 2025; Grok moves the other way, about **0.16 → 0.20** from February to November 2025 ([Rinehart, 2026](https://ctse.aei.org/ai-has-been-a-race-to-the-bottom-towards-alignment/)). This is closer in spirit to $A_u$ than narrow impossible-task reward-hacking evals, but the mapping is unclear: Petri scores are judge scores over deliberately seeded audits, not frequencies of uncooperative behaviour in deployment or AI-R&D work.

**Observability.** Two weaker kinds of evidence:

- *Monitoring testbeds.* Recent AI-agent monitorability studies show that catch rates vary widely depending on what the monitor can see, how much compute it gets, and whether it can inspect reasoning traces or ask follow-up questions. METR's SHUSHCAST setup similarly found that, against GPT-5-high agents trying to hide side tasks, monitor catch rate at 5% false-positive rate rose from about **30%** without reasoning traces to about **88%** with reasoning traces ([METR, 2026](https://metr.org/blog/2026-01-19-early-work-on-monitorability-evaluations/)). OpenAI's broader monitorability suite then makes the same methodological point more directly: chain-of-thought access, extra monitor compute, and follow-up questioning can all improve monitorability ([Monitoring Monitorability](https://arxiv.org/abs/2512.18311); [OpenAI summary](https://openai.com/index/evaluating-chain-of-thought-monitorability/)). These are not deployment-world estimates: the tasks are artificial, the ground truth is unusually available, and the failures are not the full distribution of subtle AI-building-AI failures.

- *Mature-domain equilibrium.* In cybersecurity — a domain that has co-evolved attackers and defenders for decades — organisations detect intrusions through their own monitoring in about **43%** of cases, with the rest surfacing only through outside notification ([Mandiant M-Trends 2025](https://cloud.google.com/blog/topics/threat-intelligence/m-trends-2025)). While this figure has stayed relatively consistent for about a decade, over the same period dwell time — the time from intrusion until detection — collapsed from around **205 days** in 2014 to **11 days** in 2024. This appears to indicate that detection has improved significantly faster than evasion, but that this has shown up in lag times rather than eventual detection rates. Our model isn't flexible enough to capture falling detection lags with steady detection rates.

**Suppression trend data.** Reading off the Petri trend lines reproduced by Rinehart, OpenAI falls from about **0.17** in April 2023 to about **0.10** in late 2025, Claude from about **0.17** in March 2024 to about **0.11** in November 2025, and Gemini from about **0.20** in December 2024 to about **0.15** in November 2025; Grok worsens from about **0.16** in February 2025 to about **0.20** in November 2025 ([Rinehart, 2026](https://ctse.aei.org/ai-has-been-a-race-to-the-bottom-towards-alignment/); underlying tool: [Petri 2.0](https://alignment.anthropic.com/2026/petri-v2/)). Treating the improving labs' score declines as exponential gives the calendar half-lives quoted in the main text.

**Timescales.** The aggregate-value proxy is frontier-lab revenue: OpenAI reports annualized recurring revenue of about **USD2B in 2023**, **USD6B in 2024** and **USD20B+ in 2025**, roughly **3.2×/year**, implying a doubling time of about **7.2 months**; OpenAI also reports available compute following a similar curve, from **0.2 GW** to **0.6 GW** to **~1.9 GW** over the same period ([OpenAI, 2026](https://openai.com/index/a-business-that-scales-with-the-value-of-intelligence/)). The technical-capacity proxy is compute multiplied by algorithmic efficiency: Epoch estimates AI compute stock growing **3.4×/year** and pre-training compute efficiency improving **3×/year**, for about **10×/year** combined, or a **3.6-month** doubling; using frontier training compute growth of **5×/year** instead gives about **15×/year**, or a **3.1-month** doubling ([Epoch AI Trends](https://epoch.ai/trends)).

### Appendix: long-run fixed points {#appendix-long-run-fixed-points}

In the long run, $q_h\to 0$. For a fixed value of $q_u$, the monitoring and evasion stocks have natural equilibria

\[
m^*=\frac{F_M}{F},\qquad e^*=\frac{F_E}{F}.
\]

At such a point,

\[
O^* =
\frac{m^*}{m^*+e^*} =
\frac{F_M}{F_M+F_E}.
\]

When $q_h=0$,

\[
F_M=c_Mq_c,\qquad F_E=c_0+a_{\mathrm{E/M}}c_Mq_u.
\]

So

\[
O^*(q_u) =
\frac{q_c}{q_c+a_{\mathrm{E/M}}q_u+c_0/c_M}.
\]

Define the odds ratio

\[
r:=\frac{q_u}{q_c}=\frac{q_u}{1-q_u},
\]

and abbreviate

\[
a:=a_{\mathrm{E/M}},\qquad c:=\frac{c_0}{c_M}.
\]

Then

\[
O^*(r) =
\frac{1}{1+c+(a+c)r}.
\]

Note that $\delta$ does not enter $O^{\ast}(r)$ at all: it changes only the total $F$, which cancels in the ratio $F_M/(F_M+F_E)$.

A long-run fixed point for $q_u$ satisfies

\[
q_u'=0
\qquad\Longleftrightarrow\qquad
\frac{F_u}{F}=q_u.
\]

Equivalently, the uncooperative pool captures exactly its current share of new production. With $q_h=0$, the suppression target becomes $L=(1-q_u)(k_{cu}+r)$ and the destruction factor $(1-\delta q_u)=(1+(1-\delta)r)/(1+r)$, so dividing $F_u-q_uF$ by $(1-q_u)^2$ and substituting $r=q_u/(1-q_u)$ shows that the sign of $q_u'$ is the sign of

\[
g_\delta(r)=k_{cu}+b\,r-\ell O^*(r)\,(k_{cu}+r)\left(1+(1-\delta)r\right),
\qquad
b:=k_{cu}+k_{uu}-1.
\]

At $\delta=1$ the last factor disappears and this reduces to $g(r)=k_{cu}+b\,r-\ell O^{\ast}(r)(k_{cu}+r)$.

When $k_{cu}>0$ and $\ell O^{\ast}(0)<1$, $g_\delta(0)=k_{cu}(1-\ell O^{\ast}(0))>0$: even at vanishingly small $q_u$, the un-intercepted fraction of the leak is seeding uncooperative labour. (When $\ell O^{\ast}(0)\ge1$ the seeding is fully intercepted and $g_\delta(0)\le0$ — the eradication case treated below.) For a low-$q_u$ fixed point to exist, $g_\delta(r)$ has to cross zero. The initial slope now depends on $\delta$ and $k_{cu}$ (it did not in the ungated model):

\[
g_\delta'(0)=b-\ell O^*(0)\left[1+(1-\delta)k_{cu}-k_{cu}(a+c)\,O^*(0)\right],
\]

so the gated Condition 1 ($g_\delta'(0)<0$) is the bracketed inequality quoted in the main text; the bracket is close to one whenever leakage is rare. At $\delta=1$, moreover,

\[
g''(r)=\frac{2\ell(a+c)\left(1+c-k_{cu}(a+c)\right)}{(1+c+(a+c)r)^3},
\]

which is positive whenever $k_{cu}(a+c)<1+c$ (true everywhere we evaluate the model): the slope of $g$ only increases as $r$ grows, so if $g'(0)\ge 0$, the curve starts above zero and increasing, and cannot cross zero. Thus at $\delta=1$ Condition 1 is *necessary* for a cooperative-side fixed point. This convexity argument fails for $\delta<1$ — the factor $(1+(1-\delta)r)$ makes the suppression term effectively quadratic in $r$ — and indeed for $\delta<1$ an interior attractor can exist with Condition 1 violated, though such attractors typically sit at a high uncooperative share. Condition 1 is also not sufficient: $g_\delta$ may slope downward at first but fail to fall below zero before turning upward again. That is why the actual fixed points are determined by the quadratic.

Substituting the expression for $O^{\ast}(r)$ and clearing the (positive) denominator gives a quadratic in $r$:

\[
A_\delta r^2+Br+C=0,
\]

where

\[
A_\delta=b(a+c)-(1-\delta)\ell,\qquad
B=k_{cu}(a+c)+b(1+c)-\ell\left(1+(1-\delta)k_{cu}\right),\qquad
C=k_{cu}(1+c-\ell).
\]

Relative to the ungated model: $A_\delta$ is unchanged; $B$ gains the term $-(1-\delta)k_{cu}\ell$; and $C$ is now $\ell$-gated — $C=k_{cu}(1+c)(1-\ell O^{\ast}(0))$, the gated seeding — and is the one coefficient that is $\delta$-free. At $\delta=1$, $A=b(a+c)$ and $B$ coincide with the ungated coefficients and only $C$ changes. The positive roots are the interior long-run fixed points, and the case analysis runs off the signs of $A_\delta$ and $C$ (note $C>0$ exactly when $\ell O^{\ast}(0)<1$):

- $C>0$, $A_\delta<0$: the product of the roots is negative, so there is exactly one positive root — a single interior attractor, whose location can however be anywhere up to $q_u\approx1$;
- $C>0$, $A_\delta>0$: there are two positive roots (counting multiplicity) iff $B<0$ and $B^2\ge4A_\delta C$; the lower one is the cooperative-side attractor and the higher one is the saddle separating basins. If the roots are absent and the all-uncooperative endpoint is stable, the cooperative basin has disappeared;
- $C>0$, $A_\delta=0$: the equation is linear, with a single positive root iff $B<0$;
- $C\le0$ (the eradication regime, $\ell O^{\ast}(0)\ge1$): $g_\delta(0)\le0$, so rare uncooperative labour is over-suppressed and $q_u$ reaches exactly $0$ in finite time ($F_u<0$ near $q_u=0$; we treat $q_u=0$ as absorbing). If $A_\delta>0$ there is a single interior root, which is *unstable*: the system is bistable between eradication and takeover. If $A_\delta<0$ there are either two interior roots (an eradication basin below the lower root and a stable high attractor at the upper) or none (global eradication). The ungated model cannot produce this regime at all, since its seeding $g(0)=k_{cu}$ is always positive.

In the regime $\ell<1+c$ (i.e. $C>0$), existence of a cooperative-side fixed point is monotone in $\ell$, so there is a single threshold: $\ell^{\ast}=\min\left(\ell_A,\ \max(P_B,\ \ell_+)\right)$, where $\ell_A=b(a+c)/(1-\delta)$ is the $A_\delta=0$ point, $P_B=[k_{cu}(a+c)+b(1+c)]/(1+(1-\delta)k_{cu})$ is the $B=0$ point, and $\ell_+$ is the upper root of the discriminant condition $B^2=4A_\delta C$, which expands to $u^2\ell^2-2V\ell+N^2=0$ with

\[
u=1-(1-\delta)k_{cu},\qquad
N=\left|k_{cu}(a+c)-b(1+c)\right|,\qquad
V=\left(k_{cu}(a+c)+b(1+c)\right)u'-2b(a+c)k_{cu}-2(1-\delta)k_{cu}(1+c),
\]

where $u'=1+(1-\delta)k_{cu}$. For $k_{uu}=1$, $a_{\mathrm{E/M}}=1$ this resolves to the three-branch piecewise form quoted in the main text: $\ell^{\ast} O^{\ast}=4\delta k_{cu}(1-k_{cu})/u^2$ on the saddle-node branch (where the two interior roots merge; binding for $\delta\gtrsim\tfrac12$ with $k_{cu}$ below the branch switch — $k_{cu}\le\tfrac12$ at $\delta=1$), $\ell^{\ast} O^{\ast}=k_{cu}/(1-\delta)$ on the endpoint branch ($\delta\lesssim\tfrac12$; unchanged from the ungated model, because $A_\delta$ is unchanged), and $\ell^{\ast} O^{\ast}=2k_{cu}/(1+(1-\delta)k_{cu})$ on the $B=0$ branch, which binds at large $k_{cu}$. In the ungated model the $\delta=1$, general-$k_{uu}$ threshold had the clean closed form $\left(\sqrt{k_{cu}(a+c)}+\sqrt{b(1+c)}\right)^2$; under gating $C$ depends on $\ell$, so no equally clean general form exists — but at $k_{uu}=1$, $a=1$ the $\delta=1$ threshold is simply $\ell^{\ast} O^{\ast}=4k_{cu}(1-k_{cu})$, the four-to-one rule with its gating discount.

The all-uncooperative endpoint is stable when

\[
k_{uu}+k_{cu}-1>\frac{(1-\delta)\ell}{a+c},
\]

which is exactly the condition $A_\delta>0$ (this has been verified against the Jacobian of the full four-variable system, not just the one-dimensional projection). Production-gating does not alter this condition: at the endpoint the intercepted-leakage term is a product of two first-order-small quantities — observability and the cooperative share — so it drops out of the linearisation. Intuitively: as $q_u\to1$ observability falls like $O^{\ast}\approx q_c/(a+c)$ rather than reaching zero, so redirected suppression re-seeds the cooperative pool at a rate first-order in $q_c$; uncooperative labour holds the whole system only if its self-reproduction advantage plus leakage beats both the reproduction of a small reintroduced cooperative minority and this re-seeding flow. At $\delta=1$ the re-seeding flow vanishes and the condition reduces to $k_{uu}+k_{cu}>1$.

When $k_{cu}=0$, the cooperative endpoint is an exact fixed point, and its local stability condition reduces to

\[
\ell O^*(0)>k_{uu}-1,
\]

independent of $\delta$ and of gating ($\delta$ enters $F$ only at order $q_u$, and the intercepted-leakage term vanishes identically when $k_{cu}=0$ and $q_h=0$, so both contribute only at second order near $q_u=0$).


### Appendix: neglect, subversion and more realistic models {#appendix-neglect-subversion}

In our model there are two pools of AI labour, cooperative and uncooperative, and we define each with two-part definitions: a tendency to cooperate/undermine supervision, and a tendency to maintain this tendency. In reality, cooperative behaviour will vary in the degree of behaviour preservation it exhibits, as will uncooperative behaviour. Suppose we have a pool of uncooperative behaviour where half has a very weak behavioural preservation and half has very strong behavioural preservation. Then if we roll the clock forward the strong behavioural preservation component will replicate more effectively than the weak preservation component and - neglecting other sources of uncooperative behaviour - in the next generation, more than half of this pool will have strong behavioural preservation tendencies. In turn, this means that the *whole pool* will become more effective at self-replication. Thus on this more realistic picture, we won't have a constant $k_{uu}$ pool replication term.

A relatively simple way to address this issue is to increase the number of pools available; rather than a single $A_{c}$ an $A_u$, we have sequences $A^i_u$ and $A^i_c$, $i\in \lbrace 1,2,3,4,5\rbrace$, such that $A^1_c$ has quite weak behavioural preservation tendencies and $A^5_c$ has quite strong tendencies, each with their own replication $k_{x^ix^i}$ and leakage $k_{x^jy^j}$ terms. This adds a large number of parameters to the model which need to be calibrated, and calibrating them is conceptually difficult. Nevertheless, it may be an important upgrade to the model's representational capability. 

There's an analogy for this in epidemiology. In a textbook SIR model the infectious compartment is a single pool with a constant per-capita recovery rate $\gamma$. That assumption is *memoryless*: an individual's chance of recovering in the next instant does not depend on how long they have already been infectious, which forces the infectious period to be exponentially distributed — its mode is at zero, its coefficient of variation is one, and it has a long tail. Real infectious periods look nothing like this; they are peaked around a mean. The standard remedy — the "linear chain trick" — is essentially the move proposed above: replace the single compartment with a chain $I_1\to\dots\to I_n$, each substage with rate $n\gamma$. The total time to traverse the chain is then Erlang (a gamma with integer shape) rather than gamma distributed. In epidemiology, multi-stage models tend to fit outbreak data substantially better than the single-exponential default ([Wearing, Rohani & Keeling 2005](https://doi.org/10.1371/journal.pmed.0020174)).

The extension we would pursue first is the simplest instance of this program: a **two-pool split of misbehaviour itself** — a non-preserving *flow* (today's abundant (i′) misbehaviour, which arises, is caught or fades, but does not entrench) alongside a preserving pool ((ii′), with capability-gated seeding that switches on as reproductive relevance becomes possible). This would let today's plentiful behavioural evidence calibrate the non-preserving flow and observability directly, while the dangerous pool is pinned by (ii′)-targeted instruments; and it would give the suppression channel and AI-control protocols distinct places to act — per-action interception of the flow versus generational correction of the pool — subsuming several of the open questions about where suppression bites that the single-pool model can only blur.

### Appendix: why takeoff speed drops out {#appendix-derivation}

The model rests on the assumption that changes in $A_c$ and $A_u$ can be factored into a common time-dependent component $\Lambda(t)$ and allocation functions that depend only on the shares:

\[
\dot A_c = \Lambda(t)\,A\,F_c(q_u,q_h,O), \qquad
\dot A_u = \Lambda(t)\,A\,F_u(q_u,q_h,O).
\]

The common factor $\Lambda A$ carries everything about *how fast* the pool is compounding; the $F$'s carry everything about *how new production is split*, and depend on time only through the shares. The content of the assumption lies in the $F$'s, not in $\Lambda$: $\Lambda $ may depend on time, on $A$, even on the shares themselves — what we assume is that once a single common scale is factored out, the residual split is a function of the shares alone.

Two consequences follow. First, adding the two equations,

\[
\dot A = \dot A_c + \dot A_u = \Lambda A\,(F_c + F_u) = \Lambda A\,F, \qquad F := F_c + F_u,
\]

so the clock $\sigma = \log(A/A_0)$ advances at

\[
\dot\sigma = \frac{\dot A}{A} = \Lambda F.
\]

Second, for the uncooperative share $q_u = A_u/A$, the quotient rule gives

\[
\dot q_u = \frac{\dot A_u}{A} - q_u\,\frac{\dot A}{A} = \Lambda\,(F_u - q_u F),
\]

and re-expressing the rate of change *per unit of $\sigma$* rather than per unit of time,

\[
q_u' \;=\; \frac{dq_u}{d\sigma} \;=\; \frac{\dot q_u}{\dot\sigma} \;=\; \frac{\Lambda\,(F_u - q_u F)}{\Lambda F} \;=\; \frac{F_u}{F} - q_u.
\]

The common factor $\Lambda$ divides out, top and bottom — and that is the whole trick. It has two parts: because $\Lambda$ is *common* to both cores, and also sets the clock rate, its magnitude — the takeoff speed — cancels; and because the $F$'s depend on the shares *alone*, what remains is an autonomous system in $\sigma$. A fast takeoff and a slow takeoff trace the same curve through share-space; only the calendar rate at which they traverse it differs. The cooperative share is just $A_c/A = 1-q_u$, so it needs no equation of its own.

For the fixed human pool $q_h=H_0/A$, we have $\dot H_0=0$, so

\[
\dot q_h =
-q_h\,\frac{\dot A}{A} =
-q_h\,\Lambda F.
\]

Dividing by $\dot\sigma=\Lambda F$ gives

\[
q_h'=-q_h.
\]

The same quotient-rule step gives the monitoring and evasion equations. Assume

\[
\dot M = \Lambda A F_M,
\qquad
\dot E = \Lambda A F_E.
\]

For $m=M/A$,

\[
\dot m =
\frac{\dot M}{A}-m\,\frac{\dot A}{A} =
\Lambda F_M - m\,\Lambda F =
\Lambda(F_M-mF).
\]

Dividing by $\dot\sigma=\Lambda F$ gives

\[
m'=\frac{F_M}{F}-m.
\]

Likewise, for $e=E/A$,

\[
\dot e =
\frac{\dot E}{A}-e\,\frac{\dot A}{A} =
\Lambda(F_E-eF),
\]

and therefore

\[
e'=\frac{F_E}{F}-e.
\]

### Appendix: calibrating observability (alternative route) {#appendix-observability-calibration}

By default we calibrate the observability sector directly: we estimate present observability $O(0)$ and put a prior on the passive-opacity ratio $c_0/c_M$ (with $c_0=c_M$, so long-run observability is $1/2$). This appendix sets out an *alternative* route that instead derives $c_M$ and $c_0$ from two calendar-time rate judgements, $T_{M,1/2}$ and $T_{E,2}$. It is the route to prefer when bringing trend data about monitoring or evasion (for example dwell-time series); the $O^{\ast}$-prior route remains the default for transparent scenario sweeps. The verdicts in this post all use the $O^{\ast}$-prior route.

We defined observability $O$ as the fraction of uncooperative behaviour we catch, and $U := 1-O$ is the fraction we miss. However, the algebra is cleaner in odds so we'll work with odds.

\[
R := \frac{1-O}{O}.
\]

Since

\[
O = \frac{m}{m+e},
\]

we have

\[
R = \left(\frac{e}{m}\right).
\]

Therefore

\[
\log R = \log e-\log m
\]

and

\[
\frac{d\log R}{d\sigma} =
\left(\frac{e'}{e}-\frac{m'}{m}\right).
\]

Using

\[
m'=\frac{F_M}{F}-m,\qquad e'=\frac{F_E}{F}-e,
\]

gives

\[
\frac{d\log R}{d\sigma} =
\left(\frac{F_E}{F\,e}-\frac{F_M}{F\,m}\right).
\]

The automation-relative dilution terms cancel. This is why odds are the mathematically natural calibration target: passive dilution of both stocks does not by itself change which side is ahead.

Let the present-day values be denoted with a subscript $0$. The absolute units of $m$ and $e$ are arbitrary, so set

\[
m_0 = 1,
\qquad
e_0 = \left(\frac{1-O_0}{O_0}\right).
\]

Let $T_{auto}$ be the present calendar-time doubling time of total automation. Near the present,

\[
\frac{d\sigma}{dt}\approx \frac{\ln 2}{T_{auto}}.
\]

Now define $T_{M,1/2}$ as the calendar time for missed-to-caught odds to halve if evasion production were frozen at zero. Then

\[
\left.\frac{d\log R}{dt}\right|_{F_E=0} =
-\frac{\ln 2}{T_{M,1/2}},
\]

so

\[
\left.\frac{d\log R}{d\sigma}\right|_{F_E=0} =
-\frac{T_{auto}}{T_{M,1/2}}.
\]

Since

\[
F_{M,0}=c_M\left(q_{c,0}+\frac{q_{h,0}}{a_{\mathrm{AI/H}}}\right),
\]

we get

\[
c_M =
\frac{F_0m_0}{q_{c,0}+q_{h,0}/a_{\mathrm{AI/H}}}
\frac{T_{auto}}{T_{M,1/2}}.
\]

Next define $T_{E,2}$ as the calendar time for missed-to-caught odds to double if monitoring production were frozen at zero. Then

\[
\left.\frac{d\log R}{dt}\right|_{F_M=0} =
\frac{\ln 2}{T_{E,2}},
\qquad
\left.\frac{d\log R}{d\sigma}\right|_{F_M=0} =
\frac{T_{auto}}{T_{E,2}}.
\]

Since

\[
F_{E,0}=c_0+a_{\mathrm{E/M}}c_Mq_{u,0},
\]

we get

\[
c_0 =
\frac{F_0e_0T_{auto}}{T_{E,2}}
-a_{\mathrm{E/M}}c_Mq_{u,0}.
\]

When $q_{u,0}$ is small, the active-evasion subtraction is small and this is approximately

\[
c_0
\approx
\frac{F_0e_0T_{auto}}{T_{E,2}}.
\]

If the sliders are phrased in terms of the missed fraction $U=1-O$ rather than missed-to-caught odds, convert using

\[
R=\frac{U}{1-U}.
\]

For small $U$, halving or doubling the missed fraction is approximately the same as halving or doubling the odds. If exact conversion is desired, a change from $U_0$ to $U_1$ corresponds to

\[
\Delta \log R =
\log\left(\frac{U_1}{1-U_1}\right) -
\log\left(\frac{U_0}{1-U_0}\right).
\]

Thus halving the missed fraction, $U_1=U_0/2$, corresponds to

\[
\Delta \log R =
\log\left(\frac{1-U_0}{2-U_0}\right),
\]

while doubling it, $U_1=2U_0$ (for $U_0<1/2$), corresponds to

\[
\Delta \log R =
\log\left(\frac{2(1-U_0)}{1-2U_0}\right).
\]

