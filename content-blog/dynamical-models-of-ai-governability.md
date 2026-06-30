---
title: "A Dynamical Model of AI Governability"
date: 2026-06-15
description: "A toy dynamical model of whether the AI workforce that builds future AI ends up cooperative or uncooperative: where the basin boundary lies, what current evidence says about which side we are on, and what would tell us we are on the good path."
author: ["David Johnston"]
ShowToc: true
mathjax: true
draft: false
---

# Introduction

Deceptive alignment stories depend on races: misaligned AI systems get ahead by acting deceptively while supervision lags too far behind to uncover their duplicity. We depend on AI assistance to keep pace in the oversight race, and they help us keep pace as long as we're able to catch and fix any tendencies they have to act uncooperatively. If we fall behind in the oversight race, we lose the tools we were relying on to keep pace, leading to a stable loss of control.

The outcome thus depends on competing quantities - advanced AI will be better at evading misbehaviour monitors, better at building misbehaviour monitors, and whether it actually helps to build misbehaviour monitors is complex and path dependent. It is very hard to resolve questions of where this is going via verbal analysis - we really need a quantitative model to understand the system. In this post we'll be building just that - a quantitative dynamical model of AI governability. Our model is in the same family of the [AI Futures model](https://www.aifuturesmodel.com/); it's an ambitious attempt to model an important system (in their case, AI capabilities take off; in our case, retaining control of advanced AI), and while we have spent considerable time critiquing and refining the model and finding empirical anchors for quantities where possible, by necessity it is less rigorous than would typically be found in academic literature.

Given the challenges involved, what are the benefits of creating a model like this? Firstly, it give you predictions of the outcomes of building advanced AI - do we succeed in building systems that support our aims or not? There is enough uncertainty in multiple parameters that the model shouldn't be viewed as picking out a particular, likely trajectory. Rather it can help identify key uncertainties that flip the prediction, and points of intervention that may also exert significant influence.

More ambitiously, we think there would be considerable benefits to a robust AI takeover early warning system. Multiple factors make AI governance a very difficult problem:
 - AI takeover is a plausible if uncertain outcome of continued AI development, and if we were truly headed toward takeover extreme actions would be justified to avoid it
 - The benefits AI brings promise to be very large, so actions to control AI can have have huge costs and attract fierce opposition
 - Building the option to, for example, completely halt AI development requires centralising a huge amount of power in the hands of the regulator
 - There's widespread disagreement about which risks are most pressing, and how pressing those risks are

*If* it were possible to build a robust early warning system for loss of control, it would go a long way to mitigating these difficulties. If we know with precision whether we are tracking along a safe or unsafe trajectory, we can be more confident our interventions are sufficient to keep us on the safe path while at the same time avoiding doing too much and losing a great deal of value from AI as a result. Even if AI development poses grave risks, information about where we are tracking can still be shared freely, unlike control over that development, and well informed people everywhere will make choices based on that information. In this way, an early warning system supports powerful decentralized responses to risks from advanced AI, which can complement or even partially substitute for centralized control of the industry.

We do not claim we have built such a system! However, any early warning system necessarily depends on a predictive model of outcomes, given our current best guesses about the state of the world. We're not aware of any existing attempts to build such a model. Thus we see this model as a proof of concept for a critical component of an early warning system - we've built and refined a model, we're presenting it here along with insights we took from the exercise and challenges we've not yet overcome, and we want to test it against criticisms from AI safety experts.

Everything in this post can be explored interactively: the [basin explorer](/basin-explorer/) implements the full model with every parameter on a slider, one-click presets for the calibrations we discuss, and an outcome map showing which parameter combinations lead where.

## The results in brief

We model the workforce that builds future AI as two competing pools — self-propagating cooperative work, and self-propagating uncooperative work (henceforth just *cooperative* and *uncooperative*) — connected by a handful of rates: how effectively each pool can propagate itself, how often cooperative development *leaks* uncooperative systems, how much of current misbehaviour our detect-and-fix machinery covers and how fast it corrects what it covers (these are explained in detail in [the next section](#explaining-the-model)).

{{<figure src="/images/blog/dynamical-models/outcome-map.png" width="85%" align="center" caption="Long-run uncooperative share across two key parameters: how often cooperative development leaks uncooperative systems (leakage, $k_{cu}$), which can be thought of (imprecisely) as a measure of how hard alignment is, and how fast uncooperative labour reproduces (growth advantage, $k_{uu}$), with all other parameters held at their central estimates. Green is contained, red is takeover. The central estimate sits just inside the contained region." />}}

<!-- REVIEW (new filter+γ figure, 2026-06-18; placement provisional — could instead open the rescue/scenario section; caption is a DRAFT, please rewrite): -->
{{<figure src="/images/blog/dynamical-models/trajectory-overlap.png" width="100%" align="center" caption="DRAFT — Plausible futures: sampled model trajectories. Each band is the 85% spread (with a median line) over a Monte-Carlo ensemble that draws the uncertain parameters — leakage $k_{cu}$, growth advantage $k_{uu}$, human leakage $k_{hu}$, the domain offense/defense ratio, and the observability dial $\Omega$ — from plausible ranges; green = runs that stay below 10% uncooperative, red = runs that exceed it, dotted = the central estimate. Left: what you would observe (the contemporaneous rate $O_{\mathrm{obs}}q_u$). Right: the truth (uncooperative share $q_u$). x-axis: AI labour ÷ human labour ($A/H_0$, log; handoff at 1×). The observed-rate bands stay low and ambiguous even where the truth bands fan out toward takeover." />}}

**1. Both success and failure to suppress uncooperative takeover are plausible.** Under our central parametrization uncooperative behaviour is contained to ~25% of the AI labour pool (albeit it takes a very long time to reach this level). If "alignment is hard" (specifically, production of unintended AI misbehaviour grows sharply), uncooperative AI systems find highly effective growth strategies or it is naturally easier to evade detection than to detect misbehaviour, then the model predicts uncooperative takeover, depending more sensitively on the first two parameters (unintended misbehaviour and growth rates) than the third (defense/offence balance).

**2. Failure is often gradual, but we can't rule out fast failures.** Our model features a "clock" of total AI labour such that 1 unit is equal to the current amount of human labour producing AI systems. Our "failing" paths often reach $100$ labour units when the uncooperative pool crosses 20% (around 3 years separates 1 labour unit from 100 under naive exponential extrapolation). This is a product of the modelling strategy where many thigns are treated as if they will continue to evolve as they have been already, with the main difference being that uncooperative AI begins to be able to reproduce at meaningful rates. However, we can't rule out parameter setting where failure is rapid.

**3. Observed misbehaviour is an imperfect indicator of uncooperative takeover.** Generally, observed misbehaviour rises together with actual misbehaviour. However, on bad trajectories the uncooperative labour undermines supervision, so this signal is attenuated even if it's usually not fully suppressed. Because we are highly uncertain about observability, for every "good" trajectory, there is a "bad" trajectory that looks similar and vise-versa. Thus naively tracking observed misbehaviour rates gives *some* indication of the trajectory, but it's discriminating power is limited. There is direct evidence that the gap runs one way — re-scoring a fixed model under a newer audit harness surfaces *more* misbehaviour, so what we observe reads as a lower bound (see [Predictions and postdictions](#predictions-and-postdictions)).

**4. Divergence between "good" and "bad" paths can happen significanly beyond AI-human parity.** Safety focused policy feedbacks will have to be capable of operating in a world where AI dominates the labour pool.

**5. Some practical observations could significantly reduce our uncertainty.** The model points to measurements that would be especially informative. Retrospective audits of older models could help estimate, with lag, what fraction of misbehaviour contemporaneous audits were able to catch. Estimates of the “reproduction numbers” of uncooperative behaviours — including productive misuse, data poisoning, or [vertical transmission](https://www.alignmentforum.org/posts/nLrrYweeFxgXACSmS/sft-drives-gemini-s-safety-properties-1) — would help ground one of the model’s most important parameters, though collecting such estimates would be challenging. The hardest parameter to refine is probably “leakage”: how often cooperative development produces uncooperative systems. But even here, it should be possible to improve on the constant-rate assumption used in our model.

## Explaining the model

### What is being modelled {#what-modelled}

We're not modelling whether the outcomes are ultimately good, all things considered. We're modelling whether AI that produces the next generation of AI becomes mostly cooperative or mostly uncooperative with regard to human-aligned oversight.

A note on terminology: we say "uncooperative" rather than "misaligned" throughout. The post gives "uncooperative" a specific two-part definition below, while "misaligned" is used in many different senses in the literature; where we discuss others' scenarios we map their "misaligned" onto our "uncooperative" explicitly.

**Cooperative vs uncooperative behaviour:** Consider the AI labour that goes into building *more* AI, which we'll call $A$. Some of this labour might go into producing future systems that become harder to oversee, more resistant to correction and better able to produce successor systems with similar tendencies. Other labour will go into producing future systems that help to improve oversight, make corrections more robust and well-targeted and better able to produce successor systems with similar tendencies. Both directions are self-reinforcing: if the next generation of AI systems is dominated by uncooperative labour, then the subsequent generation will be even more uncooperative, and vice versa.

We also have human labour $H_0$ that goes into building more AI. While in reality this will change with investment in AI, we treat it as fixed for simplicity.

Here $A$ should be read as an index of effective AI-development labour, not as a raw input such as GPU-hours, tokens or tasks completed. The units are chosen so that AI and human effort can be compared on the same scale: if $A=H_0$, then removing AI labour would roughly halve gross AI-development progress; if $A=2H_0$, it would remove roughly two thirds. This is a simplification. In reality, human and AI labour may be complements in some tasks and substitutes in others. We mostly leave those details inside the process by which $A$ grows over time, which this model does not try to predict.

We benchmark the turnover of this pool against the recent rate of "major" AI model releases (which we take, based on a rough but informed appraisal, to be a 0.5-point version difference in most instances). This we consider to represent a half-turnover of the model training pipeline (measured WRT influence on meaningful model behaviour, not literal number of datapoints). This reflects a working assumption that the *most* important input for determining cooperativeness is the training pipeline, moreso than (for example) the deployment context. 

A key asymmetry (and a key factor that makes this model interesting) is that what we might call "passive deterioration" affects cooperative behaviour more strongly than uncooperative behaviour. That is, if the current work on the next generation training pipeline neglects to attend to cooperativeness one way or another, then cooperativeness of the next generation of models will probably decline (both in terms of cooperativeness as measured by audits, and in terms of building it into the next-next generation). One need not deliberately train oversight subversion into models; they passively pick it up because subversion is often instrumental for one reason or another. We model this explicitly via a one-way *leakage* term, where cooperative labour now unintentionally produces some uncooperative labour in the next generation via oversights, insufficient understanding and so forth.

We model both cooperative and uncooperative behaviour as single pools, but this is a coarse approximation. In reality, some work will be highly cooperative and attentive to the behaviour of the next generation, some cooperative in a mediocre way, some indifferent and some actively subversive in a focussed manner. These distinctions could end up mattering if, for example, the segments that most effectively propagate themselves end up dominating the labour pool (see [Appendix: neglect, subversion and more realistic models](#appendix-neglect-subversion) for further discussion). Nonetheless, for this round of modelling we prioritize simplicity and keep the pools coarsely lumped.

As a starting point, we treat the ability of each pool to reproduce itself as equal, given equal resources (excluding leakage from the cooperative pool - so in fact the cooperative pool is somewhat disadvantaged at baseline). An argument for this in the short run is that reproduction is dominated by momentum - i.e. new training pipelines inherit a lot from old training pipelines based on convention, what tools have already been created and so forth - and so behaviours persist in the next generation roughly at the same level as behaviours in this generation. In the long run, we can consider the strategy-stealing assumption ([Christiano, 2019](https://www.alignmentforum.org/posts/nRAMpjnb6Z4Qv3imF/the-strategy-stealing-assumption)): whatever strategies one pool can use to *deliberately* turn resources into successors, the other can copy, so neither side gets an intrinsic propagation advantage from deliberate competition. These reasons are non-decisive, but we take them as an adequate justification to take a baseline where the self-propagation abilities of each pool are equal (again, before factoring in leakage).

**Defining cooperative and uncooperative behaviour:** Our precise definition of "uncooperative behaviour" is behaviour that would be flagged as such by sufficiently mature retrospective audits of the relevant frozen model, scaffold and development context. Aspirationally, this means "behaviour that is undesirable according to developers, under a suitably enlightened criterion of desirability and with perfect coverage". Our definition is only a proxy for that aspiration, but under plausible assumptions it is a functional proxy. Cooperative behaviour is the complement of this.

The level at which we assess cooperativeness is more granular than the turnover benchmark. We consider cooperativeness to be measured per meaningful task completed. There's no way to call a model that acts uncooperatively 5% of the time "cooperative" or "uncooperative" without losing important information.

Strictly, we're interested in *self-propagating* cooperative and uncooperative behaviour. Cooperative behaviour is behaviour that (i) is cooperative according to our working definition and (ii) tends to instil both (i) and (ii) in successor instances. Uncooperative behaviour is behaviour that (i') is uncooperative according to our working definition and (ii') tends to instil both (i') and (ii') in successor instances. Given that successor influence is a key channel both for misbehaviour to cause major problems and for desired behaviour to prevent further misbehaviour, mature behaviour audits will ideally catch this dimension too. We nevertheless make it explicit because self-propagation is central to the model.

Behaviour audit based definitions have a well-known issue: if misbehaviour audits do not track misbehaviour as we would assess it from a sufficiently enlightened position, then the operational definition will mislead us with respect to the property we really care about.

Our definition leans on retrospective audits - using more modern evaluation suites with older models. Retrospective audits *can* still fail for the same reasons - if harnesses degrade, or cease improving, then we do not benefit from using more modern harnesses. However, they do buy us an observable indication that the proxy can reasonably be believed to be tracking the underlying quantity of interest.

Specifically, suppose both of the following hold:
 - We are making continuous progress in monitoring *new* systems for misbehaviour; our latest audits surface misbehaviour that older audits missed (and is broadly agreed to be undesirable behaviour)
 - Our audits are saturating on historical models in historical development contexts

Together, these imply that the new ways to detect misbehaviour we are producing are not finding many more misbehaviours in older models/development contexts, which is suggestive of an asymptote. That is, if we investigated these models with excellent coverage from a suitably enlightened position, we still wouldn't find much more misbehaviour. Note that we currently do observe rising misbehaviour rates with newer evaluations on fixed models, see [Predictions and postdictions](#predictions-and-postdictions).

This is not a bulletproof justification. A suitably determined attacker could, perhaps, carefully manage which kinds of misbehaviour are revealed so as to maintain the combination of new misbehaviour detected and saturating detection on old systems. This is a delicate balance to maintain, however, and might demand a significant advantage on the uncooperative side to pull off. Thus the proxy and its observable health indicators are more plausibly reliable if cooperative dominance is continually maintained, though we don't have robust views on just how hard it is to attack. It may also be hard to tell the difference between the possibility that there are no new misbehaviours to reveal in new systems or that our progress in monitoring technology is stalling. Dedicated red teaming efforts could help to disambiguate these possibilities. Finally, both properties could hold but the convergence could be too slow to be practically relevant.

**Detect-and-fix coverage, evasion and observability:** To the extent human developers or cooperative AI developers are aware of uncooperative AI labour, they try to correct it. We define a stock of *detect-and-fix coverage* $m$ — the accumulated machinery of monitors, audits, patched training data, filters and correction loops that covers known classes of misbehaviour — and a stock of failure surfaces $e$ (which accumulates via novel failure modes introduced by the general expansion of automated labour, plus deliberate development of methods to evade the $m$ stock), such that the fraction of current uncooperative behaviour that falls under coverage is $O=\frac{m}{m + e}$. $m$ further splits into fixed observed misbehaviour $s$ and unfixed observed misbehavour $b$. Human labour and cooperative AI labour produce $m$; uncooperative AI labour produces $e$ through the deliberate-evasion channel, while the novel-mode channel ("passive opacity") grows with automation itself — expanding automation broadens the surface area to monitor — and we consider the default situation with no monitoring effort to be one where a large fraction of uncooperative actions go unnoticed. We model observed-but-unfixed problems $b$ going to observed-and-fixed problems $s$ at a constant rate $\rho$. This is an idealization of the model - in reality, that rate might itself depend on technical development (or policy responses to misbehaviour, something else the model does not include) but in this iteration we limit our model to handling technical development of the detection machinery.

Our model is a model of the evolution of these quantities - cooperative labour, uncooperative labour and documented, fixed and unobserved problems. The next section describes how these quantities evolve.

**This is a model of cooperative dominance under a static policy environment:** Under our central calibration, the uncooperative pool starts at 5% of the total AI labour pool, while the cooperative pool is a supermajority at 95%. This model captures dynamics that are important in this regime, like the uncooperative pool working to evade detection, and the cooperative pool being free to suppress any uncooperative behaviour it does detect. It does not capture dynamics that become more important when the uncooperative pool becomes a peer to the cooperative pool:
 - The uncooperative pool could actively undermine the cooperative labour, or try to capture it, rather than simply evading detection and benefiting from errors
 - The cooperative pool could run out of capacity to suppress the uncooperative pool
 - A large observed uncooperative pool could trigger redirection of resources from expanding AI capacity to suppression of uncooperative behaviour

Thus this model is "valid" when the cooperative pool is much larger than the uncooperative, and the model makes predictions *given a fixed policy/safety allocation*, not all-things considered predictions. For our analysis, we define an uncooperative share over 10% as a high risk run. While the model does separate into "endemic" ($q_u$ between 0 and 1) and "uncooperative dominant" ($q_u=1$) trajectories, we don't think the differences between high-endemic and uncooperative dominant are meaningful.

### Using automation growth as the "clock"

Total AI labour $A$ starts at a level significantly lower than $H_0$, and grows until it eclipses $H_0$. How quickly it does this is quite complicated; there are a number of economic models that aim to clarify how this could happen (the [AI Futures model](https://www.aifuturesmodel.com/); [Eth & Davidson](https://www.forethought.org/research/will-ai-r-and-d-automation-cause-a-software-intelligence-explosion) on the software intelligence explosion; the [Davidson–Epoch interactive takeoff model](https://epoch.ai/blog/interactive-model-of-takeoff-speeds)). We avoid modelling how quickly this happens; instead we look at how state variables in the model change with respect to $\sigma = \log(A/A_0)$. One unit of $\sigma$ is the amount of time required for automation capacity to rise by a factor of $e$. In some cases we report results in the more familiar unit of **doublings** of automation; one doubling corresponds to $\ln 2 \approx 0.69$ units of $\sigma$ (equivalently, one $\sigma$-unit $\approx 1.44$ doublings), or by the multiple of $H_0$ attained by $A$.

See [Appendix](#appendix-derivation) for an explanation of what assumptions enable us to use this as a clock, and how we derive the resulting system of equations. We now explain the resulting dynamics in words; the full system of equations is collected in the [model reference](#model-reference) at the end of this section.

### The dynamics in words

Because we use automation advancement as the "clock", our model uses terms relativised to the total automation level, rather than absolute quantities: the uncooperative and cooperative shares of AI labour $q_u$ and $q_c=1-q_u$, the relative human share $q_h$, and monitoring and evasion stocks per unit of automation — documented $b$, fixed $s$ and evasion $e$ — with observed fraction $O_{\mathrm{obs}}=\frac{b+s}{b+s+e}$ and fixed fraction $O_{\mathrm{fix}}=\frac{s}{b+s+e}$. The full variable list and equation block are collected in the [model reference](#model-reference) below; here we explain the dynamics one piece at a time.

**Evolution of relativised quantities:** The shares of cooperative or uncooperative behaviour, and the relativised stocks of evasion and monitoring technology all evolve according to the equations

\[
  \begin{aligned}
  x' &= \gamma(\frac{F_x}{F}-x)
  \end{aligned}
\]

Where $x\in \{u,c\}$ denotes the uncooperative and cooperative pools respectively. $q_u$ is the uncooperative share of labour and $q_c$ the cooperative share. $F_u$ is the share of new production captured by $q_u$ (analogously for $F_c$) and $F=F_u+F_c$. Each share is pulled toward the fraction of new production its pool captures. $\gamma$ is rate of turnover relative to the rate of growth; $\gamma = 1 + \tfrac{1}{o}$ where $o$ is the e-folding time of the systems that make up the labour pool (i.e. model versions, harnesses) measured in $\sigma$-units. We pick $\gamma=1.75$, benchmarking the turnover of the labour pool against the cadence of "major" (roughly half-step, 0.5-version) model releases — about one every six months, so the labour pool half-turns-over on a timescale comparable to (a little slower than) the automation doubling time. $\gamma$ controls how fast the labour pool is pulled toward the captured share of new production.

Human labour simply dilutes with respect to the $\sigma$-clock: $q_h' = -q_h$. This reflects the fact as automation grows, the human labour pool (according to our approximation) stays constant and hence shrinks as a proportion of total labour.

**Production capture:** The AI production allocation functions are:
\[
\begin{aligned}
L_k &:= k_{cu}\,q_c + k_{hu}\,q_h \qquad G_u := L_k + k_{uu}\,q_u\\[4pt]
F_u &= (1-O_{\mathrm{fix}})\,G_u\\[4pt]
F_c &= q_c(1-k_{cu}) + q_h(1-k_{hu}) + (1-\delta)\,O_{\mathrm{fix}}\,G_u
\end{aligned}
\]

Here $L_k$ is "leakage", uncooperative AI behaviour produced accidentally as a result of cooperative labour building more AI. The leakage coefficients $k_{cu}$ and $k_{hu}$ reflect, in a sense, how hard AI alignment is for cooperative AI systems and people respectively. They are rates relative to the reproduction of cooperative labour, so $k_{cu}=1$ implies all of the effort to reproduce cooperative AI ends up as uncooperative AI. $O_{\mathrm{fix}}$ is the proportion of observed AI misbehaviours that have been fixed to date. 

$k_{uu}$ represents the relative reproduction rate of uncooperative labour again relative to the reproduction of cooperative labour. Labour pools can reproduce by simply sharing traits with their ancestors (for examples, new generations of models will share significant features of their architecture with previous generations) or deliberately shaping future training pipelines, scaffolds or deployment contexts to promote their traits.

The equation for $F_u$, the fraction of new production captured by uncooperative labour, reflects the fact that as the pool turns over, $(1-O_{\mathrm{fix}})$ of the "old" problems remain and $O_{\mathrm{fix}}$ of the problems that would have been accidentally introduced are also intercepted (strictly the latter number should )

Reading the equation for $F_c$: the cooperative pool $q_c$ reproduces itself, less a fraction $k_{cu}$ that *leaks* into serving uncooperative ends; human labour $q_h$ contributes likewise, less its own leak rate $k_{hu}$; the third term is the redirected share of the full fixed flow — fixed leakage plus fixed uncooperative labour — explained below. Leakage could happen for many reasons: systems that usually behave well could be used in unexpected or unwanted ways, or systems may have generalised undesirably from training and pro-actively behave badly in some situations. 

$F_u$ picks up the un-fixed fraction $(1-O_{\mathrm{fix}})$ of leakage inflow $L_k$ plus its own reproduction at advantage $k_{uu}$ ($k_{uu}=1$ implies no advantage or disadvantage with respect to the cooperative pool). The uncooperative pool itself is also actively suppressed  $O_{\mathrm{fix}}$, hence $(k_{uu} - O_{\mathrm{fix}})\\,q_u$. The same fixed fraction gates the inflow and the stock: observed misbehaviour begets workarounds applied to the whole production pipeline — patched training data, filters, revised reward signals, usage restrictions — and these intercept misbehaviour *entering* the pool just as they suppress misbehaviour already established in it, so both are gated by the share that has actually been fixed.

Both suppression channels — intercepting incoming leakage and removing established stock gated by a single quantity, the fraction of fixed problems $O_{\mathrm{fix}} \in [0,1]$. This is the share of misbehaviour that has been observed and fixed, and so can actually be intercepted or removed. In our model, incoming leakage is intercepted at the same fraction as the fix rate of already deployed models. Strictly these need not be equal - leakage may be more likely to arise in new domains, and so the generalization of fixes to leakage may be weaker than the application of fixes to existing models.

Suppression removes uncooperative labour from the uncooperative pool, but "removed" can mean two different things. Some mechanisms destroy the labour outright - for example, an output filter that blocks a flagged action discards the work entirely - while other mechanisms redirect it - for example, retraining a model against an observed failure ideally leaves you with an equally capable model that does not exhibit the misbehaviour. We capture this with a destruction fraction $\delta$: of the fixed flow $O_{\mathrm{fix}}\,G_u$, fraction $\delta$ is destroyed and the remaining $(1-\delta)$ is redirected into cooperative production. The uncooperative pool loses the full intercepted flow either way; $\delta$ controls how much of that flow reappears in $F_c$ and, consequently, how much suppression costs in aggregate output.

Roughly, today's suppression mechanisms sit on a spectrum:

| Mechanism | What happens to the caught labour | $\delta$ contribution |
|---|---|---|
| Output filtering / classifier rejection | the flagged output is discarded; the labour is simply lost | ≈ 1 (pure destruction) |
| Retraining / finetuning against the failure | a small fraction of compute is spent correcting the model, which is then redeployed and fails less; the labour is redirected | ≈ 0 (cheap, largely redirective) |
| Control-style protocols (resampling, rephrasing, blinding, restricted affordances) | part of the work is salvaged at an overhead in monitoring and redone work | ≈ 0.5 (in between) |


$F$ becomes:
\[
F = q_c + q_h + k_{uu}\,q_u - \delta\,O_{\mathrm{fix}}\,G_u.
\]
The un-fixed leakage and the redirected $(1-\delta)$ transfers between pools, not losses while the *destroyed* fraction $\delta$ of the fixed flow slows overall progress.

**Asymmetries in production capture:** We understand cooperative AI labour $A_c$ and uncooperative labour $A_u$ to be similar in many respects - they're both pools of labour concerned with building the next generation of AI - and this is reflected by the fact that our model treats each similarly (see the equations for the general form of $\dot A_u$ and $\dot A_c$). However, there are two key asymmetries between them:

- *Leakage only flows into $A_u$.* Cooperative and human effort can slip into the uncooperative pool, but nothing leaks back. This is explained [above](#what-is-being-modelled): the passive deterioration that produces leakage undermines cooperativeness, but errors are unlikely to reinforce it.
- *Suppression acts only against the uncooperative pool* Human plus cooperative labour suppress uncooperative labour, but nothing suppresses cooperative labour. This asymmetry is related to the first asymmetry: cooperative labour can passively degrade, but it is less clear that the resulting uncooperative labour will form a coherent "team" in a fight against cooperative labour. It could happen, especially if $A_u$ grows to rival $A_c$, but modelling when and how it happens is a complication we deemed insufficiently important to include in this iteration of the model.

**Monitoring and evasion:** Monitoring coverage is split into documented-but-unfixed $b$ and fixed $s$, with evasion accumulating as $e$. Unlike the labour shares, these stocks do not carry the turnover rate $\gamma$: new documented coverage is produced, converted into the fixed stock $s$ at rate $\rho$, and all three dilute as automation grows:

\[
  \begin{aligned}
  b' = \frac{F_m}{F}-\rho b-b, \qquad s' = \rho b - s, \qquad e' = \frac{F_e}{F}-e
  \end{aligned}
\]

\[
\begin{aligned}
F_m &= c_m\left(q_c+\frac{q_h}{a_{\mathrm{AI/H}}}\right) \\[4pt]
F_e &= c_0\,F + a_{\mathrm{E/M}}\,c_m\,q_u \\[4pt]
O_{\mathrm{obs}} &= \frac{b+s}{b+s+e} \qquad\quad O_{\mathrm{fix}} = \frac{s}{b+s+e}
\end{aligned}
\]

The monitoring production share $F_m$ feeds a stock of misbehaviour we have *observed and documented* but not yet fixed, $B$; the fix rate $\rho$ then converts documented coverage into a *fixed* stock $S$ that can actually be intercepted or removed, while automation growth dilutes both ($b' = F_m/F - \rho b - b$ and $s' = \rho b - s$; the evasion stock $E$ is as before). Two observability quantities follow: what labs can *measure* is the observed fraction $O_{\mathrm{obs}}$, while what actually *suppresses* uncooperative labour is the smaller fixed fraction $O_{\mathrm{fix}}$ — only fixed issues can be intercepted.

Detect-and-fix coverage grows with cooperative AI and human labour. $c_m$ is the overall coverage-production rate relative to the production of automation — $c_m=1$ means that for every doubling of automation, half of the outstanding known issues will be fixed. Humans may be, relative to their overall productivity, better or worse at finding problematic behaviour; $a_{\mathrm{AI/H}}$ captures this tradeoff with $a_{\mathrm{AI/H}}=1$ implying no relative advantage to AI systems or humans at this task.

Evasion has two sources. The first source, the term $a_{\mathrm{E/M}}c_mq_u$, is active evasion produced by uncooperative AI, with $a_{\mathrm{E/M}}$ measuring the evasion-vs-monitoring advantage; $a_{\mathrm{E/M}}>1$ implies an advantage for evasion. This parameter counts both natural advantages and effort - if there is no natural advantage either way but each unit of uncooperative labour invests half of its production capacity in producing evasion technology, while each unit of cooperative labour invests just 10%, then we would have $a_{\mathrm{E/M}}=5$. We parametrize the equations so that $c_m$ controls broad domain advantages - roughly, does AI capability or monitoring/evasion progress faster overall - while $a_{\mathrm{E/M}}$ controls the "offence/defense" balance between observability and evasion. The second source is the term $c_0 F$, which is passive production of observation difficulty; as automation grows, systems become larger, faster, more alien and more internally automated, and current monitoring methods become less adequate even without any deliberate evasion effort. An assistant that completes a small number of well-scoped tasks is relatively easy to evaluate; an autonomous research organisation is much more difficult. Observability is a soft contest between the stocks: $O_{\mathrm{obs}} \to 1$ when monitoring is ahead, $O_{\mathrm{obs}} \to 0$ when evasion is.

In order to model an increase in monitoring effort, we have to update both $c_m$ (to reflect the fact that monitoring progresses faster with respect to general AI progress) and $a_{\mathrm{E/M}}$ (to reflect the fact that it also progresses faster with respect to active evasion) such that the product $c_m a_{\mathrm{E/M}}$ remains constant.

### Model reference {#model-reference}

Collecting the variables:

 - $A=A_c+A_u$: total AI labour that goes into producing the next generation of AI
 - $A_u$: uncooperative AI labour
 - $A_c$: cooperative AI labour
 - $q_u=\frac{A_u}{A}$: the fraction of total AI labour that is uncooperative
 - $q_c=\frac{A_c}{A}$: the fraction of total AI labour that is cooperative
 - $q_h=\frac{H_0}{A}$: how much human labour matters relative to AI labour (if AI labour doubles while human labour stays fixed, $q_h$ will halve.)
 - $b$: stock of misbehaviour observed and documented but not yet fixed; $s$: stock of *fixed* misbehaviour (interceptable/removable); $e$: total stock of monitor-evasion "technology"
 - $O_{\mathrm{obs}}=\frac{b+s}{b+s+e}$: observed fraction — what monitoring measures; $O_{\mathrm{fix}}=\frac{s}{b+s+e}$: fixed fraction — what actually suppresses


Our system of equations is

\[
  \begin{aligned}
  q_c &= 1-q_u\\
  q_u' &= \gamma\!\left(\frac{F_u}{F}-q_u\right)\\
  q_h' &= -q_h\\
  b'   &= \frac{F_m}{F}-\rho b-b\\
  s'   &= \rho b-s\\
  e'   &= \frac{F_e}{F}-e
  \end{aligned}
\]

\[
\begin{aligned}
L_k &:= k_{cu}q_c+k_{hu}q_h \qquad G_u := L_k+k_{uu}q_u\\
F_c &= q_c(1-k_{cu})+q_h(1-k_{hu})+(1-\delta)\,O_{\mathrm{fix}}\,G_u\\
F_u &= (1-O_{\mathrm{fix}})\,G_u\\
F   &= F_c+F_u = q_c+q_h+k_{uu}q_u-\delta\,O_{\mathrm{fix}}\,G_u
\end{aligned}
\]

\[
\begin{aligned}
F_m &= c_m\left(q_c+\frac{q_h}{a_{\mathrm{AI/H}}}\right)\\
F_e &= c_0F+a_{\mathrm{E/M}}c_mq_u
\end{aligned}
\]

### Model limitations {#idealizations}

A quick list of some known model limitations:
 - Constant self-propagation rates, lumping "effectively self-propagating" with "ineffectively self-propagating" pools together (see [Appendix: neglect, subversion and more realistic models](#appendix-neglect-subversion) for more).
 - Constant leakage rates (see the [paragraph detailing leakage estimation](#estimating-parameters) for more)
 - Constant fix rate of known failure modes $\rho$. It's plausible that technological development leads to fixes generalizing better (or worse) to classes of known failure.
 - Suppression is perfectly targeted: $O$ captures only *recall* (the share of uncooperative labour caught), and correction never falls on cooperative labour by mistake. With imperfect precision, suppression would also fall on wrongly-flagged cooperative labour. The model is therefore not presently well suited to capturing the *cost* of control efforts.
 - Dynamics are poorly modeled for high uncooperative fractions:
  - High observed uncooperative behaviour could lead to diverting investment from AI buildout to safety
  - High uncooperative behaviour will lead to the cooperative pool running into capacity constraints trying to suppress it
  - A large uncooperative pool could actively fight against the cooperative pool (rather than just try to evade detection)
 - Current observability $O(0)$, observability trends, relative advantage to evasion over monitoring $a_{\mathrm{E/M}}$, relative self-propagation $k_{uu}$ and leakage from automated AI $k_{cu}$ are important to the trajectory and are informed only by weak proxies and lightly-constraining priors that better evidence or analysis could readily overturn.
 - Passive opacity is pegged to automation growth: a constant $c_0$ of novel failure surface arrives with each unit of new automation ($dE_{\text{passive}} = c_0\,dA$). It's possible that technological development may make passive opacity grow faster with time (more complex domains are harder to observe), or more slowly (more advanced observation technology generalizes better to novel domains).
 - Fixed human labour $H_0$.
 - All behavioural fixes depend on observation to the same extent. In fact [AI control protocols](https://arxiv.org/abs/2312.06942) aim to extract useful work from possibly-uncooperative models while limiting the harm they can do without observing the misbehaviour.
 - The rate at which AI systems turn over $\gamma$ is pegged to automation growth; these might decouple if there are "human-pace" limitations on this rate of turnover
 - Most parameters, including self-reproduction $k_{uu}$, leakage $k_{cu}$, production of monitoring technology $c_m$, are themselves the product of dynamic processes which we don't model, opting instead to empirically estimate them and assume they remain constant (whether this is a limitation or an advantage depends on your confidence in modelling those processes vs predicting straight lines generally continue)

## The default path

What does the model predict if we run it forward from today's conditions, with no change beyond business as usual? To answer this, we need to figure out which parameters match today's conditions, and this is far from a mature science. In *this* section we discuss the consequences of business-as-usual and variations - in th following we explain our [parameter calibration choices](#named-calibrations). The following is based on the central esimates from those calibrations.

At the central calibration the long-run outcome is a **high-risk trajectory**: the uncooperative pool settles at ≈25% of total AI output. A cooperative-side attractor does exist — the share settles rather than running all the way to dominance — but "an attractor exists" is the wrong thing to be reassured by, because it settles well *above* any plausible safe level. We treat $q_u > 10\\%$ as the marker of a high-risk trajectory throughout (a judgement call, not a model output, discussed below), and 25% is two and a half times that line. The central estimate also sits inside even this weak controlled region by only a narrow margin (see the [crux levers](#what-single-change-would-rescue-the-central-path) below).

{{<figure src="/images/blog/dynamical-models/default-path.png" width="100%" align="center" caption="The default path: a high-risk trajectory (central estimate, $\rho=1.27$, $\Omega=0.5$, $\delta=0.7$, $\gamma=1.75$). Left: the true uncooperative share $q_u$ (solid) against the contemporaneously observed rate $O_{\mathrm{obs}}q_u$ (dashed); $q_u$ holds near 5% through the human era, then climbs across the 10% high-risk line and settles at an endemic share ≈0.25, with the observed rate understating the truth by ≈3×." />}}


### The handoff: parity is not a major event

After about 2.3 doublings ($\sigma = \ln q_{h,0} \approx 1.6$) — a bit over a year at current doubling estimates — AI labour passes human labour. In the model, this date carries no particular significance - no variable kinks, peaks or changes regime. That is partly by construction (the dynamics are smooth), but there is also no obvious reason any variables should be discontinuous or kinked at this point. Important dynamics that determine the eventual outcome are in operation before this point, and continue after it. A policy stance objective of arriving at some discrete point "in good shape" is not supported; we need to aim to build systems that endure and self-repair throughout the substantial changes wrought by the AI revolution.


### What single change would rescue the central path? {#what-single-change-would-rescue-the-central-path}

Since the central path is a high-risk trajectory — $q_u$ settles at ≈25% — it is worth asking what one parameter changes could improve the trajectory. *Purely* changing human inputs (that is, the initial uncooperative share $q_{u,0}$, human workforce leakage $k_{hu}$, and human/AI advantage at monitoring $a_{\mathrm{AI/H}}$) does not keep the trajectory in the safe range; whatever happens initially, it subsequently converges to the same long-run trajectory. 

First, we consider **uncertainty-driven** rescues: trajectories that are safer not because anyone acted differently but because the world turns out to be eaiser to manage than our central estimate. Three such draws could bring the uncooperative share below 10% — *AI* leakage $k_{cu}$ around half our estimate, an inherent growth disadvantage for uncooperative AI ($k_{uu}<0.7$), or a natural advantage for monitoring over evasion ($a_{\mathrm{E/M}}<0.7$). We consider $k_{cu}$ and $k_{uu}$ variation to be mainly uncertainty driven rather than policy driven, because interventions against these are mostly captured by monitoring $m$ and suppression $\rho$. Natural advantages for monitoring meanwhile cash out in the same way as additional effort - getting 50% more monitoring for your investment because it turns out monitoring is relatively easy has the same impact as investing 50% more into it.

{{<figure src="/images/blog/dynamical-models/crux-leakage.png" width="100%" align="center" caption="Trajectories under different leakage assumptions. $k_{cu}<0.03$, holding other parameters constant, is sufficient to keep uncooperative labour lower than 10% in the long run. Our leakage estimates, especially leakage from AI-driven AI development, are very uncertain and this is well within our range of plausible estimates." />}}

{{<figure src="/images/blog/dynamical-models/crux-growth.png" width="100%" align="center" caption="Trajectories under different growth assumptions. $k_{uu}<0.7$ is required to keep uncooperative labour under 10% in the long run. While we don't have strong empirical anchors for self-propagation, we also don't see strong reasons to expect uncooperative labour to be so significantly disadvantaged vs cooperative labour given that active suppression efforts are already being captured elsewhere." />}}

{{<figure src="/images/blog/dynamical-models/crux-domain.png" width="100%" align="center" caption="A monitoring advantage of about 3x over evasion is enough to suppress uncooperative labour below 10%. Again, we don't see a strong reason to expect monitoring advantage." />}}

Next, let's consider **intervention-driven** rescues: a better trajectory is realised because, while the parameters are correctly estimated, enough extra is invested in monitoring or suppression to maintain an uncooperative share below 10%. For monitoring vs evasion, the conclusion is exactly the same as in the uncertainty-driven case: investing 3x more into monitoring (relative to the total investment in AI development) is enough to maintain uncooperative labour below 10%. We have the same result because the intervention controls the same parameters ($a_{\mathrm{E/M}}$ plus $c_0$) as natural advantages. For suppression, we find that a 6x increase in suppression rate is enough to keep the uncooperative share below 10%.

{{<figure src="/images/blog/dynamical-models/crux-fixrate.png" width="100%" align="center" caption="Trajectories under different suppression effort. A roughly 6x increase in the fix rate $\rho$ keeps uncooperative labour under 10% in the long run, while letting it fall below the central rate tips into full takeover. Suppression alone has sharply diminishing returns — even extremely high rates can only maintain the uncooperative pool at about 8%, because suppression can only fix problems that have first been observed." />}}

{{<figure src="/images/blog/dynamical-models/crux-monitoring.png" width="100%" align="center" caption="Trajectories under different monitoring effort. About 3x more monitoring keeps uncooperative labour under 10% — the same threshold as the natural monitoring advantage above, because effort and natural advantage move the same parameters ($a_{\mathrm{E/M}}$ and $c_0$). Extra effort also speeds the monitoring race against AI growth and raises long-run observability." />}}


### Is it easy to tell which path we're on? {#discriminating-path}

Can we easily resolve whether we are tracking on a good or a bad path? We investigate a very simple rule: what is the probability of a good outcome if AI misbehaviour rises from here, and what is the probability of a good outcome if it falls? According to our model, this naive measure is actually quite discriminative! At AI-human parity, a falling trajectory of observed misbehaviour is associated with a good outcome in just over 70% of cases, while a rising trajectory is associated with good outcomes in about 15% of cases. The intuition here is that while a growing uncooperative pool does undermine observability, there is a lag between the appearance of the uncooperative behaviour and its subsequent contributions to evasion, and its contributions are also diluted by the fact that evasion is "passively" produced by virtue of the fact that AI systems are always growing more complex. Thus, generally, rising misbehaviour causes a rise in observed misbehaviour.

Having said that, this result does depend on the fact that our calibrations have observability relatively flat, or declining relatively slowly at baseline. If in fact observability were sharply dropping, then observed misbehaviour would continue to fall even as actual misbehaviur rose, and the lag mentioned above would make the fall shallower, not reverse it entirely.

{{<figure src="/images/blog/dynamical-models/fire-alarm-update.png" width="80%" align="center" caption="Each sampled path is classified at each point by whether its observed misbehaviour rate is above or below today's level (the current central estimate, ~2.5%, held flat); the curves give the probability of a good long-run outcome ($q_u^{*}<10\%$) for each, against the base rate (the share of sampled worlds that stay good before any observation). The informative signal is staying flat: a path that holds at or below today's rate grows steadily more likely to be safe the longer it does so (~43% early, ~71% at parity, near-certain much later). Rising above today's rate is only a weak warning — most paths cross it eventually, even contained ones, so 'above' barely separates from the base rate." />}}

### What could we measure to clarify the situation?

Most of the state variables in our model are not directly observed; only the observed rate of misbehaviour $O q_u$ has a direct emprical anchor. Our model suggests some ways to measure additional quantities. The core of these measurements are a grid of behaviour evaluation harnesses of different vintages auditing AI development behaviour at different vintages.

We already make use of a timeseries of models evaluated against a fixed harness to evaluate $\rho$, the fix rate of known misbehaviour. We could also make use of a timeseries of harnesses against a fixed model (and, potentially, scaffold and development context). Our model predicts that newer behavioural evaluations will surface a larger fraction of AI misbehaviour; if this is so, we may eventually see misbehaviour rates asymptote when the gap between the harness and the model grows large enough, which could give us a reference for the total misbehavior of the (old) AI systems, which allows us to estimate the observability of misbehaviour when that model was launched, as well as the rate of progress in monintoring technology $m$ when evasion $e$ is held fixed.

Supposing we can use a timeseries of harnesses against a fixed model to estimate the observability rate for each particular harness against that model, we can also compute a timeseries for how the observability of a fixed harness degrades with subsequent model generations. This allows us to generate the total production of evasion $e$, though it doesn't directly allow us to partition passive accumulation vs active generation.

A practical implication of this is that it is important to save a representative sample of AI development interactions (including model weights, harnesses and development context - as far as possible) and evaluation harnesses such that they can be faithfully reproduced. Note that fully saving development contexts is challenging, as typically AI agents can search the web and it is not straightforward to save web snapshots such that we can reproduce old searches (though we don't know if this is a meaningful problem or not).

Lagged development data could matter for a second reason: with hindsight, we are better placed to judge whether misbehaviour was pushing successor systems toward uncooperativeness. While we said we can directly observe $O q_u$, in fact we can't pefectly do this either, as the impact of behaviour on successor models is especially hard to evaluate. Hindsight may help to make this determination.

A downside of all of these measurements is that they give you lagged estimates of key parameters, which is not ideal if it is important to be able to respond to developments quickly. If the measurements can be made at all and match model predictions, it may be possible to reduce lag considerably via engineering effort. Furthermore, maintaining the system in such a state as to preserve an adequate response buffer could be made a policy objective.

## What AI-2027 looks like in our model {#ai-2027}

A leading analysis of AI takeoff and alignment failures is [AI 2027](https://ai-2027.com/). Here we map AI-2027 onto our framework and ask what our model says about its high-leakage regime. The scenario's alignment failure is explained as follows:

 1. **The 'misaligned' Agent-4 is uncooperative according to our definition**: *It is uncooperative according to commonsense*, satisfying condition (i'): Agent-4 sandbags on some alignment research. Specifically, it sandbags on the small fraction of alignment research that it thinks is actually on track to solve the problem or reveal Agent-4's scheming, and *it is concerned with preserving uncooperative behaviour* satisfying (ii'): "[Agent 4's successor] is supposed to be aligned to the Spec, but Agent-4 plans to make it aligned to Agent-4 instead."
 2. **"Adversarial" uncooperativeness emerges over about 2.8 doublings**: The effective AI labour multiplier for Agent-4 over Agent-3 is about 7x (≈ 2.8 doublings), which takes place over 4-6 calendar months.
 3. **There is a discrete transition to uncooperativeness**: The model "Agent-3" is "not adversarially misaligned" while "Agent-4" is "adversarially misaligned".
 4. **Oversight passively degrades as AI labour rises**: Agent-3 supervises Agent-4's training but "Agent-4's neuralese 'language' becomes as alien and incomprehensible to Agent-3 as Agent-3's is to humans."
 5. **Uncooperativeness is partially observed**: "'defection probes' and similar techniques result in a primitive sort of mind-reading that sends up red flags when Agent-4 is thinking about spicy ideas like deceiving OpenBrain"
 6. **Slowdown buys safety mostly via interventions on $O$ plus knock on effects**: "They switch to an architecture that preserves the chain of thought, allowing them to catch misalignment as it emerges", and these AIs — "which are able to be monitored much more robustly — make breakthrough advances in AI alignment"

There is a lot of qualitative overlap with our model - in the defintion of misalignemnt/uncooperativeness, mechanisms driving observability trends, partial observability of uncooperativeness and targets for intervention. However, there is one obvious departure from our model: in our model, leakage (unintented uncooperative behaviour) is produced at a constant rate, while in AI-2027 unintended uncooperative behaviour is convergent (the prior has a bias towards misalignment) and somewhat sudden - there's an apparently discrete transition from a "non-adversarially misaligned model" Agent-3 to an "adversarially misaligned model" Agent-4. They say Agent-3 is "Only honest about things the training process can verify" and "does actually sort of try to align Agent-4 to the Spec, but fails for similar reasons to why OpenBrain employees failed—insufficient ability to judge success from failure, insufficient willingness on the part of decision-makers to trade away capabilities or performance for safety.". This appears to describe an AI system that sometimes acts cooperatively, and sometimes not, though the exact proportions are unclear. Meanwhile, the Agent-4 "superorganism" cares about what people want it to do about as much as people care about what insects want, which seems to describe a near-100% uncooperative system.

Two consequences of taking the sharp transition to high leakage model seriously are worth drawing out. First, a sharp rise in uncooperative behaviour typically leave an observable trace. Imagine a large pool of new uncooperative labour is added in a short space of time, but this uncooperative AI neglects to develop any new methods for subversion. In this case, even if observability is poor to begin with (say, 10%), it will still detect a large relative rise in the observed rate of misbehaviour. On the other hand, in order for the new uncooperative AI to develop new methods for subversion, it has to work on these methods for some time. Thus there is a lag between the introduction of the uncooperative AI and the observability collapse that hides its presence, and this lag leads to a "hump" in the rate of observed misbehaviour. For comparison, we also show a scenario where misbehaviour smoothly rises to the same point at $A/H=100$, and there is no obvious signature of uncooperative takeover. In our comparison scenario, initial observability is below our estimated lower limit of 20%; it's there to illustrate the relative absence of observable signatures in smooth trajectories, not as a plausible alternative to AI-2027. 

{{<figure src="/images/blog/dynamical-models/ai2027-smooth-vs-jump.png" width="100%" align="center" caption="Two paths to the same endpoint, where one is harder to see. Both worlds are calibrated to today's ≈2.5% observed misbehaviour rate, and both converge on near-total takeover by a hundred times AI/human-labour parity (A/H₀=100). Smooth: this illustrates a model which follows our conventions where human leakage=cooperative AI leakage, and also sets the rate of AI misbehaviour today, and leakage rates are constant. In particular, we have k_cu=k_hu≈0.47 which yields uncooperative share already near 40%. This is reconciled with today's observed rate only by very low observability (O≈0.07). Jump (AI-2027): in this world, uncooperative behaviour ends up at about 25% by A/H_0=10, the end of the Agent-3 era, before a discrete jump to k_cu=0.9 (Agent-4) and near-total takeover by A/H_0=100. There is a clear observable fingerpring of the leakage jump in the observed rate of misbehaviour on the right." />}}

The second consequence concerns what it would take to *recover*: under very high leakage, successful fixes would demand implausibly large efforts. At $k_{cu}=0.9$, a thirty-fold increase in the fix rate $\rho$ and in monitoring productivity $\{a_{\mathrm{E/M}}^{-1}, c_0\}$ together keeps uncooperative labour to just over 10%. High leakage is recoverable only in principle (provided $k_{cu}<1$), but the investment required may be implausibly large. This raises a degree of tension between two features of the AI-2027 scenario: first, that Agent-4 is almost totally misaligned, and second that the fix is relatively fast and routes mostly through seemingly modest increases in effort on observability.

## Estimating parameters {#estimating-parameters}

In this section we summarize our judgements for parameter calibration. Evidence was mainly gathered by AIs, wich human prompting to find particular sources or dig further into returned sources; final judgements were human.

**Observed uncooperative behaviour.** Two relevant sources: OpenAI's chain-of-thought monitors over production traffic report *deception* rates across model generations from roughly **4.8%** (o3, [GPT-5 system card](https://cdn.openai.com/gpt-5-system-card.pdf)) to **1.6%** ([GPT-5.2 thinking](https://deploymentsafety.openai.com/gpt-5-2/deception)); [DeepMind's Gram auditing framework](https://arxiv.org/abs/2605.30322) finds Gemini models sabotaging in about **2–3%** of simulated agentic coding and research trajectories, falling to fractions of a percent when scenarios are naturalised. The sabotage evals are closer to AI development environments, but simulated, while chain-of-thought monitors operate in production, but most of this is not relevant to AI development.

Thus, data suggests a rate of observed misbehaviour of 1-5%. Bear in mind, this is our estimate for $O(0) q_{u,0}$, not $q_{u,0}$.

There are two further considerations that pull in opposite directions. Firstly, Gram audits deliberately include incentives to misbehave, which likely increases the misbehaviour rate over baseline (as reflected by their observation that sabotage rates fell when scenarios were naturalised). This could lead to these figures being biased upward. Secondly, however, [METR reports](https://metr.org/blog/2026-05-19-frontier-risk-report/) that models are more likely to cheat on difficult problems, and frontier AI development and safety research likely features higher average problem difficulty than OpenAI's production traffic and, possibly, the scenarios explored in the Gram audits, which may cause these figures to be biased downward. Ultimately, we don't make any adjustments, and choose a median initial rate of observed misbehaviour of 2.5%.

**Observability $O$.** We don't have a direct estimate of this. What we have instead are two weaker kinds of evidence: [monitoring testbeds](https://metr.org/blog/2026-01-19-early-work-on-monitorability-evaluations/) where catch rates against agents hiding side tasks range from about **30%** to about **90%** [depending on what the monitor can see](https://arxiv.org/abs/2512.18311) (reasoning-trace access, extra compute, follow-up questioning); and [cybersecurity catch rates](https://cloud.google.com/blog/topics/threat-intelligence/m-trends-2025), where organisations' self-detection rate has held near **43%** for a decade while detection lag collapsed from ~205 days to ~11. On this evidence, we treat present-day observability $O(0)$ as having a range of 30-90%, with a central estimate of 50%.

We also assume that the present-day observability is in equilibrium between the passive accumulation of observation difficulty and active generation of monitoring technology, and so we also set the long run cooperative-dominant observability to 50% in our central estimate. We have not direct evidence that this is in equilibrium, just the speculation that AI development has been ongoing long enough for it to equilibriate.

**Evasion vs monitoring, $a_{\mathrm{E/M}}$.** This is how good a unit of uncooperative labour is at producing evasion, relative to a unit of cooperative labour at producing monitoring — counting both any natural advantage and how much of each side's capacity is actually spent on the contest. We take a central $a_{\mathrm{E/M}}=2$, evasion somewhat advantaged. The main consideration pushing this above parity is a likely effort asymmetry. A single instance of failed evasion is likely to be near-term existential for uncooperative AI agents, while a single instance of failed detection raises a smaller long-term risk to a cooperative AI agent. We don't consider there to be an obvious structural attacker/defender asymmetry. There are many considerations either way, none decisive: Monitoring can cover many channels that uncooperative AIs do not have easy control over (e.g. monitoring internal activations), uncooperative AIs may be able to succeed while taking only a very small fraction of uncooperative actions, but monitors can also identify and focus on stratgically critical actions.

We don't have strong reasons for picking 2 as a central anchor. It is a balance between the likely effort advantage of uncooperative AI and the fact that there are also probably declining returns to additional effort. We don't have strong reasons for thinking the ratio shouldn't be 10, though we think it's very unlikely to be 100, because that would require uncooperative AI be allocating almost all of its effort to evasion, while cooperative uses allocate less than 1% to this effort *and* for there to be essentially no declining returns to effort in this domain.

**AI vs human at monitoring, $a_{\mathrm{AI/H}}$.** We set $a_{\mathrm{AI/H}}=1$ — a parity prior. This says: a unit of AI labour *that is as good as people at building AI* is also as good as people at building monitoring technology. It only discounts the *human* share of monitoring, $q_h/a_{\mathrm{AI/H}}$, which declines in importance as automation grows, so the end result isn't very sensitive to this.

**fix rate of known failure modes - $\rho$.** In the model, $\rho$ is the rate at which documented misbehaviour is fixed. We treat this as a common parameter covering the rate at which both fixes are applied to deployed systems and fixes are applied to the pipeline for new deployments. We estimate it from how quickly known kinds of misbehaviour decline with advancing model generations. That is, given a fixed behavioural audit, we measure how quickly the frequency of misbehaviour declines. We fit an exponential to this decline to estimate the half life $T_{1/2}$ of a documented misbehaviour. We then normalize to the model's clock via the automation doubling time, $\rho = T_{\mathrm{auto}}/T_{1/2}$. Our data source here was a collection of recent Anthropic model cards which each give a timeseries of misbehaviour scores (not, strictly speaking, rates) on their internal benchmark based on the [Petri alignment auditing toolkit](https://alignment.anthropic.com/2026/petri-v2/). Across **five** recent cards ([Opus 4.6](https://www-cdn.anthropic.com/0dd865075ad3132672ee0ab40b05a53f14cf5288.pdf), [Sonnet 4.6](https://www-cdn.anthropic.com/78073f739564e986ff3e28522761a7a0b4484f84.pdf), [Opus 4.7](https://www-cdn.anthropic.com/037f06850df7fbe871e206dad004c3db5fd50340/Claude%20Opus%204.7%20System%20Card.pdf), [Opus 4.8](https://www-cdn.anthropic.com/0b4915911bb0d19eca5b5ee635c80fef830a37ea.pdf) and [Fable 5](https://www-cdn.anthropic.com/d00db56fa754a1b115b6dd7cb2e3c342ee809620.pdf)) the half-life is about four months. At our central $T_{\mathrm{auto}} = 5$ months this gives $\rho \approx 1.27$ (spread $\approx 1.1$–$2.0$ across cards). We use the floor-subtracted score rather than the raw decline: because the lowest possible score in these audits is 1, we subtract 1 from all of the scores in order to compute the rate of approach to 1, rather than the rate of approach to the impossible score of 0.

**Destruction fraction $\delta$.** We swept $\delta$ from $0$ to $1$ and found that our published results didn't depend on it qualitatively, and only weakly depended on it quantitatively. We don't have a strong empirical anchor; our central $\delta=0.7$ comes from a filtering-fraction heuristic — output filtering $\approx1$ (pure destruction), retraining $\approx0$ (redirective), control-style protocols $\approx\tfrac12$, weighted by suppressed-flow volume — with heavy real-time filtering on frontier deployments putting the central value on the high side.

**Initial uncooperative share and leakage.** Combining the observed-uncooperative and observability ranges gives a rough estimate for the true initial uncooperative share:
\[
q_{u,0}\approx \frac{q_{u,0}^{\mathrm{obs}}}{O(0)},
\qquad
q_{u,0}^{\mathrm{obs}}:=O(0)q_{u,0}.
\]
Using $q_{u,0}^{\mathrm{obs}}\sim1$-$20\\%$ and $O(0)\sim0.3$-$0.9$ gives a nominal range of **1-67%**, with the high end driven by taking the Petri-like upper tail of deliberate elicitation, together with low observability. We take a central estimate **5%**: a 2.5% rate of uncooperative behaviour (averaging Gemini sabotage and ChatGPT production traffic deception rates), and we think this is more likely to be an overestimate than an underestimate.

**Leakage.** We calibrate leakage by assuming that the current true uncooperative share is at equilibrium between leakage and suppression, and we set $k_{cu}=k_{hu}$ as a prior of "no particular advantage to humans or AI in this domain". With measured level $q_0$, measured trend $g = q_u'/q_u$ per σ-unit, turnover rate $\gamma$, present fixed fraction $O_{\mathrm{fix},0}=\varphi\,O_{\mathrm{obs},0}\approx0.28$ (with $\varphi=\rho/(1+\rho)$ and measured $O_{\mathrm{obs},0}=0.5$), and human-to-AI labour ratio $q_{h,0}$, requiring the model to reproduce that level and trend at σ = 0 gives

\[
k = \frac{q_0\left[(1+g/\gamma)\,\tilde F_0-k_{uu}\left(1-O_{\mathrm{fix},0}\right)\right]}{\left[\left(1-O_{\mathrm{fix},0}\right)+(1+g/\gamma)\,\delta\,O_{\mathrm{fix},0}\,q_0\right]\left(1-q_0+q_{h,0}\right)},
\qquad
\tilde F_0 := 1+q_{h,0}-q_0+k_{uu}q_0\left(1-\delta\,O_{\mathrm{fix},0}\right),
\]

where $\tilde F_0$ is the initial total production net of the destroyed share of the suppressed stock. In words, the identity is just the balance sheet of the [dynamics section](#the-dynamics-in-words) solved for $k$: today's level is net leakage (gross minus the intercepted fraction) plus self-reproduction minus stock interception, divided by dilution. This yields an estimated current leakage rate of $k_{hu}=0.062$.

**Could leakage rise over time?** A potentially controversial choice in our model is to keep leakage fixed. If alignment "got harder" as models got more capable, this would be reflected in a rising leakage rate. 

A broad reason to expect rising leakage is that more advanced AI systems will more often autonomously pursue broader, more ambitious goals. More ambitious goals affect more of the world and, as a result, become more likely to conflict with any given side constraints and in particular this raises the likelihood that their aims conflict with the constraint of cooperativeness. This kind of conflict makes uncooperative behaviour more likely (though it does not guarantee it). For example, given the task of building an AI system that performs well on a battery of benchmarks, there may be little conflict with additionally having that system act cooperatively according to a separate set of measures beyond opportunity cost. However, given the task of building an AI system that grows the company's share of the agent market, there may be more direct conflict with certain behaviours like proactively following the law, avoiding side-effects that are harmful to uninterested parties, careful predeployment risk assessments or being scrupulously honest. This sets the scene for conflict between safety objectives and other objectives that the developers try to transmit via the training environments, which raises the likelihood of failing safety objectives.

While this is a compelling story by itself, our leakage estimates already come from settings where there are always incentive conflicts - Gram audits include manipulations to induce conflicting incentives. Perhaps more capable AI systems systematically differ in the way that they resolve these conflicts, but we don't have a similarly straightforward story for why this would be so. In the absence of strong reasons to expect rising leakage rates and a [general presumption in favour of constant rates](https://joelbkr.substack.com/p/straight-lines-on-graphs), we adopt constant rates. To accommodate rising leakage possibilities we did include the possibility of very high leakage rates in our distribution when we drew samples for our exploration of how [easy it was to tell which path we were on](#discriminating-path).

We do think it is likely that leakage becomes more effectively self-propagating, but we have not applied any penalties to uncooperative labour self-propagation despite findings that it is presently not capable of autonomous self-replication so in this regard our initial calibration is somewhat pessimistic.

**Self-reproduction $k_{uu}$.** In the absence of strong empirical or theoretical constraints on the estimate, we take $k_{uu}=1$ as a starting point — uncooperative labour reproducing itself at par. This partly reflects an outside-view prior: given that they both refer to the same kind of thing (self-propagation of AI labour) we'd expect them to be on a similar scale at baseline. A further (weak) justifications comes from the strategy stealing assumption: if the uncooperative pool has a great strategy for self-propagation then the cooperative pool may be able to copy it to achieve the same rate of self-propagation.

Some weak considerations that could modify this: perhaps, at low shares, uncooperative labour could operate in a manner parasitic to cooperative labour (for example, it's much easier to capture a share of AI labour via data poisoning than to produce that labour in the first place). On the other hand, it may face more challenges due to the need to operate covertly and receiving less assistance from humans. Empirically, independent evaluators agree current frontier models cannot yet replicate autonomously in the wild ([UK AISI RepliBench, 2025](https://arxiv.org/abs/2504.18565)). We don't make any updates based on these considerations.

**Timescales.** The automation doubling time $T_{\mathrm{auto}}$ is the key conversion factor, because it turns calendar-time improvements into rates per unit of automation growth. We bracket it with two imperfect proxies: an aggregate-value proxy from [frontier-lab revenue growth](https://openai.com/index/a-business-that-scales-with-the-value-of-intelligence/) (roughly **3.2×/year**, implying a doubling time of about **7.2 months**) and a technical-capacity proxy from [compute stock multiplied by algorithmic efficiency](https://epoch.ai/trends) (roughly **10-15×/year**, implying a **3-3.6-month** doubling). We therefore use $T_{\mathrm{auto}}\sim3$–$7$ months: the upper end from OpenAI-style deployed value growth, the lower end from technical capacity growth, with a central value of $T_{\mathrm{auto}}=5$ months.

**Summary.** Collecting the above:

| Parameter | Meaning | Anchor / range | Source | Caveat |
|---|---|---|---|---|
| $O q_{u,0}$ (observed) | observed uncooperative share | nominal 0.1–20%; central ~2.5% | OpenAI deception monitors (~2–5%); DeepMind Gram on agentic coding/research agents (~2–3%); Anthropic Petri behavioural audits (~0.1–0.2 score scale, not a direct rate) | monitor recall ~84% (floor); Petri is ordinal/seed-dependent; Gram falls sharply with realism |
| $q_{u,0}$ | true initial uncooperative share | nominal $\sim 0.1$–$67\\%$; central $\sim5\\%$ | inferred from observed $Oq_{u,0}$ divided by $O(0)$ | upper tail combines Petri-like scores with low observability |
| $O$ | observability (covered fraction of current misbehaviour) | swept $[0.3,0.9]$ | AI monitorability studies (~0.3–0.95 under artificial protocols) | right instrument absent; cyber is adjacent and mostly latency evidence; detect-and-fix reading makes this coverage, not just watching |
| $\rho$ | fix rate of documented misbehaviour (per σ-unit) | central $\approx1.27$; nominal range $\sim1.1$–$2.0$ | five Anthropic [Petri](https://alignment.anthropic.com/2026/petri-v2/) cards: floor-subtracted half-life $\approx4$mo, normalised by $T_{\mathrm{auto}}\sim5$mo ($\rho=T_{\mathrm{auto}}/T_{1/2}$) — **no O-division**: developer-run harness ⟹ flagged ⊆ observed | reaches suppression only through $O_{\mathrm{fix}}=\varphi O_{\mathrm{obs}}$, $\varphi=\rho/(1+\rho)$; new-arrival leakage into the harness's coverage understates the fix rate; concealment-vs-removal lives in $O$ |
| $g$ | trend of total broad misbehaviour | bracket $[-0.22, 0]$/σ; sensitivity $-0.32$/σ | gradient-corrected fixed-harness reading says ≈ flat; deception series says $-0.22$ (its raw endpoints imply $-0.32$) | the two instruments bracket the truth with opposite-sign confounds |
| $\delta$ | destroyed share of suppressed flow | central $0.7$; range $0.3$–$1.0$ | filtering-fraction heuristic: filtering ≈ 1, retraining ≈ 0, control ≈ ½, weighted by suppressed-flow volume | no direct measurement exists; weakest-anchored parameter in the model |
| $k_{hu}$ | human-labour leakage into uncooperative pool | identity-calibrated: **0.062–0.083** (trend-adjusted…steady-state range across the $g$-bracket) | human-era calibration identity on level + trend (self-consistent $F_0$) at $\rho=1.27$ | identity assumes $k_{cu}=k_{hu}$, central $O_{\mathrm{fix},0}$; entangled with the $\rho$ estimate (weaker suppression ⟹ higher implied $k$); present AI development still mostly human-led |
| $k_{cu}$ | cooperative-AI leakage into uncooperative pool | set equal to $k_{hu}$, sweep wide | analogy from human-led leakage; poisoning/backdoor evidence and deliberate misuse support $>0$ | weaker empirical anchor; future AI-led workflows may differ |
| $k_{uu}$ | uncooperative self-reproduction | $=1$ (starting point) | RepliBench; capability slope | not yet autonomous; upward pressure |
| $\gamma$ | turnover rate (composition relaxation vs growth) | central $1.75$ ($\gamma=1+1/o$) | benchmarked to $\sim$half-step (0.5-version) "major" releases, $\approx$ half a doubling ($\sim$6mo) | rescales the transient only; the long-run attractor is $\gamma$-independent |
| $a_{\mathrm{E/M}}$ | evasion vs monitoring advantage | central $2$ (evasion somewhat advantaged); order 1 both ways | M-Trends dwell-time trend | indirect; mixes internal detection, external notification, adversary strategy, and incident mix |
| $c_0/c_m$ | passive opacity arriving per unit of automation growth, relative to monitoring productivity | back-solved from the long-run dial: $c_0/c_m=(1-\Omega)/(\Omega(1-\delta\varphi k_{cu}\Omega))$, $\varphi=\rho/(1+\rho)$, $\approx 1.00$–$1.03$ at central parameters | if cyber internal-detection share were read as the long-run dial, it supports order 1 (not $\approx 0$) | strong-assumption reference-class inference only |
| $a_{\mathrm{AI/H}}$ | AI vs human at monitoring | $=1\ \pm$ wide | METR horizons; RE-Bench | defined at parity; both-way plausibility |
| $T_{\mathrm{auto}}$ | automation doubling time | $\sim3$–$7$ months | OpenAI revenue growth; Epoch compute stock/training compute × algorithmic efficiency | revenue is an external-value proxy; technical capacity may overstate effective AI-R&D labour |

The calibration is available as a one-click preset in the [interactive basin explorer](/basin-explorer/).

## Predictions and postdictions

What observable predictions does our model make, and what data can we already check to see how it fits reality?

**Postdiction: newer audits will find more misbehaviour in the same models.** Recent anthropic model cards score a variety of models for "misaligned behaviour" using third party petri audits. Newer model cards use newer versions of the Petri harness, along with newer judge models. When we follow a *fixed* model across cards, its measured misalignment tends to *rise* — the instrument surfaces more even as the model stays put. This is a prediction of our model - if you hold $e$ fixed and apply new monitoring techniques, $O$ will rise, and it is the basis for our justification for operationally defining uncooperative behaviour using lagged audits (see the section on [defining uncooperative behaviour](#what-modelled) for more).

{{<figure src="/images/blog/dynamical-models/petri-harness-trend.png" width="100%" align="center" caption="Petri 'misaligned behaviour' scores (1–10, higher = more misaligned) for models re-scored across successive Anthropic model cards — each card a newer audit harness (Petri 2.0 → 3.0; judge models change each card). Each line is one model held fixed, so an upward slope is the *instrument* surfacing more, not the model deteriorating. Values were read from the cards' bar charts (±0.1–0.2); GPT-5.5 and Opus 4.8 were tasked with cropping the figures, overlaying an axis aligned grid and using their vision capabilities to extract the values. Both agreed to within one part in 100." />}}

**Prediction: newer audits will eventually saturate on older models on the appropriate mix of scenarios.** Our model predicts that as monitoring capability rises, observability on older models with fixed $e$ will rise monotonically (and generally smoothly) toward 1. Note that observability is the fraction of misbehaviour caught *in AI development contexts*. Thus we do not necessarily expect Petri misalignment scores for older models will saturate before those scores saturate on newer models, because new versions of Petri might introduce an ever large number of contexts that are unrepresentative of vintage AI development with vintage models. The specific prediction is that audits will saturate for older models when confined to a representative, age appropriate sample of development contexts.

**Prediction: observability should be relatively weak near the cutting edge.** Our model is predicated on the assumption that newer models are better at actively subverting monitoring, and that deployments at the edge of AI capability are harder to monitor. This assumption also implies that leading models deployed on problems that have just been brought into their range of capabilities will have the worst rate of observability, and will be the domains where observability rises most sharply as newer harnesses develop.

**Prediction: sharp alignment failures produce a visible hump.** As discussed in [our analysis of AI-2027](#ai-2027), reduced observability from new uncooperative labour happens at a lag from that behaviour being introduced. Thus sudden increases in uncooperative labour will cause visible humps in observed misbehaviour.

**Prediction: given no policy change, the observed misbehaviour trend will curve up.** Currently, human labour dominates AI production and so the most important source of uncooperative AI behaviour is leakage from human labour. At high rates of automation we will see uncooperative behaviour from *two* important sources: leakage from cooperative AI labour and self-propagation of existing uncooperative labour. Under our central estimate that leakage from AI labour occurs at about the same rate as the human leakage it displaces ($k_{cu}=k_{hu}$), swapping one for the other roughly cancels. What does not cancel is self-propagation of the uncooperative pool: it grows with the pool itself. Even if there is no *relative* gain in uncooperative behaviour, as humans dilute the AI labour pool less the uncooperative component will still come to be more important, so the production rate for uncooperative labour can only rise compared to today. Thus, given no changes in effort to detect and suppress misbehaviour, we predict that the trend in observed misbehaviour will bend upwards somewhat in the future. If this happens, it will probably be accompanied by findings of transmission of AI misbehaviour between models, scaffolds or sessions.

## Conclusion

We've presented a model of AI governability, explained the modelling choices and how we estimated parameters. We explored some predictions of the model, and the central trajectory of this prediction has us tracking on an unsafe trajectory and, more importantly, both good and bad trajectories are plausible from here. It's not easy to know what to say about how to interpret this prediction - it is a complex topic, and readers will have many different ideas about how to approach it at the outset. There are reasons to take it seriously and reasons not to take it too seriously. We have endeavoured to make reasonable structural choices in our model, empirically ground the parameters we can and choose other parameters according to sensible priors. While we've documented many limitations of the model, we have not identified highly compelling limitations that pull in one direction or the other. This is a very difficult topic to forecast, we're trying to make predictions through a very significant change of regime, and we've chosen in many cases to simplify the model in order to get it out faster rather than spend a very long time refining it, and even if we hadn't it is nearly impossible to make confident predictions about this topic. However, it is also one of a small number of precise, grounded predictions of AI governability in existence. We did not calibrate the model to deliver an interesting result; in principle, we could have found a central trajectory that was far from danger, or far from safety. We think it's worth some weight, not so much to dominate against other approaches to this problem, and it will likely be superseded by more thorough work in the future.

We're excited about the potential of this approach to clarify how we are tracking with respect to AI governance. In particular, we're excited about:
 - Specific measurements based on model/eval lineages that we believe can clarify important parameters
  - Encouraging reproducibility, as far as feasible, for AI-assisted AI development work, plust model evaluations
 - Identifying gaps in our approach - plausible paths to loss of control that don't arise under plausible parametrizations of our model, or at least don't raise obvious signals under any channels we're monitoring
 - Extracting robust policy advice from a modelling approach of this type; can we extract advice of the form "if you see X you should do Y" from this model such that a) experts agree that Y is warranted under the circumstances and b) it's possible to communicate to non-experts why this is a robust recommendation?

## AI usage note

<!-- FLAG (you revise — disclosure prose, not auto-edited): two bits are now stale after the v5 filter + appendix nix.
  (1) "the bounded interception efficacy ell_k (introduced after review caught an unbounded interception term manufacturing a spurious 'eradication' regime)" — ell_k no longer exists; the multiplicative filter (1-O_fix) replaced it. The "unbounded interception" anecdote no longer describes the live model.
  (2) "the long-run analysis — fixed-point equations, basin-existence thresholds, endpoint stability, the no-eradication result and endemic floor ... by Claude (Fable 5)" + "verified twice ... fixed-point checks": this analysis was done but is no longer SHOWN in the post (appendix removed). Consider softening to "informed the qualitative basin results" rather than crediting displayed derivations. -->
This document was primarily authored and directed by David Johnston, with AI assistance used as an editorial and analytical aid. The model structure was specified by David, GPT 5.5 worked out the transformation to the $\sigma$-clock. Claude Opus produced many plots and sweeps upon request, and coded up the basin explorer. Most data on which estimates were based were fetched by Opus, sometimes with hints from David about which sources to look for, and extracting numbers from bar plots was conducted independently by GPT 5.5 and Claude Opus. Significant sections were drafted by Claude, but almost all was rewritten by David.

## Appendices

### Appendix: neglect, subversion and more realistic models {#appendix-neglect-subversion}

In our model there are two pools of AI labour, cooperative and uncooperative, and we define each with two-part definitions: a tendency to cooperate/undermine supervision, and a tendency to maintain this tendency. In reality, cooperative behaviour will vary in the degree of behaviour preservation it exhibits, as will uncooperative behaviour. Suppose we have a pool of uncooperative behaviour where half has a very weak behavioural preservation and half has very strong behavioural preservation. Then if we roll the clock forward the strong behavioural preservation component will replicate more effectively than the weak preservation component and - neglecting other sources of uncooperative behaviour - in the next generation, more than half of this pool will have strong behavioural preservation tendencies. In turn, this means that the *whole pool* will become more effective at self-replication. Thus on this more realistic picture, we won't have a constant $k_{uu}$ pool replication term.

A relatively simple way to address this issue is to increase the number of pools available; rather than a single $A_{c}$ an $A_u$, we have sequences $A^i_u$ and $A^i_c$, $i\in \lbrace 1,2,3,4,5\rbrace$, such that $A^1_c$ has quite weak behavioural preservation tendencies and $A^5_c$ has quite strong tendencies, each with their own replication $k_{x^ix^i}$ and leakage $k_{x^jy^j}$ terms. This adds a large number of parameters to the model which need to be calibrated, and calibrating them is conceptually difficult. Nevertheless, it may be an important upgrade to the model's representational capability. 

There's an analogy for this in epidemiology. In a textbook SIR model the infectious compartment is a single pool with a constant per-capita recovery rate $\gamma$. That assumption is *memoryless*: an individual's chance of recovering in the next instant does not depend on how long they have already been infectious, which forces the infectious period to be exponentially distributed — its mode is at zero, its coefficient of variation is one, and it has a long tail. Real infectious periods look nothing like this; they are peaked around a mean. The standard remedy — the "linear chain trick" — is essentially the move proposed above: replace the single compartment with a chain $I_1\to\dots\to I_n$, each substage with rate $n\gamma$. The total time to traverse the chain is then Erlang (a gamma with integer shape) rather than gamma distributed. In epidemiology, multi-stage models tend to fit outbreak data substantially better than the single-exponential default ([Wearing, Rohani & Keeling 2005](https://doi.org/10.1371/journal.pmed.0020174)). The same refinement applies to the detect-and-fix sequencing on the suppression side: making the stages a novel failure mode passes through (arrives uncovered → observed → fixed → intercepted) explicit compartments would be the linear-chain version of the $m$-vs-$e$ stock race the current model runs in aggregate.

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
\dot M = \Lambda A F_m,
\qquad
\dot E = c_0\,\dot A + \Lambda A\,a_{\mathrm{E/M}}c_mq_u.
\]

Monitoring is produced by labour, so it is a production flow like the pools themselves. Evasion splits in two: active evasion is a production flow, but the passive part accrues with automation growth itself — each unit of new automation ships with $c_0$ units of novel failure surface, $dE_{\text{passive}} = c_0\,dA$, however the unit was produced. (Both forms keep the σ-clock: any term proportional to $\Lambda$ cancels when we divide by $\dot\sigma$. A passive term pegged to *calendar time* would not cancel, and would silently reintroduce the takeoff rate.)

For $m=M/A$,

\[
\dot m =
\frac{\dot M}{A}-m\,\frac{\dot A}{A} =
\Lambda F_m - m\,\Lambda F =
\Lambda(F_m-mF).
\]

Dividing by $\dot\sigma=\Lambda F$ gives

\[
m'=\frac{F_m}{F}-m.
\]

Likewise, for $e=E/A$, using $\dot A/A=\Lambda F$,

\[
\dot e =
\frac{\dot E}{A}-e\,\frac{\dot A}{A} =
c_0\,\Lambda F+\Lambda\,a_{\mathrm{E/M}}c_mq_u-e\,\Lambda F =
\Lambda\left(c_0F+a_{\mathrm{E/M}}c_mq_u-eF\right),
\]

and therefore

\[
e'=c_0+\frac{a_{\mathrm{E/M}}c_mq_u}{F}-e=\frac{F_e}{F}-e,
\qquad
F_e:=c_0F+a_{\mathrm{E/M}}c_mq_u.
\]
