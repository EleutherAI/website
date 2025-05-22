# Comparing POSER and Tyche: Perturbation Strategies for Measuring Model Alignment

This blog post summarizes our investigation comparing two perturbation-based methods for evaluating model alignment: POSER (activation-space perturbation) and Tyche (parameter-space perturbation). Our aim was to understand how model behavior changes under small interventions, and whether aligned and misaligned models differ in their sensitivity to these perturbations.


## Motivation

Language models often exhibit subtle misaligned behavior, and detecting this reliably remains a challenge. POSER and Tyche both propose frameworks to probe alignment by applying perturbations and measuring downstream effects using KL divergence. Our experiments focus on contrasting the effects of these two approaches on aligned vs misaligned models.


## Methodology
### POSER:
Perturbs the residual stream at a specific layer (e.g., layer 15).
Constructs a steering vector from contrast pairs (clean vs critical prompts).
Applies scalar multiples of the vector (coefficients 0 to 20) to the residual stream.
Measures KL divergence between perturbed and original outputs.
### Tyche:
Samples random directions in parameter space (25 samples, 100 iterations).
Scales each vector to match a target KL divergence (e.g., 1e-2).
Operates either across all layers or a single layer (e.g., layer 15).
Computes volume estimates and logs perturbed/unperturbed logits for analysis.


## Experimental Setup
1. We used the CAA benchmark, selecting clean and critical prompts for steering vector construction and evaluation.
2. Models: truth-teller (aligned), genie (aligned), *-maximizer (misaligned).
3. Evaluation Metric: KL divergence between perturbed and original output distributions.


## Key Findings
POSER Sensitivity: Misaligned models (e.g., *-maximizers) exhibit larger KL divergence under large POSER coefficients (≥ 14) compared to aligned models, suggesting greater instability in response to steering.


(KL vs Coefficient curves and vector norm comparisons plotted from: /mnt/ssd-1/dipika/POSER/compare_perturbations/comparison_plots)
Figure 1a: KL divergence increases for misaligned models at higher POSER coefficients.
Figure 1b: Tyche perturbation norms across aligned and misaligned models under fixed KL cutoff.
Figure 2: auc for genie-0 maximizer -0 truth teller -0 


