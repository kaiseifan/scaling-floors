Pre-registration: Predicting Scaling Floors from Human Annotator Disagreement
Author: Kai Seifan · Date: Friday, September  4, 2026 · Status: registered before any model training has occurred.
1. Hypotheses
Given the data-scaling law L(D) = E + A/Dᵝ, where E is the asymptotic (irreducible) loss floor and β the rate at which loss falls as training data grows:
H1 (floor): L(D, H) = E(H) + A/Dᵝ — greater disagreement raises the minimum achievable loss; learning proceeds at approximately the same rate.
H2 (rate): L(D, H) = E + A/Dᵝ⁽ᴴ⁾ — ambiguous examples require more data but approach approximately the same floor.
H3 (both): L(D, H) = E(H) + A/Dᵝ⁽ᴴ⁾ — ambiguity changes both where learning stops and how quickly it gets there.
Throughout, "rate" means the data-scaling exponent β, compared between entropy strata of items within the same experiment — it is not the optimizer's learning rate, which is held fixed (§3).
Prediction, soft-label condition: H3. The floor rises with entropy as a matter of algebra (CE = H(p) + KL(p‖q)), and I additionally predict the rate β falls with entropy, because matching a spread distribution requires calibrating several proportions where matching a near-unanimous one requires only confident correctness.
Prediction, hard-label condition: H3. Contested items yield conflicting training labels (slowing the rate), while near-tie majorities are both statistically fragile and frequently inconsistent with the original training labels (elevating the observed floor within any finite data range).
Calibration note (declared for honesty): the floor half of the soft-label prediction is near-forced by the identity above; confirming it validates the machinery rather than testing a risky claim. The genuinely falsifiable components are: (a) β falling with entropy under soft labels, (b) the entire hard-label prediction, and (c) the quantitative floor values in §6.
2. Confirmatory test
For each dataset I fit L(D) = E + A·D⁻ᵝ twice: once with E fixed at the pre-registered human floor in §6, once with E free. The fits are compared by likelihood-ratio test and AIC; the prediction succeeds if the fixed-floor model fits comparably to the free model. Fits are additionally validated by extrapolation — fitting on all but the largest D and predicting the held-out point (per Alabdulmohsin et al., 2022). The H1/H2/H3 comparison is made by fitting curves within entropy strata and testing whether E, β, or both vary across strata.
3. Design — fixed as of this date
Scaling datasets: SNLI and MNLI (evaluated on ChaosNLI-SNLI and ChaosNLI-MNLI respectively); CIFAR-10 → CIFAR-10H as the objective-task contrast.
Training sizes: 1k / 5k / 25k / 100k / 550k (SNLI; MNLI capped at its 393k). Seeds: 3, to check reliability. Model: BERT-Tiny. Primary loss: cross-entropy; 0–1 error secondary.
Rationale: many training runs across log-spaced data sizes trace the full data-scaling curve rather than a single point; three seeds separate real curve structure from run-to-run noise.
Clarification of scope: one architecture, many runs (15 per text dataset). A model-size sweep (BERT-Tiny → Medium) is a stretch extension, predicted to change β, not E.
Optimizer hyperparameters (learning rate, batch size, early-stopping rule on validation loss) are held fixed across all runs and dataset sizes, so that differences between curves reflect data quantity and item ambiguity only.
αNLI is excluded from the main analysis (K=2); it is reserved for validating that the multiclass estimator reduces to the established binary estimator (Ushio et al., ICLR 2026).
4. Pilot results already obtained — declared for transparency
Floor-estimate bias at small annotator counts: confirmed in the predicted direction (both floors underestimated, shrinking ~1/m) on synthetic ground truth and on all three datasets.
The loss-dependent bias flip (0–1 bias relatively worse on ambiguous data; cross-entropy bias relatively worse on near-unanimous data): discovered post-hoc, not predicted; mechanism proposed after observation.
MNLI floor exceeding SNLI's: no explicit prediction was recorded before running MNLI. The result is treated as observed, not predicted.
5. Why these predictions are model-independent
The floor is the same no matter which model is trained. This is not an assumption but a consequence of the identity CE = H(p) + KL(p‖q): the KL term is non-negative for any distribution q a model produces, so nothing that can occupy the q slot — any architecture, any size — scores below H(p). Models can change, and what changes with them is the approach to the floor, not its height: capacity and data move β. Actually reaching the floor is a stronger claim than never beating it, and requires three things: a model expressive enough to represent p, a training target that is p (hence the sharp prediction in §6 applies only to soft-label training), and optimization that succeeds — which is why training spans a large data range: to give the curve every chance to arrive.
6. Registered predictions
Dataset
CE floor [95% CI] (nats)
0–1 floor [95% CI]
ChaosNLI-SNLI
0.553 [0.542, 0.565]
0.245 [0.238, 0.253]
ChaosNLI-MNLI
0.743 [0.734, 0.752]
0.349 [0.343, 0.356]
CIFAR-10H
0.155 [0.150, 0.159]
0.046 [0.044, 0.047]

1. No fitted asymptote, in any condition, on any dataset, will land below its floor. A violation falsifies the framework.
2. Under soft-label training, the fitted cross-entropy asymptote lands inside the interval for SNLI and MNLI. Landing outside fails the sharp prediction.
3. Under hard-label training, the asymptote lands above the floor; the gap is interpreted as training-target mismatch (the KL term).
CIFAR-10H is held to predictions 1 and 3 only, owing to its documented labeling shift (annotations collected on downscaled 32×32 images) — declared here as a limitation rather than discovered later.

