# Predicting Scaling Floors from Human Annotator Disagreement

Can the asymptotic performance limit of a machine-learning classifier be
predicted **before training**, from human annotator disagreement alone?

This repository contains the pre-registration and pilot artifacts for a
research project testing that question on ChaosNLI (Nie, Zhou & Bansal 2020;
100 annotations/item) and CIFAR-10H (Peterson et al. 2019; ~51/item).

## The claim

For any model producing a distribution q, cross-entropy against the human
label distribution p decomposes as CE = H(p) + KL(p‖q) ≥ H(p). The conditional
entropy of human judgments is therefore a model-independent floor — computable
from annotations alone. This project (1) computes those floors with
confidence intervals, (2) registers them as predictions (`PREREGISTRATION.md`),
and (3) tests whether empirical data-scaling curves L(D) = E + A/D^β flatten
where the floors say they must.

## Contents

- `PREREGISTRATION.md` — hypotheses, design, and quantitative predictions,
  committed **before any model training**. The commit timestamp is the seal.
- `floor_analysis.py` — computes the floors (0–1 and cross-entropy) with
  bootstrap CIs, plus the annotator-subsampling bias experiment; includes a
  synthetic self-test against known ground truth. No model training involved.
- `results/` — pilot outputs: floors for CIFAR-10H, ChaosNLI-SNLI, ChaosNLI-MNLI.
- `figures/` — pilot figures, including the three-dataset bias replication and
  the H1/H2/H3 hypothesis illustration.
- `LAB_NOTE_week1.md` — pilot-phase notes.

## Pilot headline (pre-training)

| Dataset | Best possible agreement with a new annotator | CE floor (nats) |
|---|---|---|
| CIFAR-10H | 95.4% | 0.155 |
| ChaosNLI-SNLI | 75.5% | 0.553 |
| ChaosNLI-MNLI | 65.1% | 0.743 |

Secondary pilot finding: with 3 annotators per item — the field's standard
budget — these floors are underestimated by roughly 30–50%, in the direction
predicted by Jensen's inequality and small-sample entropy bias, replicated
across all three datasets.

## Data

- ChaosNLI: https://github.com/easonnie/ChaosNLI (data via the Dropbox link
  in their README)
- CIFAR-10H: https://github.com/jcpeterson/cifar-10h

Research project, 2026. Pipeline drafted with AI assistance;
all predictions, design decisions, and this registration are the author's.
