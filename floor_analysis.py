"""
floor_analysis.py
=================
Week 1-2 pipeline: estimate human-derived performance floors (E_human) from
multi-annotator label distributions, with uncertainty, and measure how the
estimates degrade as annotators-per-item shrink.

Floors computed (per the proposal, Section 6):
  * 0-1 floor      : R*_01 = 1 - mean_i( max_k p_hat[i,k] )
                     = minimum error of any classifier predicting a NEW
                       annotator drawn from the same population.
  * Cross-entropy  : E_CE = mean_i( H(p_hat[i]) )   [nats]
                     = minimum achievable CE = conditional entropy H(Y|X).
  * Miller-Madow   : E_CE + mean_i( (K_obs_i - 1) / (2 m_i) )
                     bias-corrected entropy floor (K_obs = observed support).

Experiments:
  1. Full-data floors with item-level bootstrap CIs  (the E_human that later
     gets compared against the fitted scaling asymptote E_ML).
  2. Annotator subsampling: m in {40,20,10,5,3,2} drawn WITHOUT replacement
     from each item's real annotations; measures bias vs the full-m reference.
     Pre-registered prediction: BOTH floors are underestimated at small m
     (Jensen / winner's-curse on max; small-sample bias on entropy).
  3. Synthetic self-test: same machinery on multinomial data with KNOWN p,
     so the bias direction is demonstrated against actual ground truth.

Data loaders: CIFAR-10H (counts .npy) and ChaosNLI (.jsonl with label_counter).

Usage:
  python floor_analysis.py --cifar10h data/cifar10h-counts.npy --out results/
  python floor_analysis.py --chaosnli chaosNLI_snli.jsonl --out results/
  python floor_analysis.py --selftest --out results/

Author: Kai (Horizon Academic project). Pipeline drafted with AI assistance;
every function is meant to be read, understood, and owned by the author.
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------------

def load_cifar10h(counts_path):
    """CIFAR-10H: (10000, 10) integer counts. Returns counts matrix."""
    counts = np.load(counts_path)
    assert counts.ndim == 2, "expected (items, classes)"
    return counts.astype(np.int64)


def load_chaosnli(jsonl_path):
    """ChaosNLI jsonl: one JSON per line with a 'label_counter' dict.
    Class order is fixed from the union of keys (sorted for determinism)."""
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    keys = sorted({k for r in rows for k in r["label_counter"].keys()})
    counts = np.zeros((len(rows), len(keys)), dtype=np.int64)
    for i, r in enumerate(rows):
        for j, k in enumerate(keys):
            counts[i, j] = int(r["label_counter"].get(k, 0))
    print(f"[chaosnli] {len(rows)} items, classes = {keys}")
    return counts


# ----------------------------------------------------------------------------
# Floor computations (all take a counts matrix: items x classes)
# ----------------------------------------------------------------------------

def phat(counts):
    m = counts.sum(axis=1, keepdims=True)
    return counts / m


def floor_01(counts):
    """Plug-in 0-1 floor: 1 - mean top-vote share."""
    return float(1.0 - phat(counts).max(axis=1).mean())


def entropy_rows(counts):
    """Per-item plug-in Shannon entropy in nats (0 log 0 := 0)."""
    p = phat(counts)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(p > 0, p * np.log(p), 0.0)
    return -t.sum(axis=1)


def floor_ce(counts):
    """Plug-in cross-entropy floor = mean conditional entropy (nats)."""
    return float(entropy_rows(counts).mean())


def floor_ce_miller_madow(counts):
    """Miller-Madow corrected CE floor: adds (K_obs - 1) / (2 m) per item."""
    m = counts.sum(axis=1)
    k_obs = (counts > 0).sum(axis=1)
    return float((entropy_rows(counts) + (k_obs - 1) / (2.0 * m)).mean())


def all_floors(counts):
    return {
        "floor_01": floor_01(counts),
        "floor_ce_nats": floor_ce(counts),
        "floor_ce_miller_madow_nats": floor_ce_miller_madow(counts),
        "mean_annotations_per_item": float(counts.sum(1).mean()),
        "n_items": int(counts.shape[0]),
        "n_classes": int(counts.shape[1]),
    }


# ----------------------------------------------------------------------------
# Uncertainty: item-level bootstrap CI on the full-data floors.
# (This is the CI that matters when later comparing E_human to E_ML,
#  because the dataset of items is itself a sample.)
# ----------------------------------------------------------------------------

def bootstrap_ci_items(counts, fn, n_boot=2000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    n = counts.shape[0]
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats[b] = fn(counts[idx])
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


# ----------------------------------------------------------------------------
# Annotator subsampling WITHOUT replacement (the bias experiment).
# Vectorized: pad each item's annotation list to max length, assign random
# sort keys (+inf on padding), argsort -> random order of REAL annotations
# first; prefix of length m = m annotators sampled without replacement.
# ----------------------------------------------------------------------------

def _label_matrix(counts):
    """Expand counts to a padded (items x max_m) matrix of class labels."""
    n_items, K = counts.shape
    m_per_item = counts.sum(axis=1)
    max_m = int(m_per_item.max())
    L = np.full((n_items, max_m), -1, dtype=np.int8)
    for i in range(n_items):
        row = np.repeat(np.arange(K, dtype=np.int8), counts[i])
        L[i, : row.size] = row
    return L, m_per_item


def subsample_floors(counts, m_values, n_reps=200, seed=1):
    """For each m and replicate: draw m annotations per item without
    replacement, recompute all three floors. Returns dict of arrays."""
    L, m_per_item = _label_matrix(counts)
    n_items, max_m = L.shape
    K = counts.shape[1]
    if max(m_values) > int(m_per_item.min()):
        raise ValueError(
            f"max m={max(m_values)} exceeds min annotations per item "
            f"({int(m_per_item.min())})"
        )
    rng = np.random.default_rng(seed)
    pad_mask = L < 0

    out = {m: {"floor_01": [], "floor_ce": [], "floor_ce_mm": []}
           for m in m_values}

    for _ in range(n_reps):
        keys = rng.random((n_items, max_m))
        keys[pad_mask] = np.inf                    # padding sorts last
        order = np.argsort(keys, axis=1)
        shuffled = np.take_along_axis(L, order, axis=1)
        for m in m_values:
            prefix = shuffled[:, :m]               # m real annotations/item
            sub_counts = np.stack(
                [(prefix == k).sum(axis=1) for k in range(K)], axis=1
            ).astype(np.int64)
            out[m]["floor_01"].append(floor_01(sub_counts))
            out[m]["floor_ce"].append(floor_ce(sub_counts))
            out[m]["floor_ce_mm"].append(floor_ce_miller_madow(sub_counts))

    for m in m_values:
        for k in out[m]:
            out[m][k] = np.asarray(out[m][k])
    return out


# ----------------------------------------------------------------------------
# Synthetic self-test: multinomial data with KNOWN p, so true floors are
# computable exactly and estimator bias is demonstrated, not inferred.
# ----------------------------------------------------------------------------

def synthetic_selftest(n_items=5000, K=3, m_values=(2, 3, 5, 10, 20, 40),
                       m_gen=200, concentration=1.0, n_reps=100, seed=7):
    """Items' true p ~ Dirichlet(concentration). True floors known exactly.
    Generate m_gen annotations/item, then subsample."""
    rng = np.random.default_rng(seed)
    P = rng.dirichlet([concentration] * K, size=n_items)
    true_01 = float(1.0 - P.max(axis=1).mean())
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(P > 0, P * np.log(P), 0.0)
    true_ce = float(-t.sum(axis=1).mean())

    counts = np.stack([rng.multinomial(m_gen, P[i]) for i in range(n_items)])
    sub = subsample_floors(counts, list(m_values), n_reps=n_reps, seed=seed + 1)
    return {"true_floor_01": true_01, "true_floor_ce": true_ce,
            "m_gen": m_gen, "subsampled": sub, "K": K, "n_items": n_items}


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------

def plot_bias_curves(sub, ref_01, ref_ce, title, path, ref_label="full-data reference"):
    m_vals = sorted(sub.keys())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    for ax, key, ref, name in [
        (axes[0], "floor_01", ref_01, "0-1 floor  (1 - mean max p)"),
        (axes[1], "floor_ce", ref_ce, "Cross-entropy floor  (nats)"),
    ]:
        means = [sub[m][key].mean() for m in m_vals]
        lo = [np.quantile(sub[m][key], 0.025) for m in m_vals]
        hi = [np.quantile(sub[m][key], 0.975) for m in m_vals]
        ax.plot(m_vals, means, "o-", label="plug-in estimate")
        ax.fill_between(m_vals, lo, hi, alpha=0.25)
        if key == "floor_ce":
            mm = [sub[m]["floor_ce_mm"].mean() for m in m_vals]
            ax.plot(m_vals, mm, "s--", label="Miller-Madow corrected")
        ax.axhline(ref, color="k", ls=":", label=ref_label)
        ax.set_xscale("log")
        ax.set_xticks(m_vals)
        ax.set_xticklabels(m_vals)
        ax.set_xlabel("annotators per item (m)")
        ax.set_title(name)
        ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_entropy_hist(counts, title, path):
    H = entropy_rows(counts)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.hist(H, bins=60)
    ax.set_xlabel("per-item entropy H(Y|x)  [nats]")
    ax.set_ylabel("items")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def run_dataset(counts, name, m_values, outdir, n_reps, n_boot):
    res = {"dataset": name, "full_data": all_floors(counts)}

    # bootstrap CIs on the headline floors (item-level resampling)
    res["full_data"]["ci_floor_01"] = bootstrap_ci_items(
        counts, floor_01, n_boot=n_boot)
    res["full_data"]["ci_floor_ce_nats"] = bootstrap_ci_items(
        counts, floor_ce, n_boot=n_boot)

    print(f"\n=== {name}: full-data floors ===")
    fd = res["full_data"]
    print(f"  items={fd['n_items']}  classes={fd['n_classes']}  "
          f"mean m={fd['mean_annotations_per_item']:.1f}")
    print(f"  0-1 floor            = {fd['floor_01']:.4f}  "
          f"95% CI {fd['ci_floor_01']}")
    print(f"  CE floor (nats)      = {fd['floor_ce_nats']:.4f}  "
          f"95% CI {fd['ci_floor_ce_nats']}")
    print(f"  CE floor, MM-corr.   = {fd['floor_ce_miller_madow_nats']:.4f}")

    # subsampling experiment
    sub = subsample_floors(counts, m_values, n_reps=n_reps)
    res["subsampling"] = {
        str(m): {k: {"mean": float(v.mean()),
                     "q025": float(np.quantile(v, 0.025)),
                     "q975": float(np.quantile(v, 0.975))}
                 for k, v in sub[m].items()}
        for m in m_values
    }
    print(f"  --- subsampling bias (reference = full-data plug-in) ---")
    for m in m_values:
        d01 = sub[m]["floor_01"].mean() - fd["floor_01"]
        dce = sub[m]["floor_ce"].mean() - fd["floor_ce_nats"]
        print(f"  m={m:>3}: bias(0-1) = {d01:+.4f}   bias(CE) = {dce:+.4f}")

    plot_bias_curves(sub, fd["floor_01"], fd["floor_ce_nats"],
                     f"{name}: floor estimates vs annotators per item",
                     os.path.join(outdir, f"{name}_subsampling_bias.png"))
    plot_entropy_hist(counts, f"{name}: distribution of per-item ambiguity",
                      os.path.join(outdir, f"{name}_entropy_hist.png"))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cifar10h", help="path to cifar10h-counts.npy")
    ap.add_argument("--chaosnli", help="path to a ChaosNLI .jsonl file")
    ap.add_argument("--selftest", action="store_true",
                    help="run synthetic validation with known ground truth")
    ap.add_argument("--out", default="results")
    ap.add_argument("--m-values", default="40,20,10,5,3,2")
    ap.add_argument("--n-reps", type=int, default=200)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    m_values = [int(x) for x in args.m_values.split(",")]
    results = {}

    if args.selftest:
        st = synthetic_selftest()
        print("\n=== synthetic self-test (ground truth KNOWN) ===")
        print(f"  true 0-1 floor = {st['true_floor_01']:.4f}   "
              f"true CE floor = {st['true_floor_ce']:.4f} nats")
        for m in sorted(st["subsampled"].keys()):
            b01 = st["subsampled"][m]["floor_01"].mean() - st["true_floor_01"]
            bce = st["subsampled"][m]["floor_ce"].mean() - st["true_floor_ce"]
            bmm = st["subsampled"][m]["floor_ce_mm"].mean() - st["true_floor_ce"]
            print(f"  m={m:>3}: bias(0-1)={b01:+.4f}  bias(CE)={bce:+.4f}  "
                  f"bias(CE, MM-corrected)={bmm:+.4f}")
        plot_bias_curves(
            st["subsampled"], st["true_floor_01"], st["true_floor_ce"],
            "Synthetic self-test (dashed reference = TRUE floors)",
            os.path.join(args.out, "selftest_bias.png"),
            ref_label="true floor (known)")
        results["selftest"] = {
            "true_floor_01": st["true_floor_01"],
            "true_floor_ce": st["true_floor_ce"],
            "bias_by_m": {
                str(m): {
                    "bias_01": float(st["subsampled"][m]["floor_01"].mean()
                                     - st["true_floor_01"]),
                    "bias_ce": float(st["subsampled"][m]["floor_ce"].mean()
                                     - st["true_floor_ce"]),
                    "bias_ce_mm": float(st["subsampled"][m]["floor_ce_mm"].mean()
                                        - st["true_floor_ce"]),
                } for m in st["subsampled"]
            },
        }

    if args.cifar10h:
        counts = load_cifar10h(args.cifar10h)
        results["cifar10h"] = run_dataset(
            counts, "cifar10h", m_values, args.out, args.n_reps, args.n_boot)

    if args.chaosnli:
        counts = load_chaosnli(args.chaosnli)
        results["chaosnli"] = run_dataset(
            counts, "chaosnli", m_values, args.out, args.n_reps, args.n_boot)

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {args.out}/results.json")


if __name__ == "__main__":
    main()
