#!/usr/bin/env python3
"""
Direction 2: Layer-wise Probing — STEP 2: Probe Training & Visualisation
=========================================================================
Run this script LOCALLY after transferring results/layer_probes/ from the cluster.
It trains a lightweight logistic regression probe on each layer's mean-pooled
representations and plots how well language identity can be decoded from each layer.

Usage (from src/, after running layer_probe_extract.py on the cluster):
    python layer_probe_visualize.py --model xlsr_augmented
    python layer_probe_visualize.py --model xlsr_augmented xlsr wav2vec2  (overlay)

Generates figures in figures/:
    layer_probe_<model>.png        -- accuracy per layer for one model
    layer_probe_comparison.png     -- all models overlaid (if multiple --model args)
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "layer_probes")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

MODEL_COLORS = {
    "wav2vec2":       "#3498db",
    "xlsr":           "#e74c3c",
    "xlsr_fulldata":  "#e67e22",
    "xlsr_augmented": "#2ecc71",
}
MODEL_DISPLAY = {
    "wav2vec2":       "wav2vec2-base (83.3% test acc)",
    "xlsr":           "XLS-R 1k/lang (87.2% test acc)",
    "xlsr_fulldata":  "XLS-R full data (89.0% test acc)",
    "xlsr_augmented": "XLS-R + Augmentation (93.6% test acc)",
}

LANG_LABELS = ["Mandarin", "Japanese", "English", "German",
               "Spanish",  "Italian",  "Arabic",  "Swahili"]


# ── Probe training ────────────────────────────────────────────────────────────

def train_probes(features, labels, n_folds=5):
    """
    Train a logistic regression probe on each layer using cross-validation.

    Args:
        features: np.ndarray of shape (n_layers, n_samples, hidden_dim)
        labels:   np.ndarray of shape (n_samples,)

    Returns:
        mean_accs: np.ndarray of shape (n_layers,)
        std_accs:  np.ndarray of shape (n_layers,)
        per_class: np.ndarray of shape (n_layers, n_classes)  per-language recall
    """
    n_layers, n_samples, hidden_dim = features.shape
    n_classes = len(np.unique(labels))
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    mean_accs = np.zeros(n_layers)
    std_accs  = np.zeros(n_layers)
    per_class = np.zeros((n_layers, n_classes))

    for layer_idx in range(n_layers):
        X = features[layer_idx]   # (n_samples, hidden_dim)
        fold_accs = []
        fold_per_class = np.zeros((n_folds, n_classes))

        for fold_i, (train_idx, val_idx) in enumerate(kf.split(X, labels)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val   = scaler.transform(X_val)

            clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs",
                                     n_jobs=-1)
            clf.fit(X_train, y_train)

            preds = clf.predict(X_val)
            fold_accs.append((preds == y_val).mean())

            for cls in range(n_classes):
                mask = y_val == cls
                if mask.sum() > 0:
                    fold_per_class[fold_i, cls] = (preds[mask] == y_val[mask]).mean()

        mean_accs[layer_idx] = np.mean(fold_accs)
        std_accs[layer_idx]  = np.std(fold_accs)
        per_class[layer_idx] = fold_per_class.mean(axis=0)

        print(f"  Layer {layer_idx:2d}: acc = {mean_accs[layer_idx]:.3f} "
              f"± {std_accs[layer_idx]:.3f}")

    return mean_accs, std_accs, per_class


# ── Load data ─────────────────────────────────────────────────────────────────

def load_probe_data(model_name):
    feat_path  = os.path.join(RESULTS_DIR, f"{model_name}_layer_features.npy")
    label_path = os.path.join(RESULTS_DIR, f"{model_name}_labels.npy")
    meta_path  = os.path.join(RESULTS_DIR, f"{model_name}_meta.json")

    for p in [feat_path, label_path, meta_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing: {p}\n"
                "Run layer_probe_extract.py on the HPC cluster first, "
                "then transfer results/layer_probes/ here."
            )

    features = np.load(feat_path)    # (n_layers, N, D)
    labels   = np.load(label_path)   # (N,)
    with open(meta_path) as f:
        meta = json.load(f)

    print(f"Loaded {model_name}: {features.shape[0]} layers, "
          f"{features.shape[1]} samples, hidden_dim={features.shape[2]}")
    return features, labels, meta


# ── Visualisations ────────────────────────────────────────────────────────────

def fig_single_model(model_name, mean_accs, std_accs, per_class, meta):
    """Detailed 2-panel figure for one model: overall + per-language accuracy."""
    n_layers = len(mean_accs)
    n_transformer = meta["n_transformer_layers"]
    x = np.arange(n_layers)
    x_labels = (["CNN\nout"] +
                [str(i) for i in range(1, n_transformer + 1)])

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Panel 1: overall accuracy with error band
    ax = axes[0]
    ax.plot(x, mean_accs, color=MODEL_COLORS.get(model_name, "steelblue"),
            linewidth=2.5, marker="o", markersize=5, zorder=3)
    ax.fill_between(x,
                    mean_accs - std_accs,
                    mean_accs + std_accs,
                    alpha=0.2, color=MODEL_COLORS.get(model_name, "steelblue"))

    # Mark peak layer
    peak = int(np.argmax(mean_accs))
    ax.axvline(peak, color="gray", linestyle="--", alpha=0.6, linewidth=1.2)
    ax.annotate(f"Peak: layer {peak}\nacc={mean_accs[peak]:.3f}",
                xy=(peak, mean_accs[peak]),
                xytext=(peak + 0.5, mean_accs[peak] - 0.05),
                fontsize=9, arrowprops=dict(arrowstyle="->", color="gray"))

    # Chance line
    chance = 1.0 / len(np.unique(np.arange(8)))
    ax.axhline(chance, color="red", linestyle=":", linewidth=1.2,
               label=f"Chance ({chance:.2f})")

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_ylabel("Probe Accuracy (5-fold CV)", fontsize=11)
    ax.set_title(
        f"Layer-wise Probing — {MODEL_DISPLAY.get(model_name, model_name)}\n"
        "Logistic regression probe accuracy at each transformer layer",
        fontsize=12, fontweight="bold"
    )
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)

    # Panel 2: per-language recall heatmap across layers
    ax2 = axes[1]
    im = ax2.imshow(per_class.T, aspect="auto", cmap="RdYlGn",
                    vmin=0, vmax=1, origin="upper")
    plt.colorbar(im, ax=ax2, fraction=0.03, pad=0.02, label="Recall")
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels, fontsize=8)
    ax2.set_yticks(range(len(LANG_LABELS)))
    ax2.set_yticklabels(LANG_LABELS, fontsize=10)
    ax2.set_xlabel("Layer (0 = CNN encoder output)", fontsize=11)
    ax2.set_ylabel("Language", fontsize=11)
    ax2.set_title("Per-language Recall at Each Layer", fontsize=11)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"layer_probe_{model_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def fig_comparison(probe_results):
    """Overlay all models on one plot for easy comparison."""
    fig, ax = plt.subplots(figsize=(14, 6))

    for model_name, (mean_accs, std_accs, _, meta) in probe_results.items():
        n_layers = len(mean_accs)
        x = np.arange(n_layers)
        color = MODEL_COLORS.get(model_name, "gray")
        label = MODEL_DISPLAY.get(model_name, model_name).split(" (")[0]
        ax.plot(x, mean_accs, color=color, linewidth=2.2, marker="o",
                markersize=4, label=label, zorder=3)
        ax.fill_between(x, mean_accs - std_accs, mean_accs + std_accs,
                        alpha=0.12, color=color)

    chance = 1.0 / 8
    ax.axhline(chance, color="black", linestyle=":", linewidth=1,
               label=f"Chance ({chance:.2f})", alpha=0.6)

    ax.set_xlabel("Layer (0 = CNN encoder output)", fontsize=12)
    ax.set_ylabel("Probe Accuracy (5-fold CV)", fontsize=12)
    ax.set_title(
        "Layer-wise Language ID Probe Accuracy — All Models\n"
        "Where is language identity encoded in the transformer?",
        fontsize=13, fontweight="bold"
    )
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "layer_probe_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model", nargs="+", required=True,
        choices=list(MODEL_DISPLAY.keys()),
        help="One or more model names to probe (e.g. --model xlsr_augmented xlsr wav2vec2)"
    )
    args = ap.parse_args()

    probe_results = {}

    for model_name in args.model:
        print(f"\n{'='*60}")
        print(f"  Probing: {model_name}")
        print(f"{'='*60}")

        features, labels, meta = load_probe_data(model_name)

        print(f"Training probes ({meta['n_total_layers']} layers)...")
        mean_accs, std_accs, per_class = train_probes(features, labels)

        probe_results[model_name] = (mean_accs, std_accs, per_class, meta)

        print("\nGenerating single-model figure...")
        fig_single_model(model_name, mean_accs, std_accs, per_class, meta)

    if len(probe_results) > 1:
        print("\nGenerating comparison figure...")
        fig_comparison(probe_results)

    print("\nDone.")


if __name__ == "__main__":
    main()
