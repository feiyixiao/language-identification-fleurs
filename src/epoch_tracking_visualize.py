#!/usr/bin/env python3
"""
Direction 5: Confusion Matrix Evolution — Visualisation (run locally)
======================================================================
Loads per-epoch confusion matrices saved by epoch_tracking_train.py and
produces three figures showing how confusion patterns evolve across training.

Key questions:
  - Does the Japanese->Spanish collapse appear suddenly or gradually?
  - Does augmentation prevent the collapse from forming at all?
  - Which epoch is the "tipping point"?

Run from src/ after transferring results/epoch_tracking/ from the cluster:
    python epoch_tracking_visualize.py

Generates in figures/:
    epoch_metrics_timeline.png   -- key metrics vs epoch (overall acc,
                                    Japanese recall, Spanish recall,
                                    Italian->Arabic errors)
    epoch_confusion_grid.png     -- small-multiple confusion matrices,
                                    one panel per epoch
    epoch_collapse_spotlight.png -- zoomed Japanese/Spanish cell evolution
"""

import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

LANGUAGES  = ["cmn_hans_cn", "ja_jp", "en_us", "de_de",
              "es_419",      "it_it", "ar_eg", "sw_ke"]
LANG_LABELS = ["Mandarin", "Japanese", "English", "German",
               "Spanish",  "Italian",  "Arabic",  "Swahili"]

JA_IDX = LANG_LABELS.index("Japanese")
ES_IDX = LANG_LABELS.index("Spanish")
IT_IDX = LANG_LABELS.index("Italian")
AR_IDX = LANG_LABELS.index("Arabic")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "epoch_tracking")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

CONDITION_STYLE = {
    "plain":     {"color": "#e74c3c", "label": "XLS-R (no augmentation)", "ls": "-"},
    "augmented": {"color": "#2ecc71", "label": "XLS-R + Augmentation",    "ls": "--"},
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_condition(condition: str):
    """Load all per-epoch confusion matrices for one condition."""
    pattern = os.path.join(RESULTS_DIR, f"{condition}_epoch_*.npy")
    paths   = sorted(glob.glob(pattern))
    if not paths:
        return None, None

    cms    = []
    epochs = []
    for path in paths:
        fname = os.path.basename(path)
        epoch = int(fname.replace(f"{condition}_epoch_", "").replace(".npy", ""))
        cms.append(np.load(path).astype(float))
        epochs.append(epoch)

    meta_path = os.path.join(RESULTS_DIR, f"{condition}_meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    print(f"  {condition}: {len(cms)} epochs loaded")
    return epochs, cms, meta


def extract_metrics(cms):
    """From a list of confusion matrices, extract per-epoch scalar metrics."""
    metrics = {
        "overall_acc":   [],
        "ja_recall":     [],
        "es_recall":     [],
        "it_ar_count":   [],   # Italian->Arabic raw errors
        "ja_es_count":   [],   # Japanese->Spanish raw errors
        "ja_es_rate":    [],   # Japanese->Spanish as fraction of Japanese clips
    }
    for cm in cms:
        total = cm.sum()
        metrics["overall_acc"].append(float(np.diag(cm).sum() / total) if total else 0)

        row_totals = cm.sum(axis=1)
        metrics["ja_recall"].append(
            float(cm[JA_IDX, JA_IDX] / row_totals[JA_IDX]) if row_totals[JA_IDX] else 0)
        metrics["es_recall"].append(
            float(cm[ES_IDX, ES_IDX] / row_totals[ES_IDX]) if row_totals[ES_IDX] else 0)
        metrics["it_ar_count"].append(float(cm[IT_IDX, AR_IDX]))
        metrics["ja_es_count"].append(float(cm[JA_IDX, ES_IDX]))
        metrics["ja_es_rate"].append(
            float(cm[JA_IDX, ES_IDX] / row_totals[JA_IDX]) if row_totals[JA_IDX] else 0)

    return {k: np.array(v) for k, v in metrics.items()}


# ── Figures ───────────────────────────────────────────────────────────────────

def fig1_metrics_timeline(all_data):
    """4-panel timeline: overall acc, Japanese recall, Japanese->Spanish rate,
    Italian->Arabic errors — comparing plain vs augmented."""

    panels = [
        ("overall_acc",  "Overall Test Accuracy",       None,  (0.6, 1.02)),
        ("ja_recall",    "Japanese Recall",             None,  (0.0, 1.05)),
        ("ja_es_rate",   "Japanese → Spanish Rate\n(fraction of Japanese clips)",
                         None,  (0.0, 1.05)),
        ("it_ar_count",  "Italian → Arabic\n(raw error count)", None, None),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    for ax, (metric, title, hline, ylim) in zip(axes, panels):
        for condition, (epochs, cms, meta) in all_data.items():
            style  = CONDITION_STYLE[condition]
            m_vals = extract_metrics(cms)[metric]
            ax.plot(epochs, m_vals,
                    color=style["color"], linestyle=style["ls"],
                    linewidth=2.2, marker="o", markersize=5,
                    label=style["label"], zorder=3)

            # Annotate first epoch that crosses below 0.5 (for recall plots)
            if metric == "ja_recall" and condition == "plain":
                below = [e for e, v in zip(epochs, m_vals) if v < 0.5]
                if below:
                    ax.axvline(below[0], color=style["color"],
                               linestyle=":", alpha=0.5, linewidth=1.2)
                    ax.annotate(f"Collapse\n(epoch {below[0]})",
                                xy=(below[0], 0.5),
                                xytext=(below[0] + 0.3, 0.35),
                                fontsize=8, color=style["color"],
                                arrowprops=dict(arrowstyle="->",
                                                color=style["color"], lw=0.8))

        if hline is not None:
            ax.axhline(hline, color="gray", linestyle=":", linewidth=1, alpha=0.6)
        if ylim:
            ax.set_ylim(ylim)

        ax.set_xlabel("Training Epoch", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_xticks(epochs if epochs else [])
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        "Direction 5: How Does the Japanese–Spanish Collapse Evolve Across Training?\n"
        "Sudden appearance vs gradual drift — and does augmentation prevent it?",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "epoch_metrics_timeline.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def fig2_confusion_grid(condition: str, epochs, cms):
    """Small-multiple confusion matrices, one panel per epoch."""
    n = len(cms)
    ncols = min(5, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.5 * ncols, 3.2 * nrows))
    axes = np.array(axes).flatten()

    # Shared colour scale: normalise each row (recall matrix)
    for ax, epoch, cm in zip(axes, epochs, cms):
        row_totals = cm.sum(axis=1, keepdims=True)
        cm_norm = np.where(row_totals > 0, cm / row_totals, 0)

        sns.heatmap(cm_norm, ax=ax, cmap="Blues", vmin=0, vmax=1,
                    xticklabels=[l[:3] for l in LANG_LABELS],
                    yticklabels=[l[:3] for l in LANG_LABELS],
                    annot=False, linewidths=0.2, cbar=False)

        # Highlight Japanese->Spanish cell
        ax.add_patch(plt.Rectangle((ES_IDX, JA_IDX), 1, 1,
                                   fill=False, edgecolor="red", lw=2.5, zorder=5))
        ja_es = cm_norm[JA_IDX, ES_IDX]
        overall = float(np.diag(cm).sum() / cm.sum()) if cm.sum() > 0 else 0
        ax.set_title(f"Epoch {epoch}\nacc={overall:.2f}  ja→es={ja_es:.2f}",
                     fontsize=8, fontweight="bold",
                     color="red" if ja_es > 0.3 else "black")
        ax.tick_params(labelsize=6)

    for ax in axes[len(cms):]:
        ax.set_visible(False)

    style = CONDITION_STYLE.get(condition, {})
    fig.suptitle(
        f"Confusion Matrix Evolution — {style.get('label', condition)}\n"
        "Red box = Japanese→Spanish cell",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"epoch_confusion_grid_{condition}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def fig3_collapse_spotlight(all_data):
    """Side-by-side: Japanese->Spanish confusion rate cell, plain vs augmented,
    shown as a heatmap over epoch (rows=epoch, cols=target language)."""
    fig, axes = plt.subplots(1, len(all_data), figsize=(7 * len(all_data), 5))
    if len(all_data) == 1:
        axes = [axes]

    for ax, (condition, (epochs, cms, meta)) in zip(axes, all_data.items()):
        # Build matrix: rows=epoch, cols=target language (only Japanese row)
        # Show the full row of where Japanese clips go, per epoch
        ja_rows = np.zeros((len(cms), len(LANG_LABELS)))
        for i, cm in enumerate(cms):
            row_total = cm[JA_IDX].sum()
            if row_total > 0:
                ja_rows[i] = cm[JA_IDX] / row_total

        im = ax.imshow(ja_rows, cmap="Reds", aspect="auto", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.04, label="Fraction of Japanese clips")

        ax.set_xticks(range(len(LANG_LABELS)))
        ax.set_xticklabels(LANG_LABELS, rotation=35, ha="right", fontsize=9)
        ax.set_yticks(range(len(epochs)))
        ax.set_yticklabels([f"Ep {e}" for e in epochs], fontsize=9)
        ax.set_xlabel("Predicted Language", fontsize=10)
        ax.set_ylabel("Training Epoch", fontsize=10)

        style = CONDITION_STYLE.get(condition, {})
        ax.set_title(f"{style.get('label', condition)}\n"
                     "Where do Japanese clips go at each epoch?",
                     fontsize=11, fontweight="bold",
                     color=style.get("color", "black"))

        # Highlight Spanish column
        ax.axvline(ES_IDX - 0.5, color="black", lw=0.5)
        ax.axvline(ES_IDX + 0.5, color="black", lw=0.5)
        ax.annotate("Spanish", xy=(ES_IDX, -0.7), ha="center",
                    fontsize=8, color="red", fontweight="bold",
                    xycoords=("data", "axes fraction"))

    fig.suptitle(
        "Japanese Clip Destinations Across Training Epochs\n"
        "Does the Spanish attractor build gradually or suddenly?",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "epoch_collapse_spotlight.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  Direction 5: Epoch Confusion Evolution Visualisation")
    print("="*60)

    conditions = ["plain", "augmented"]
    all_data = {}
    for cond in conditions:
        result = load_condition(cond)
        if result[0] is not None:
            epochs, cms, meta = result
            all_data[cond] = (epochs, cms, meta)

    if not all_data:
        print("\n  No epoch data found in results/epoch_tracking/")
        print("  Run epoch_tracking_train.py on the cluster first, then")
        print("  transfer results/epoch_tracking/ here.")
        return

    # Use epochs list from first available condition
    epochs = next(iter(all_data.values()))[0]

    print("\nGenerating figures...")
    fig1_metrics_timeline(all_data)

    for condition, (epochs, cms, meta) in all_data.items():
        fig2_confusion_grid(condition, epochs, cms)

    fig3_collapse_spotlight(all_data)

    print("\n" + "="*60)
    print("  Done. Figures saved to figures/:")
    print("    epoch_metrics_timeline.png")
    for cond in all_data:
        print(f"    epoch_confusion_grid_{cond}.png")
    print("    epoch_collapse_spotlight.png")
    print("="*60)


if __name__ == "__main__":
    main()
