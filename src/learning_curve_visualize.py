#!/usr/bin/env python3
"""
Direction 3: Learning Curve — STEP 2: Visualisation (run locally)
==================================================================
Plots overall accuracy and per-language recall vs. training samples per language.

Works immediately with existing data (1000 and ~2700 sample points from
confusion_matrix_xlsr.npy and confusion_matrix_xlsr_fulldata.npy).
Additional points from learning_curve_train.py are merged in automatically
when their JSON files appear in results/learning_curve/.

Run from the src/ directory:
    python learning_curve_visualize.py

Generates figures in figures/:
    learning_curve_accuracy.png       -- overall accuracy vs. samples/lang
    learning_curve_per_language.png   -- per-language recall vs. samples/lang
    learning_curve_heatmap.png        -- recall heatmap (language × data size)
"""

import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

LANGUAGES = ["cmn_hans_cn", "ja_jp", "en_us", "de_de",
             "es_419",      "it_it", "ar_eg", "sw_ke"]
LANG_LABELS = ["Mandarin", "Japanese", "English", "German",
               "Spanish",  "Italian",  "Arabic",  "Swahili"]

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
LC_DIR      = os.path.join(RESULTS_DIR, "learning_curve")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Languages to highlight in the per-language plot
HIGHLIGHT = {
    "Japanese": ("#e74c3c", 2.5, "o"),   # red, thicker, circle
    "Spanish":  ("#e67e22", 2.0, "s"),   # orange, square
    "Mandarin": ("#9b59b6", 1.5, "^"),   # purple, triangle
}
DEFAULT_STYLE = ("#aaaaaa", 1.0, ".")


# ── Data collection ───────────────────────────────────────────────────────────

def recall_from_cm(cm, lang_idx):
    row = cm[lang_idx]
    total = row.sum()
    return float(cm[lang_idx, lang_idx] / total) if total > 0 else 0.0


def load_known_points():
    """
    Extract data points from existing confusion matrices.
    Returns list of dicts with keys:
      samples_per_lang, overall_accuracy, per_lang_recall, source, augmented
    """
    points = []

    # Mapping: (confusion matrix file, approx samples/lang, label, augmented)
    known = [
        ("confusion_matrix_xlsr.npy",          1000,  "xlsr",           False),
        ("confusion_matrix_xlsr_fulldata.npy",  2700,  "xlsr_fulldata",  False),
        ("confusion_matrix_xlsr_augmented.npy", 2700,  "xlsr_augmented", True),
    ]

    # Overall accuracies from results txt files (diagonal sum / total)
    # We recompute from the confusion matrix so we don't need to parse text.

    for fname, n_samples, label, augmented in known:
        path = os.path.join(RESULTS_DIR, fname)
        if not os.path.exists(path):
            print(f"  [skip] Missing: {path}")
            continue
        cm = np.load(path).astype(float)
        total = cm.sum()
        overall_acc = float(np.diag(cm).sum() / total) if total > 0 else 0.0
        per_lang = {LANG_LABELS[i]: recall_from_cm(cm, i)
                    for i in range(len(LANG_LABELS))}
        points.append({
            "samples_per_lang": n_samples,
            "overall_accuracy": overall_acc,
            "per_lang_recall":  per_lang,
            "source":           label,
            "augmented":        augmented,
        })
        print(f"  Loaded existing: {label} ({n_samples} samples/lang, "
              f"acc={overall_acc:.3f})")

    return points


def load_cluster_points():
    """Load JSON results from learning_curve_train.py cluster runs."""
    points = []
    pattern = os.path.join(LC_DIR, "*_samples.json")
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            data = json.load(f)
        data["source"]    = "cluster"
        data["augmented"] = False
        points.append(data)
        print(f"  Loaded cluster result: {data['samples_per_lang']} samples/lang, "
              f"acc={data['overall_accuracy']:.3f}")
    return points


def build_curve(points):
    """Merge and sort all data points; separate augmented vs. non-augmented."""
    plain = [p for p in points if not p["augmented"]]
    aug   = [p for p in points if p["augmented"]]
    plain.sort(key=lambda p: p["samples_per_lang"])
    aug.sort(  key=lambda p: p["samples_per_lang"])
    return plain, aug


# ── Figures ───────────────────────────────────────────────────────────────────

def fig1_accuracy(plain, aug):
    """Overall accuracy vs. samples/lang."""
    fig, ax = plt.subplots(figsize=(10, 6))

    if plain:
        x = [p["samples_per_lang"] for p in plain]
        y = [p["overall_accuracy"]  for p in plain]
        ax.plot(x, y, color="#2980b9", linewidth=2.2, marker="o", markersize=8,
                label="XLS-R (no augmentation)", zorder=3)
        for xi, yi, p in zip(x, y, plain):
            ax.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points",
                        xytext=(6, 4), fontsize=8, color="#2980b9")

    if aug:
        x_a = [p["samples_per_lang"] for p in aug]
        y_a = [p["overall_accuracy"]  for p in aug]
        ax.scatter(x_a, y_a, color="#2ecc71", s=120, marker="*", zorder=5,
                   label="XLS-R + Augmentation")
        for xi, yi in zip(x_a, y_a):
            ax.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points",
                        xytext=(6, 4), fontsize=8, color="#27ae60")

    # Whisper zero-shot reference line
    ax.axhline(0.9914, color="gray", linestyle="--", linewidth=1.4, alpha=0.7,
               label="Whisper zero-shot (99.14%) — upper bound")
    ax.axhline(0.8327, color="#95a5a6", linestyle=":", linewidth=1.2, alpha=0.7,
               label="wav2vec2-base (83.27%) — English-only baseline")

    ax.set_xlabel("Training samples per language", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title("Learning Curve: How much data does XLS-R need?\n"
                 "Overall test accuracy vs. training samples per language",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0.7, 1.02)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25, which="both")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "learning_curve_accuracy.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def fig2_per_language(plain, aug):
    """Per-language recall vs. samples/lang with Japanese and Spanish highlighted."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)

    for ax, data, title_suffix in [
        (axes[0], plain, "XLS-R (no augmentation)"),
        (axes[1], aug,   "XLS-R + Augmentation"),
    ]:
        if not data:
            ax.text(0.5, 0.5, "No data yet\n(run training script)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=12)
            ax.set_title(title_suffix, fontsize=11)
            continue

        x_all = [p["samples_per_lang"] for p in data]

        for lang in LANG_LABELS:
            y = [p["per_lang_recall"][lang] for p in data]
            color, lw, marker = HIGHLIGHT.get(lang, DEFAULT_STYLE)
            ax.plot(x_all, y, color=color, linewidth=lw, marker=marker,
                    markersize=6 if lang in HIGHLIGHT else 4,
                    alpha=1.0 if lang in HIGHLIGHT else 0.35,
                    label=lang if lang in HIGHLIGHT else None,
                    zorder=3 if lang in HIGHLIGHT else 1)

        # Annotate Japanese values
        ja_vals = [p["per_lang_recall"]["Japanese"] for p in data]
        for xi, yi in zip(x_all, ja_vals):
            ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points",
                        xytext=(6, 2), fontsize=8,
                        color=HIGHLIGHT["Japanese"][0], fontweight="bold")

        # 50% recall reference
        ax.axhline(0.5, color="black", linestyle=":", linewidth=1, alpha=0.5,
                   label="50% recall threshold")

        ax.set_xlabel("Training samples per language", fontsize=11)
        ax.set_ylabel("Per-language Recall (test set)", fontsize=11)
        ax.set_title(f"Per-language Recall vs. Data Size\n{title_suffix}",
                     fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, alpha=0.25, which="both")

        # Grey out non-highlighted
        grey_patch = mpatches.Patch(color="#aaaaaa", alpha=0.5, label="Other languages")
        handles, labels_ = ax.get_legend_handles_labels()
        ax.legend(handles + [grey_patch], labels_ + ["Other languages"],
                  fontsize=9, loc="lower right")

    fig.suptitle(
        "Japanese Recall: The Key Diagnostic Metric",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "learning_curve_per_language.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def fig3_heatmap(plain, aug):
    """Recall heatmap: language × data size, for plain and augmented separately."""
    all_data = plain + aug
    if not all_data:
        print("  [skip] No data for heatmap")
        return

    fig, axes = plt.subplots(1, 2 if aug else 1,
                             figsize=(14 if aug else 8, 5))
    if not aug:
        axes = [axes, None]

    for ax, data, title in [
        (axes[0], plain, "XLS-R — no augmentation"),
        (axes[1], aug,   "XLS-R — with augmentation") if aug else (None, None, None),
    ]:
        if ax is None or not data:
            continue

        x_vals = [p["samples_per_lang"] for p in data]
        # Matrix: rows=languages, cols=data sizes
        grid = np.array([
            [p["per_lang_recall"][lang] for p in data]
            for lang in LANG_LABELS
        ])
        x_labels = [f"{n:,}" for n in x_vals]

        im = ax.imshow(grid, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="Recall")

        ax.set_xticks(range(len(x_vals)))
        ax.set_xticklabels(x_labels, fontsize=9, rotation=30)
        ax.set_yticks(range(len(LANG_LABELS)))
        ax.set_yticklabels(LANG_LABELS, fontsize=10)
        ax.set_xlabel("Training samples per language", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")

        # Annotate cells
        for i in range(len(LANG_LABELS)):
            for j in range(len(x_vals)):
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                        fontsize=8, color="black" if 0.3 < grid[i, j] < 0.8 else "white")

    fig.suptitle("Per-language Recall Heatmap — at each Training Data Size",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "learning_curve_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  Direction 3: Learning Curve Visualisation")
    print("=" * 60)

    print("\nLoading data points...")
    points = load_known_points() + load_cluster_points()

    if not points:
        print("No data found. Check that results/ has confusion matrix .npy files.")
        return

    plain, aug = build_curve(points)
    print(f"\n  Non-augmented points: {[p['samples_per_lang'] for p in plain]}")
    print(f"  Augmented points:     {[p['samples_per_lang'] for p in aug]}")

    print("\nGenerating figures...")
    fig1_accuracy(plain, aug)
    fig2_per_language(plain, aug)
    fig3_heatmap(plain, aug)

    print("\n" + "=" * 60)
    print("  Done. Figures saved to figures/:")
    print("    learning_curve_accuracy.png")
    print("    learning_curve_per_language.png")
    print("    learning_curve_heatmap.png")
    print()
    print("  To add more data points:")
    print("    1. Run learning_curve_train.py on the cluster for N=100,250,500,1500")
    print("    2. Copy results/learning_curve/*.json here")
    print("    3. Re-run this script")
    print("=" * 60)


if __name__ == "__main__":
    main()
