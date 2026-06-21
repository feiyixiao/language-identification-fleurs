#!/usr/bin/env python3
"""
Direction 1: Phonological Distance Analysis
============================================
Compares PHONOLOGICAL distances against TYPOLOGICAL (syntactic) distances
to determine which better predicts model confusion across 8 languages.

Hypothesis: For audio-based language ID, phonological similarity should be
a stronger predictor of model confusion than syntactic typological distance.

Key finding investigated: Japanese and Spanish share a 5-vowel system
{a, e, i, o, u}, which may explain the XLS-R Japanese->Spanish collapse.

Generates 4 figures in figures/:
  phonological_distance_heatmap.png    -- syntactic vs phonological distance matrices
  vowel_inventory_grid.png             -- per-language vowel inventory presence/absence
  phonological_scatter_grid.png        -- phonological distance vs confusion by model
  predictor_rho_comparison.png         -- Spearman rho: syntax vs phonology vs vowel overlap

Run from the src/ directory:
    python phonological_analysis.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import spearmanr
import lang2vec.lang2vec as l2v

# ── Constants ─────────────────────────────────────────────────────────────────

LANGUAGES  = ["cmn_hans_cn", "ja_jp",    "en_us",   "de_de",
              "es_419",      "it_it",    "ar_eg",   "sw_ke"]
LANG_LABELS = ["Mandarin",  "Japanese", "English", "German",
               "Spanish",   "Italian",  "Arabic",  "Swahili"]
L2V_CODES   = ["cmn",       "jpn",      "eng",     "deu",
               "spa",       "ita",      "ara",     "swa"]

MODELS = ["wav2vec2", "xlsr", "xlsr_fulldata", "xlsr_augmented"]
MODEL_DISPLAY = {
    "wav2vec2":       "wav2vec2-base\n(83.3% acc)",
    "xlsr":           "XLS-R 1k/lang\n(87.2% acc)",
    "xlsr_fulldata":  "XLS-R full data\n(89.0% acc)",
    "xlsr_augmented": "XLS-R + Augment\n(93.6% acc)",
}

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Vowel inventories (quality only, length markers stripped).
# Source: PHOIBLE aggregated data / standard IPA references.
# Key observation: Japanese and Spanish share exactly the 5-cardinal vowels.
VOWEL_INVENTORIES = {
    "Mandarin": {"a", "o", "e", "ɤ", "ə", "i", "u", "y", "ɛ", "ɔ"},
    "Japanese": {"a", "e", "i", "o", "u"},
    "English":  {"i", "ɪ", "e", "ɛ", "æ", "ɑ", "ɒ", "ɔ", "ʊ", "u", "ʌ", "ɜ", "ə"},
    "German":   {"i", "ɪ", "e", "ɛ", "a", "ɔ", "o", "ʊ", "u", "y", "ʏ", "ø", "œ", "ə", "ɐ"},
    "Spanish":  {"a", "e", "i", "o", "u"},
    "Italian":  {"a", "e", "ɛ", "i", "o", "ɔ", "u"},
    "Arabic":   {"a", "i", "u"},
    "Swahili":  {"a", "e", "i", "o", "u"},
}

# Ordered IPA vowels for the grid (roughly: front→back, high→low)
VOWEL_ORDER = [
    "i", "ɪ", "y", "ʏ",      # high front
    "e", "ø",                   # upper-mid front/round
    "ɛ", "œ",                   # mid front/round
    "æ",                        # near-low front
    "ə", "ɐ",                   # central mid/near-low
    "ɤ",                        # upper-mid back unround
    "ɜ",                        # mid central
    "a",                        # low
    "ɑ", "ɒ",                   # low back
    "ɔ", "o",                   # mid-upper back round
    "ʊ", "u",                   # high back round
    "ʌ",                        # mid back unround
]

# ── 1. Load confusion matrices ────────────────────────────────────────────────

def load_all_confusion_matrices():
    matrices = {}
    for model in MODELS:
        path = os.path.join(RESULTS_DIR, f"confusion_matrix_{model}.npy")
        if not os.path.exists(path):
            print(f"  [warning] Missing: {path} — skipping {model}")
            continue
        matrices[model] = np.load(path)
        print(f"  Loaded {model}: {matrices[model].shape}")
    return matrices


def symmetric_confusion_rates(cm):
    """Return upper-triangle symmetric confusion rates for all language pairs."""
    row_totals = cm.sum(axis=1)
    n = cm.shape[0]
    rates = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                rates[i, j] = (cm[i, j] + cm[j, i]) / (row_totals[i] + row_totals[j])
    return rates


def upper_triangle_pairs(matrix):
    """Return (values, pair_labels) for the upper triangle of a distance/rate matrix."""
    n = matrix.shape[0]
    vals, labels = [], []
    for i in range(n):
        for j in range(i + 1, n):
            vals.append(matrix[i, j])
            labels.append(f"{LANG_LABELS[i]}–{LANG_LABELS[j]}")
    return np.array(vals), labels


# ── 2. Compute distances ───────────────────────────────────────────────────────

def _l2v_distance_matrix(feature_set):
    """Fetch lang2vec features and compute pairwise normalised Euclidean distance."""
    feats = l2v.get_features(L2V_CODES, feature_set)
    n = len(L2V_CODES)

    def to_vec(code):
        raw = feats[code]
        return np.array([float(x) if x != "--" else np.nan for x in raw])

    vecs = [to_vec(c) for c in L2V_CODES]
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            mask = ~(np.isnan(vecs[i]) | np.isnan(vecs[j]))
            if mask.sum() == 0:
                dist[i, j] = np.nan
            else:
                diff = vecs[i][mask] - vecs[j][mask]
                dist[i, j] = np.linalg.norm(diff) / np.sqrt(mask.sum())
    return dist


def compute_syntactic_distances():
    print("  Fetching lang2vec syntax_wals features...")
    return _l2v_distance_matrix("syntax_wals")


def compute_phonological_distances():
    print("  Fetching lang2vec phonology_knn features...")
    return _l2v_distance_matrix("phonology_knn")


def compute_vowel_jaccard():
    """Pairwise Jaccard SIMILARITY (not distance) between vowel inventories.
    Higher = more vowels in common relative to the union."""
    n = len(LANG_LABELS)
    sim = np.zeros((n, n))
    for i, li in enumerate(LANG_LABELS):
        for j, lj in enumerate(LANG_LABELS):
            if i == j:
                sim[i, j] = 1.0
            else:
                a, b = VOWEL_INVENTORIES[li], VOWEL_INVENTORIES[lj]
                sim[i, j] = len(a & b) / len(a | b)
    # Return as DISTANCE so higher = more different (consistent with other metrics)
    return 1.0 - sim


# ── 3. Correlation analysis ───────────────────────────────────────────────────

def spearman_rho(dist_matrix, conf_rates):
    """Compute Spearman rho between upper-triangle of dist_matrix and conf_rates."""
    d_vals, _ = upper_triangle_pairs(dist_matrix)
    c_vals, _ = upper_triangle_pairs(conf_rates)
    rho, p = spearmanr(d_vals, c_vals)
    return rho, p


def run_correlation_analysis(matrices, syn_dist, phon_dist, vowel_dist):
    print("\n" + "=" * 65)
    print(f"{'Model':<22} {'Predictor':<22} {'rho':>7}  {'p':>7}")
    print("=" * 65)

    results = {}
    for model, cm in matrices.items():
        conf = symmetric_confusion_rates(cm)
        row = {}
        for label, dist in [("Syntactic",  syn_dist),
                             ("Phonological", phon_dist),
                             ("Vowel Jaccard", vowel_dist)]:
            rho, p = spearman_rho(dist, conf)
            sig = "*" if p < 0.05 else ""
            print(f"  {MODEL_DISPLAY[model].split(chr(10))[0]:<20} {label:<22} {rho:>7.3f}  {p:>7.3f} {sig}")
            row[label] = (rho, p)
        results[model] = row
        print()
    print("=" * 65)
    print("  * p < 0.05")
    return results


# ── 4. Visualisations ─────────────────────────────────────────────────────────

def fig1_distance_heatmaps(syn_dist, phon_dist):
    """Side-by-side: syntactic distance vs phonological distance."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax, dist, title in zip(
        axes,
        [syn_dist, phon_dist],
        ["Syntactic Distance\n(lang2vec syntax_wals)",
         "Phonological Distance\n(lang2vec phonology_knn)"]
    ):
        mask = np.eye(len(LANG_LABELS), dtype=bool)
        sns.heatmap(
            dist, annot=True, fmt=".2f",
            xticklabels=LANG_LABELS, yticklabels=LANG_LABELS,
            cmap="YlOrRd", mask=mask, ax=ax,
            vmin=0, vmax=np.nanmax([syn_dist, phon_dist])
        )
        ax.set_title(title, fontsize=13, pad=12)
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)

    fig.suptitle(
        "Pairwise Language Distances: Syntactic vs Phonological",
        fontsize=15, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "phonological_distance_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def fig2_vowel_inventory_grid(vowel_dist):
    """Grid showing which vowels each language has; Japanese-Spanish highlighted."""
    # Build presence/absence matrix
    all_vowels = VOWEL_ORDER  # use our ordered list
    present_in_order = [v for v in all_vowels
                        if any(v in VOWEL_INVENTORIES[l] for l in LANG_LABELS)]

    grid = np.zeros((len(LANG_LABELS), len(present_in_order)))
    for i, lang in enumerate(LANG_LABELS):
        for j, vowel in enumerate(present_in_order):
            grid[i, j] = float(vowel in VOWEL_INVENTORIES[lang])

    fig, axes = plt.subplots(1, 2, figsize=(18, 5),
                             gridspec_kw={"width_ratios": [3, 1]})

    # Left: vowel presence grid
    ax = axes[0]
    cmap = plt.cm.Blues
    im = ax.imshow(grid, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(present_in_order)))
    ax.set_xticklabels(present_in_order, fontsize=11)
    ax.set_yticks(range(len(LANG_LABELS)))
    ax.set_yticklabels(LANG_LABELS, fontsize=11)
    ax.set_xlabel("IPA Vowel (quality, length markers stripped)", fontsize=11)
    ax.set_title("Vowel Inventory Presence/Absence by Language\n"
                 "(ordered: front → back, high → low)", fontsize=12)

    # Annotate inventory size
    for i, lang in enumerate(LANG_LABELS):
        n = len(VOWEL_INVENTORIES[lang])
        ax.text(len(present_in_order) + 0.3, i, f"n={n}",
                va="center", ha="left", fontsize=9, color="dimgray")

    # Highlight Japanese and Spanish rows
    ja_idx = LANG_LABELS.index("Japanese")
    es_idx = LANG_LABELS.index("Spanish")
    for idx, color in [(ja_idx, "red"), (es_idx, "darkorange")]:
        rect = plt.Rectangle((-0.5, idx - 0.5), len(present_in_order), 1,
                              linewidth=2.5, edgecolor=color, facecolor="none",
                              zorder=5)
        ax.add_patch(rect)

    ja_patch = mpatches.Patch(edgecolor="red",     facecolor="none", label="Japanese (5 vowels)")
    es_patch = mpatches.Patch(edgecolor="darkorange", facecolor="none", label="Spanish  (5 vowels)")
    ax.legend(handles=[ja_patch, es_patch], loc="lower right", fontsize=9,
              title="Identical vowel systems", title_fontsize=9)

    # Right: vowel Jaccard SIMILARITY heatmap (1 - distance)
    ax2 = axes[1]
    sim = 1.0 - vowel_dist
    mask = np.eye(len(LANG_LABELS), dtype=bool)
    sns.heatmap(sim, annot=True, fmt=".2f",
                xticklabels=LANG_LABELS, yticklabels=LANG_LABELS,
                cmap="Greens", mask=mask, ax=ax2, vmin=0, vmax=1)
    ax2.set_title("Vowel Jaccard\nSimilarity", fontsize=12)
    ax2.tick_params(axis="x", rotation=45)
    ax2.tick_params(axis="y", rotation=0)

    # Highlight Japanese-Spanish cell
    ax2.add_patch(plt.Rectangle((es_idx, ja_idx), 1, 1,
                                 fill=False, edgecolor="red", lw=3, zorder=5))
    ax2.add_patch(plt.Rectangle((ja_idx, es_idx), 1, 1,
                                 fill=False, edgecolor="red", lw=3, zorder=5))

    fig.suptitle(
        "Vowel Inventory Analysis: Japanese and Spanish share the same 5-vowel system {a, e, i, o, u}",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "vowel_inventory_grid.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def fig3_phonological_scatter_grid(matrices, phon_dist):
    """4-panel scatter (one per model): phonological distance vs confusion rate.
    The Japanese-Spanish pair is highlighted in red on each panel."""
    ja_idx = LANG_LABELS.index("Japanese")
    es_idx = LANG_LABELS.index("Spanish")
    # Identify which upper-triangle index corresponds to ja-es
    idx = 0
    ja_es_pair_idx = None
    for i in range(len(LANG_LABELS)):
        for j in range(i + 1, len(LANG_LABELS)):
            if (i == ja_idx and j == es_idx) or (i == es_idx and j == ja_idx):
                ja_es_pair_idx = idx
            idx += 1

    phon_vals, pair_labels = upper_triangle_pairs(phon_dist)

    n_models = len(matrices)
    ncols = 2
    nrows = (n_models + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 5 * nrows))
    axes = axes.flatten()

    for ax_i, (model, cm) in enumerate(matrices.items()):
        ax = axes[ax_i]
        conf = symmetric_confusion_rates(cm)
        conf_vals, _ = upper_triangle_pairs(conf)

        colors = ["#c0392b" if k == ja_es_pair_idx else "steelblue"
                  for k in range(len(phon_vals))]
        sizes  = [120 if k == ja_es_pair_idx else 45
                  for k in range(len(phon_vals))]

        ax.scatter(phon_vals, conf_vals, c=colors, s=sizes, alpha=0.75, zorder=3)

        # Trend line
        z = np.polyfit(phon_vals, conf_vals, 1)
        x_line = np.linspace(phon_vals.min(), phon_vals.max(), 100)
        ax.plot(x_line, np.poly1d(z)(x_line), "k--", alpha=0.4, lw=1.5)

        # Label top 3 confused pairs
        top = np.argsort(conf_vals)[-3:]
        for k in top:
            ax.annotate(pair_labels[k], (phon_vals[k], conf_vals[k]),
                        fontsize=7, xytext=(4, 4), textcoords="offset points")

        rho, p = spearmanr(phon_vals, conf_vals)
        sig = " *" if p < 0.05 else ""
        title = MODEL_DISPLAY[model].split("\n")[0]
        ax.set_title(f"{title}\nSpearman ρ = {rho:.3f}, p = {p:.3f}{sig}", fontsize=10)
        ax.set_xlabel("Phonological Distance (lang2vec)", fontsize=9)
        ax.set_ylabel("Symmetric Confusion Rate", fontsize=9)
        ax.grid(True, alpha=0.25)

    # Hide any spare panels
    for ax_i in range(len(matrices), len(axes)):
        axes[ax_i].set_visible(False)

    red_dot  = mpatches.Patch(color="#c0392b", label="Japanese–Spanish pair")
    blue_dot = mpatches.Patch(color="steelblue", label="Other pairs")
    fig.legend(handles=[red_dot, blue_dot], loc="lower right", fontsize=10,
               bbox_to_anchor=(0.98, 0.02))

    fig.suptitle("Phonological Distance vs Confusion Rate — by Model",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    path = os.path.join(FIGURES_DIR, "phonological_scatter_grid.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def fig4_predictor_rho_comparison(corr_results):
    """Grouped bar chart: Spearman rho for each predictor x model."""
    predictors = ["Syntactic", "Phonological", "Vowel Jaccard"]
    colors     = ["#3498db",   "#e67e22",       "#2ecc71"]

    model_names    = list(corr_results.keys())
    model_shortnames = [MODEL_DISPLAY[m].split("\n")[0] for m in model_names]

    n_models = len(model_names)
    n_pred   = len(predictors)
    x        = np.arange(n_models)
    width    = 0.25

    fig, ax = plt.subplots(figsize=(13, 6))

    for pi, (pred, color) in enumerate(zip(predictors, colors)):
        rhos = [corr_results[m][pred][0] for m in model_names]
        pvals = [corr_results[m][pred][1] for m in model_names]
        bars = ax.bar(x + (pi - 1) * width, rhos, width,
                      label=pred, color=color, alpha=0.82, edgecolor="white")

        # Asterisk above significant bars
        for bar, pval, rho in zip(bars, pvals, rhos):
            if pval < 0.05:
                y_pos = bar.get_height() + (0.01 if rho >= 0 else -0.04)
                ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                        "*", ha="center", va="bottom", fontsize=14, color="black")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax.set_xticks(x)
    ax.set_xticklabels(model_shortnames, fontsize=11)
    ax.set_ylabel("Spearman ρ (distance vs confusion rate)", fontsize=11)
    ax.set_title(
        "Which distance metric better predicts model confusion?\n"
        "Spearman ρ: Syntactic vs Phonological vs Vowel Inventory Overlap\n"
        "* = p < 0.05",
        fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=11, title="Distance Predictor", title_fontsize=10)
    ax.set_ylim(-0.6, 0.8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "predictor_rho_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 65)
    print("  Direction 1: Phonological Distance Analysis")
    print("=" * 65)

    print("\n[1/4] Loading confusion matrices...")
    matrices = load_all_confusion_matrices()
    if not matrices:
        sys.exit("No confusion matrices found. Run model training scripts first.")

    print("\n[2/4] Computing distance matrices...")
    syn_dist   = compute_syntactic_distances()
    phon_dist  = compute_phonological_distances()
    vowel_dist = compute_vowel_jaccard()

    print("\n[3/4] Correlation analysis (Spearman ρ)...")
    corr_results = run_correlation_analysis(matrices, syn_dist, phon_dist, vowel_dist)

    print("\n[4/4] Generating figures...")
    fig1_distance_heatmaps(syn_dist, phon_dist)
    fig2_vowel_inventory_grid(vowel_dist)
    fig3_phonological_scatter_grid(matrices, phon_dist)
    fig4_predictor_rho_comparison(corr_results)

    print("\n" + "=" * 65)
    print("  Done. Figures saved to figures/:")
    print("    phonological_distance_heatmap.png")
    print("    vowel_inventory_grid.png")
    print("    phonological_scatter_grid.png")
    print("    predictor_rho_comparison.png")
    print("=" * 65)


if __name__ == "__main__":
    main()
