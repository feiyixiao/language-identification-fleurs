"""
lang2vec_analysis.py — Typological Distance vs Model Confusion Analysis
========================================================================
This script implements the advanced direction from our proposal:
"Analyze confusions against typological distance (lang2vec, language family)"

The core question: do languages that are typologically closer get confused
more often by our classifier? If yes, that suggests the model is picking up
on genuine linguistic features. If no, the confusions are driven by something
else (e.g. acoustic recording conditions, speaker characteristics).

Prerequisites:
    - Run svc.py first to generate results/confusion_matrix.npy
    - pip install lang2vec

Usage:
    cd src
    python lang2vec_analysis.py

Outputs:
    - Printed correlation statistics
    - Figures saved to ../figures/
"""

import os
import numpy as np
import lang2vec.lang2vec as l2v
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine

# ── Language metadata ────────────────────────────────────────────────────────
# lang2vec uses ISO 639-3 codes, not HuggingFace codes
# Reference: https://lang2vec.readthedocs.io/
LANGUAGES = [
    "cmn_hans_cn",  # Mandarin
    "ja_jp",        # Japanese
    "en_us",        # English
    "de_de",        # German
    "es_419",       # Spanish
    "it_it",        # Italian
    "ar_eg",        # Arabic
    "sw_ke",        # Swahili
]

# Mapping from HuggingFace codes to ISO 639-3 codes that lang2vec understands
# lang2vec uses WALS/URIEL feature databases which require ISO 639-3
HF_TO_ISO = {
    "cmn_hans_cn": "cmn",  # Mandarin Chinese
    "ja_jp":       "jpn",  # Japanese
    "en_us":       "eng",  # English
    "de_de":       "deu",  # German
    "es_419":      "spa",  # Spanish
    "it_it":       "ita",  # Italian
    "ar_eg":       "arb",  # Arabic (standard)
    "sw_ke":       "swh",  # Swahili
}

LANGUAGE_NAMES = {
    "cmn_hans_cn": "Mandarin",
    "ja_jp":       "Japanese",
    "en_us":       "English",
    "de_de":       "German",
    "es_419":      "Spanish",
    "it_it":       "Italian",
    "ar_eg":       "Arabic",
    "sw_ke":       "Swahili",
}

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(FIGURES_DIR, exist_ok=True)


# ── 1. Load confusion matrix ─────────────────────────────────────────────────
def load_confusion_matrix():
    """
    Load the confusion matrix saved by svc.py.
    
    The confusion matrix is an 8x8 array where cm[i][j] is the number of
    times language i was predicted as language j. The diagonal is correct
    predictions, off-diagonal are errors.
    """
    path = os.path.join(RESULTS_DIR, "confusion_matrix.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "confusion_matrix.npy not found. Run svc.py first!"
        )
    cm = np.load(path)
    print(f"Loaded confusion matrix: {cm.shape}")
    return cm


# ── 2. Compute confusion rates ───────────────────────────────────────────────
def compute_confusion_rates(cm):
    """
    Convert raw confusion counts to symmetric confusion rates.
    
    Why symmetric? Typological distance is symmetric (distance from English
    to German == distance from German to English), so we need to match that.
    We average confusion in both directions: rate(i,j) = (cm[i,j] + cm[j,i]) / 2
    
    We also normalize by the total samples per language so that languages
    with more test samples don't dominate.
    """
    n = cm.shape[0]
    # Normalize each row by total samples for that language
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = cm / row_sums  # now cm_norm[i][j] = P(predict j | true i)

    # Make symmetric: average both directions
    confusion_rates = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                confusion_rates[i][j] = (cm_norm[i][j] + cm_norm[j][i]) / 2

    return confusion_rates


# ── 3. Compute typological distances ─────────────────────────────────────────
def compute_typological_distances():
    """
    Use lang2vec to compute pairwise typological distances between languages.
    
    lang2vec encodes languages as feature vectors from URIEL, a typological
    database covering phonology, syntax, morphology, and geography.
    We use the 'syntax_knn' feature set which captures syntactic typology
    and tends to correlate well with language family structure.
    
    Other feature sets you could try:
    - 'phonology_knn': phonological features (more relevant for LID!)
    - 'geo': geographic proximity
    - 'fam': language family one-hot encoding
    - 'learned': learned embeddings
    
    Cosine distance between feature vectors = typological distance.
    Distance of 0 = identical typological profile.
    Distance of 1 = maximally different.
    """
    iso_codes = [HF_TO_ISO[lang] for lang in LANGUAGES]
    
    print("\nFetching lang2vec features...")
    # syntax_knn uses k-nearest-neighbor imputation for missing values
    features = l2v.get_features(iso_codes, "syntax_knn")
    
    n = len(LANGUAGES)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                vec_i = features[iso_codes[i]]
                vec_j = features[iso_codes[j]]
                distances[i][j] = cosine(vec_i, vec_j)

    return distances


# ── 4. Correlation analysis ──────────────────────────────────────────────────
def correlate(confusion_rates, distances):
    """
    Compute Spearman correlation between typological distance and confusion rate.
    
    Why Spearman (not Pearson)?
    - We care about rank correlation, not linear correlation
    - Typological distances are not normally distributed
    - Spearman is more robust to outliers
    
    Hypothesis: higher typological distance → lower confusion rate
    So we expect a NEGATIVE correlation.
    
    If rho is significantly negative: the model confuses typologically
    similar languages more, suggesting it captures real linguistic structure.
    
    If rho is near zero or positive: confusions are driven by other factors
    (recording conditions, speaker variation, acoustic overlap unrelated
    to typology).
    """
    n = len(LANGUAGES)
    # Extract upper triangle only (avoid counting pairs twice)
    dist_vals = []
    conf_vals = []
    pair_labels = []

    for i in range(n):
        for j in range(i+1, n):
            dist_vals.append(distances[i][j])
            conf_vals.append(confusion_rates[i][j])
            pair_labels.append(
                f"{LANGUAGE_NAMES[LANGUAGES[i]]}–{LANGUAGE_NAMES[LANGUAGES[j]]}"
            )

    dist_vals = np.array(dist_vals)
    conf_vals = np.array(conf_vals)

    rho, pval = spearmanr(dist_vals, conf_vals)

    print("\n" + "="*60)
    print("SPEARMAN CORRELATION: Typological Distance vs Confusion Rate")
    print("="*60)
    print(f"  rho = {rho:.4f}")
    print(f"  p-value = {pval:.4f}")
    if pval < 0.05:
        if rho < 0:
            print("  ✓ Significant NEGATIVE correlation:")
            print("    Typologically closer languages are confused more often.")
            print("    The model captures some real linguistic structure!")
        else:
            print("  ✓ Significant POSITIVE correlation:")
            print("    Typologically distant languages are confused more often.")
            print("    This is unexpected — worth investigating why.")
    else:
        print("  ✗ No significant correlation found (p > 0.05).")
        print("    Confusions may be driven by factors other than typology")
        print("    (e.g. recording conditions, speaker variation).")

    return dist_vals, conf_vals, pair_labels, rho, pval


# ── 5. Visualizations ────────────────────────────────────────────────────────
def plot_typological_distance_matrix(distances):
    """Heatmap of pairwise typological distances."""
    names = [LANGUAGE_NAMES[l] for l in LANGUAGES]
    plt.figure(figsize=(9, 7))
    sns.heatmap(distances, annot=True, fmt=".2f",
                xticklabels=names, yticklabels=names, cmap="YlOrRd")
    plt.title("Pairwise Typological Distances (lang2vec syntax_knn)")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "typological_distances.png")
    plt.savefig(path)
    print(f"\n✓ Saved: {path}")
    plt.close()


def plot_confusion_rates(confusion_rates):
    """Heatmap of symmetric confusion rates."""
    names = [LANGUAGE_NAMES[l] for l in LANGUAGES]
    plt.figure(figsize=(9, 7))
    sns.heatmap(confusion_rates, annot=True, fmt=".3f",
                xticklabels=names, yticklabels=names, cmap="Blues")
    plt.title("Symmetric Confusion Rates (Official Test Split)")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "confusion_rates.png")
    plt.savefig(path)
    print(f"✓ Saved: {path}")
    plt.close()


def plot_scatter(dist_vals, conf_vals, pair_labels, rho, pval):
    """
    Scatter plot: typological distance (x) vs confusion rate (y).
    Each point is a language pair. This is the key figure for the paper.
    """
    plt.figure(figsize=(10, 7))
    plt.scatter(dist_vals, conf_vals, alpha=0.6, s=60)

    # Annotate the most confused pairs
    top_confused = np.argsort(conf_vals)[-5:]
    for idx in top_confused:
        plt.annotate(pair_labels[idx],
                     (dist_vals[idx], conf_vals[idx]),
                     fontsize=7, ha="left",
                     xytext=(5, 5), textcoords="offset points")

    # Trend line
    z = np.polyfit(dist_vals, conf_vals, 1)
    p = np.poly1d(z)
    x_line = np.linspace(dist_vals.min(), dist_vals.max(), 100)
    plt.plot(x_line, p(x_line), "r--", alpha=0.7, label="trend")

    plt.xlabel("Typological Distance (lang2vec cosine)")
    plt.ylabel("Symmetric Confusion Rate")
    plt.title(f"Typological Distance vs Confusion Rate\n"
              f"Spearman rho={rho:.3f}, p={pval:.3f}")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "distance_vs_confusion.png")
    plt.savefig(path)
    print(f"✓ Saved: {path}")
    plt.close()


def print_top_confusions(confusion_rates, distances, n=10):
    """Print the most confused language pairs with their typological distances."""
    pairs = []
    n_langs = len(LANGUAGES)
    for i in range(n_langs):
        for j in range(i+1, n_langs):
            pairs.append((
                confusion_rates[i][j],
                distances[i][j],
                LANGUAGE_NAMES[LANGUAGES[i]],
                LANGUAGE_NAMES[LANGUAGES[j]]
            ))

    pairs.sort(reverse=True)
    print("\n" + "="*60)
    print(f"TOP {n} MOST CONFUSED LANGUAGE PAIRS")
    print("="*60)
    print(f"{'Pair':<30} {'Confusion':>10} {'Typo Dist':>10}")
    print("-"*55)
    for conf, dist, lang1, lang2 in pairs[:n]:
        print(f"{lang1}–{lang2:<20} {conf:>10.4f} {dist:>10.4f}")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Lang2Vec Typological Distance Analysis")
    print("="*60)

    # Load confusion matrix from svc.py results
    cm = load_confusion_matrix()

    # Compute symmetric confusion rates
    confusion_rates = compute_confusion_rates(cm)

    # Compute typological distances
    distances = compute_typological_distances()

    # Print top confusions
    print_top_confusions(confusion_rates, distances)

    # Correlation analysis
    dist_vals, conf_vals, pair_labels, rho, pval = correlate(
        confusion_rates, distances
    )

    # Visualizations
    print("\nGenerating figures...")
    plot_typological_distance_matrix(distances)
    plot_confusion_rates(confusion_rates)
    plot_scatter(dist_vals, conf_vals, pair_labels, rho, pval)

    print("\n" + "="*60)
    print("Analysis complete! Check figures/ for plots.")
    print("="*60)