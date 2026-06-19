#!/usr/bin/env python3
import argparse
import numpy as np
import lang2vec.lang2vec as l2v
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from adjustText import adjust_text

LANGUAGES = ["cmn_hans_cn", "ja_jp", "en_us", "de_de",
             "es_419", "it_it", "ar_eg", "sw_ke"]
LANG_LABELS = ["Mandarin", "Japanese", "English", "German",
               "Spanish", "Italian", "Arabic", "Swahili"]

# lang2vec uses ISO 639-3 codes
L2V_CODES = ["cmn", "jpn", "eng", "deu", "spa", "ita", "ara", "swa"]


def analyze(model_name):
    cm = np.load(f"../results/confusion_matrix_{model_name}.npy")

    # Pairwise symmetric confusion rate: (cm[i,j] + cm[j,i]) / (row_total_i + row_total_j)
    row_totals = cm.sum(axis=1)
    n = len(LANGUAGES)

    confusion_rates = []
    typological_distances = []
    pair_labels = []

    # Get all language vectors at once
    all_feats = l2v.get_features(L2V_CODES, "syntax_wals")
    def to_vec(feat_list):
        return np.array([float(x) if x != '--' else np.nan for x in feat_list])

    vectors = {code: to_vec(all_feats[code]) for code in L2V_CODES}

    for i in range(n):
        for j in range(i + 1, n):
            conf = (cm[i, j] + cm[j, i]) / (row_totals[i] + row_totals[j])

            vec_i = vectors[L2V_CODES[i]]
            vec_j = vectors[L2V_CODES[j]]

            # Use only dimensions where both languages have values
            mask = ~(np.isnan(vec_i) | np.isnan(vec_j))
            if mask.sum() == 0:
                continue
            dist = np.linalg.norm(vec_i[mask] - vec_j[mask]) / np.sqrt(mask.sum())

            confusion_rates.append(conf)
            typological_distances.append(dist)
            pair_labels.append(f"{LANG_LABELS[i]}–{LANG_LABELS[j]}")

    confusion_rates = np.array(confusion_rates)
    typological_distances = np.array(typological_distances)

    rho, p = spearmanr(typological_distances, confusion_rates)
    print(f"[{model_name}] Spearman ρ = {rho:.3f}, p = {p:.3f}")

    # Plot
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.scatter(typological_distances, confusion_rates, color="steelblue", alpha=0.8, s=60, zorder=3)

    texts = [ax.text(x, y, label, fontsize=8)
             for label, x, y in zip(pair_labels, typological_distances, confusion_rates)]
    adjust_text(texts, ax=ax,
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
                expand=(1.5, 1.8))

    ax.set_xlabel("Typological Distance (lang2vec syntax_wals)", fontsize=11)
    ax.set_ylabel("Symmetric Confusion Rate", fontsize=11)
    ax.set_title(f"Typological Distance vs. Confusion Rate — {model_name}\nSpearman ρ = {rho:.3f}, p = {p:.3f}", fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"../results/typological_analysis_{model_name}.png", dpi=150)
    print(f"Saved results/typological_analysis_{model_name}.png")
    plt.close()

    # Print all pairs sorted by confusion rate
    print("\nTop confused pairs:")
    order = np.argsort(confusion_rates)[::-1]
    for idx in order[:10]:
        print(f"  {pair_labels[idx]:30s}  confusion={confusion_rates[idx]:.4f}  dist={typological_distances[idx]:.3f}")

    return rho, p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="wav2vec2",
                     help="Suffix of confusion_matrix_<model>.npy in results/ (e.g. wav2vec2, xlsr, baseline)")
    args = ap.parse_args()
    analyze(args.model)
