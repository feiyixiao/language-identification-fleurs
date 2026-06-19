#!/usr/bin/env python3
"""Compare per-language precision/recall/F1 across all available models."""
import numpy as np
import glob
import os

LANGUAGES = ["cmn_hans_cn", "ja_jp", "en_us", "de_de",
             "es_419", "it_it", "ar_eg", "sw_ke"]
LANG_LABELS = ["Mandarin", "Japanese", "English", "German",
               "Spanish", "Italian", "Arabic", "Swahili"]


def per_class_metrics(cm):
    n = cm.shape[0]
    precision, recall, f1 = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(n):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0.0
    return precision, recall, f1


if __name__ == "__main__":
    paths = sorted(glob.glob("../results/confusion_matrix_*.npy"))
    models = {}
    for p in paths:
        name = os.path.basename(p).replace("confusion_matrix_", "").replace(".npy", "")
        models[name] = np.load(p)

    print("Models found:", list(models.keys()))
    print()

    header = f"{'Language':<10}" + "".join(f"{m + ' F1':>14}" for m in models)
    print(header)
    print("-" * len(header))

    all_f1 = {m: [] for m in models}
    for idx, lang in enumerate(LANG_LABELS):
        row = f"{lang:<10}"
        for m, cm in models.items():
            _, _, f1 = per_class_metrics(cm)
            row += f"{f1[idx]:>14.3f}"
            all_f1[m].append(f1[idx])
        print(row)

    print("-" * len(header))
    row = f"{'Macro avg':<10}"
    for m in models:
        row += f"{np.mean(all_f1[m]):>14.3f}"
    print(row)

    # Biggest movers between wav2vec2 and xlsr if both present
    if "wav2vec2" in models and "xlsr" in models:
        print("\n=== wav2vec2 -> XLS-R per-language F1 change ===")
        _, _, f1_w2v = per_class_metrics(models["wav2vec2"])
        _, _, f1_xlsr = per_class_metrics(models["xlsr"])
        deltas = sorted(zip(LANG_LABELS, f1_w2v, f1_xlsr, f1_xlsr - f1_w2v),
                         key=lambda x: x[3])
        for lang, f1a, f1b, d in deltas:
            sign = "+" if d >= 0 else ""
            print(f"  {lang:<10} {f1a:.3f} -> {f1b:.3f}  ({sign}{d:.3f})")
