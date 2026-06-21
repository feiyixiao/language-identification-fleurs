#!/usr/bin/env python3
"""
Direction 4: Confusion Asymmetry Analysis — Italian → Arabic
=============================================================
Investigates WHY Italian clips are systematically misclassified as Arabic
(~192 errors), while Arabic clips are almost never misclassified as Italian.

Two-part analysis:
  Part A — Quantify the asymmetry across all models and all language pairs.
  Part B — Extract acoustic features from Italian and Arabic test clips to
            identify which acoustic dimensions make Italian "Arabic-like".

No cluster needed. Runs locally; downloads FLEURS test clips on first run
(cached by HuggingFace datasets after that).

Run from the src/ directory:
    python asymmetry_analysis.py

Generates in figures/:
    asymmetry_confusion_bars.png    -- Italian↔Arabic error counts per model
    asymmetry_acoustic_violins.png  -- acoustic feature distributions per language
    asymmetry_mfcc_profiles.png     -- mean MFCC profiles (Italian, Arabic, others)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from datasets import load_dataset, Audio
import librosa
import warnings
warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────

LANGUAGES = ["cmn_hans_cn", "ja_jp", "en_us", "de_de",
             "es_419",      "it_it", "ar_eg", "sw_ke"]
LANG_LABELS = ["Mandarin", "Japanese", "English", "German",
               "Spanish",  "Italian",  "Arabic",  "Swahili"]

IT_IDX = LANG_LABELS.index("Italian")   # 5
AR_IDX = LANG_LABELS.index("Arabic")    # 6

MODELS = ["wav2vec2", "xlsr", "xlsr_fulldata", "xlsr_augmented"]
MODEL_DISPLAY = {
    "wav2vec2":       "wav2vec2-base\n(83.3%)",
    "xlsr":           "XLS-R 1k/lang\n(87.2%)",
    "xlsr_fulldata":  "XLS-R full data\n(89.0%)",
    "xlsr_augmented": "XLS-R + Augment\n(93.6%)",
}

SR = 16_000
MAX_DUR_S = 15.0
N_MFCC = 13

# Clip counts: detailed analysis only needs Italian + Arabic;
# MFCC profiles use a small subsample of all 8 languages.
MAX_CLIPS_DETAIL = 200   # per language, for violin plots (Italian & Arabic only)
MAX_CLIPS_MFCC   = 100   # per language, for MFCC profile comparison (all 8)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


# ── Part A: Asymmetry quantification ─────────────────────────────────────────

def load_all_cms():
    cms = {}
    for m in MODELS:
        path = os.path.join(RESULTS_DIR, f"confusion_matrix_{m}.npy")
        if os.path.exists(path):
            cms[m] = np.load(path).astype(float)
    return cms


def compute_asymmetry(cms):
    """
    For every ordered language pair (i→j), compute the raw error count
    and the asymmetry ratio:  A(i→j) = cm[i,j] / (cm[i,j] + cm[j,i] + 1e-9)

    Also return, for each model, a full (n,n) asymmetry matrix.
    """
    results = {}
    for model, cm in cms.items():
        it_to_ar = cm[IT_IDX, AR_IDX]
        ar_to_it = cm[AR_IDX, IT_IDX]
        ratio     = it_to_ar / (it_to_ar + ar_to_it + 1e-9)

        # Full asymmetry matrix: A[i,j] = cm[i,j] / (cm[i,j]+cm[j,i]+ε)
        asym = cm / (cm + cm.T + 1e-9)
        np.fill_diagonal(asym, 0.0)

        results[model] = {
            "it_to_ar":    it_to_ar,
            "ar_to_it":    ar_to_it,
            "ratio":       ratio,
            "asym_matrix": asym,
        }
        print(f"  {model:<20}  Italian→Arabic: {it_to_ar:5.0f}  "
              f"Arabic→Italian: {ar_to_it:5.0f}  "
              f"ratio: {ratio:.3f}")
    return results


def fig1_asymmetry_bars(asym_results):
    """Grouped bar chart + full asymmetry heatmap side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6),
                             gridspec_kw={"width_ratios": [1.4, 1]})

    # Left: bar chart of Italian↔Arabic raw counts per model
    ax = axes[0]
    model_names = list(asym_results.keys())
    x = np.arange(len(model_names))
    w = 0.35

    it_ar_vals = [asym_results[m]["it_to_ar"] for m in model_names]
    ar_it_vals = [asym_results[m]["ar_to_it"] for m in model_names]

    bars1 = ax.bar(x - w/2, it_ar_vals, w, label="Italian → Arabic (errors)",
                   color="#e74c3c", alpha=0.85)
    bars2 = ax.bar(x + w/2, ar_it_vals, w, label="Arabic → Italian (errors)",
                   color="#3498db", alpha=0.85)

    for bar, val in zip(list(bars1) + list(bars2),
                        list(it_ar_vals) + list(ar_it_vals)):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_DISPLAY[m].replace("\n", "\n") for m in model_names],
                       fontsize=9)
    ax.set_ylabel("Raw error count (test set)", fontsize=11)
    ax.set_title("Italian↔Arabic Confusion Asymmetry\nacross all models",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # Right: full asymmetry heatmap for best model (xlsr_augmented)
    ax2 = axes[1]
    best_model = "xlsr_augmented" if "xlsr_augmented" in asym_results else model_names[-1]
    asym_mat = asym_results[best_model]["asym_matrix"]
    mask = np.eye(len(LANG_LABELS), dtype=bool)
    sns.heatmap(
        asym_mat, annot=True, fmt=".2f",
        xticklabels=LANG_LABELS, yticklabels=LANG_LABELS,
        cmap="RdBu_r", center=0.5, vmin=0, vmax=1,
        mask=mask, ax=ax2, linewidths=0.3,
    )
    # Highlight the Italian-Arabic cell
    ax2.add_patch(plt.Rectangle((AR_IDX, IT_IDX), 1, 1,
                                 fill=False, edgecolor="gold", lw=3, zorder=5))
    ax2.set_title(f"Asymmetry Matrix — {MODEL_DISPLAY[best_model].split(chr(10))[0]}\n"
                  f"Cell(i,j) = P(error goes i→j | either direction)\n"
                  f"Gold box = Italian→Arabic",
                  fontsize=9, fontweight="bold")
    ax2.tick_params(axis="x", rotation=45)
    ax2.tick_params(axis="y", rotation=0)

    fig.suptitle("Directional Confusion Asymmetry: Italian → Arabic is Systematic",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "asymmetry_confusion_bars.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Part B: Acoustic feature extraction ───────────────────────────────────────

def extract_features(audio_array, sr=SR):
    """Extract a dict of acoustic features from a single audio clip."""
    arr = np.array(audio_array, dtype=np.float32)
    # Clip to max duration
    arr = arr[:int(MAX_DUR_S * sr)]
    if len(arr) == 0:
        return None

    feats = {}

    # MFCCs (mean over time)
    mfcc = librosa.feature.mfcc(y=arr, sr=sr, n_mfcc=N_MFCC)
    for i in range(N_MFCC):
        feats[f"mfcc_{i+1}"] = float(np.mean(mfcc[i]))

    # Spectral features
    feats["spectral_centroid"] = float(np.mean(
        librosa.feature.spectral_centroid(y=arr, sr=sr)))
    feats["spectral_rolloff"]  = float(np.mean(
        librosa.feature.spectral_rolloff(y=arr, sr=sr, roll_percent=0.85)))
    feats["spectral_bandwidth"] = float(np.mean(
        librosa.feature.spectral_bandwidth(y=arr, sr=sr)))

    # Zero crossing rate (roughness / consonant density proxy)
    feats["zcr"] = float(np.mean(librosa.feature.zero_crossing_rate(arr)))

    # RMS energy
    feats["rms"] = float(np.mean(librosa.feature.rms(y=arr)))

    # F0 / pitch — using yin (faster than pyin, sufficient for statistics)
    try:
        f0 = librosa.yin(arr, fmin=60, fmax=400, sr=sr)
        # yin returns 0 for unvoiced frames; filter them out
        voiced_f0 = f0[f0 > 60]
        if len(voiced_f0) > 0:
            feats["f0_mean"]      = float(np.mean(voiced_f0))
            feats["f0_std"]       = float(np.std(voiced_f0))
            feats["f0_range"]     = float(np.ptp(voiced_f0))
            feats["voiced_ratio"] = float(len(voiced_f0) / len(f0))
        else:
            feats["f0_mean"] = feats["f0_std"] = feats["f0_range"] = 0.0
            feats["voiced_ratio"] = 0.0
    except Exception:
        feats["f0_mean"] = feats["f0_std"] = feats["f0_range"] = 0.0
        feats["voiced_ratio"] = 0.0

    return feats


def extract_mfcc_only(audio_array, sr=SR):
    """Lightweight extraction for all-language MFCC profile comparison."""
    arr = np.array(audio_array, dtype=np.float32)[:int(MAX_DUR_S * sr)]
    if len(arr) == 0:
        return None
    mfcc = librosa.feature.mfcc(y=arr, sr=sr, n_mfcc=N_MFCC)
    return {f"mfcc_{i+1}": float(np.mean(mfcc[i])) for i in range(N_MFCC)}


def load_audio_features(lang_id, lang_label, max_clips=None):
    """Load FLEURS test clips for one language and extract acoustic features."""
    print(f"  Loading {lang_label} ({lang_id}) test set...")
    ds = load_dataset("google/fleurs", lang_id, split="test", trust_remote_code=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=SR))

    if max_clips:
        ds = ds.select(range(min(max_clips, len(ds))))

    all_feats = []
    for i, ex in enumerate(ds):
        if i % 20 == 0:
            print(f"    {lang_label}: {i}/{len(ds)} clips...", end="\r")
        f = extract_features(ex["audio"]["array"])
        if f is not None:
            f["language"] = lang_label
            all_feats.append(f)
    print(f"    {lang_label}: {len(all_feats)} clips extracted.       ")
    return all_feats


def fig2_acoustic_violins(feature_data):
    """Violin plots of key acoustic features: Italian vs Arabic."""
    import pandas as pd
    df = pd.DataFrame(feature_data)

    # Subset to Italian and Arabic
    df_sub = df[df["language"].isin(["Italian", "Arabic"])]

    features_to_plot = [
        ("f0_mean",           "Mean F0 (Hz)\n[pitch level]"),
        ("f0_std",            "F0 Std Dev (Hz)\n[pitch variability]"),
        ("f0_range",          "F0 Range (Hz)\n[pitch excursion]"),
        ("spectral_centroid", "Spectral Centroid (Hz)\n[brightness]"),
        ("spectral_rolloff",  "Spectral Rolloff (Hz)\n[spectral energy distribution]"),
        ("zcr",               "Zero Crossing Rate\n[consonant density / roughness]"),
        ("voiced_ratio",      "Voiced Frame Ratio\n[vowel-heaviness]"),
        ("rms",               "RMS Energy\n[loudness]"),
    ]

    n_feats = len(features_to_plot)
    ncols = 4
    nrows = (n_feats + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    axes = axes.flatten()

    palette = {"Italian": "#e74c3c", "Arabic": "#3498db"}

    for ax_i, (feat, label) in enumerate(features_to_plot):
        ax = axes[ax_i]
        if feat not in df_sub.columns:
            ax.set_visible(False)
            continue

        sns.violinplot(data=df_sub, x="language", y=feat, ax=ax,
                       palette=palette, inner="box", cut=0.5, width=0.7)

        # Mann-Whitney U test
        it_vals = df_sub[df_sub["language"] == "Italian"][feat].dropna()
        ar_vals = df_sub[df_sub["language"] == "Arabic"][feat].dropna()
        if len(it_vals) > 0 and len(ar_vals) > 0:
            _, p = stats.mannwhitneyu(it_vals, ar_vals, alternative="two-sided")
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
            ax.set_title(f"{label}\n{sig} (p={p:.3f})", fontsize=9)
        else:
            ax.set_title(label, fontsize=9)

        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(axis="y", alpha=0.25)

    for ax_i in range(len(features_to_plot), len(axes)):
        axes[ax_i].set_visible(False)

    fig.suptitle(
        "Acoustic Feature Distributions: Italian vs Arabic\n"
        "Which features make Italian clips acoustically 'Arabic-like'?\n"
        "* p<0.05  ** p<0.01  *** p<0.001  (Mann-Whitney U)",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "asymmetry_acoustic_violins.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def fig3_mfcc_profiles(feature_data):
    """Mean MFCC profile for each language — shows where Italian and Arabic overlap."""
    import pandas as pd
    df = pd.DataFrame(feature_data)

    mfcc_cols = [f"mfcc_{i}" for i in range(1, N_MFCC + 1)]
    x = np.arange(1, N_MFCC + 1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: mean MFCC profiles for all languages
    ax = axes[0]
    highlight = {"Italian": ("#e74c3c", 2.5), "Arabic": ("#3498db", 2.5)}
    for lang in df["language"].unique():
        sub = df[df["language"] == lang][mfcc_cols].mean().values
        color, lw = highlight.get(lang, ("#cccccc", 1.0))
        alpha = 1.0 if lang in highlight else 0.35
        ax.plot(x, sub, color=color, linewidth=lw, alpha=alpha,
                label=lang, marker="o" if lang in highlight else None,
                markersize=4, zorder=3 if lang in highlight else 1)

    ax.set_xlabel("MFCC Coefficient", fontsize=11)
    ax.set_ylabel("Mean Value", fontsize=11)
    ax.set_title("Mean MFCC Profile — All Languages\n"
                 "Italian (red) vs Arabic (blue) vs others (grey)",
                 fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.25)
    ax.axhline(0, color="black", linewidth=0.5)

    # Right: MFCC difference (Italian - Arabic) with ±1 std band
    ax2 = axes[1]
    it_data = df[df["language"] == "Italian"][mfcc_cols]
    ar_data = df[df["language"] == "Arabic"][mfcc_cols]

    diff_mean = it_data.mean().values - ar_data.mean().values
    # Combined SE band
    se = np.sqrt((it_data.std().values**2 / len(it_data)) +
                 (ar_data.std().values**2 / len(ar_data)))

    colors = ["#2ecc71" if abs(d) < se_i else "#e74c3c"
              for d, se_i in zip(diff_mean, se)]
    ax2.bar(x, diff_mean, color=colors, alpha=0.75, edgecolor="white", width=0.7)
    ax2.fill_between(x, -se, se, alpha=0.2, color="gray",
                     label="±1 SE (non-significant zone)")
    ax2.axhline(0, color="black", linewidth=0.8)

    # Annotate direction
    for xi, val in zip(x, diff_mean):
        if abs(val) > 2:
            ax2.text(xi, val + (0.5 if val > 0 else -0.5), f"{val:.1f}",
                     ha="center", va="bottom" if val > 0 else "top",
                     fontsize=8, color="#c0392b" if val > 0 else "#2980b9")

    ax2.set_xlabel("MFCC Coefficient", fontsize=11)
    ax2.set_ylabel("Italian mean − Arabic mean", fontsize=11)
    ax2.set_title("MFCC Profile Difference: Italian − Arabic\n"
                  "Red bars = Italian > Arabic  |  where they converge = source of confusion",
                  fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.25)

    fig.suptitle("MFCC Profiles: Identifying Acoustic Overlap Between Italian and Arabic",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "asymmetry_mfcc_profiles.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def print_summary(feature_data):
    """Print a clean numerical summary to the terminal."""
    import pandas as pd
    df = pd.DataFrame(feature_data)

    feats = ["f0_mean", "f0_std", "voiced_ratio",
             "spectral_centroid", "zcr"]
    feat_names = ["F0 mean (Hz)", "F0 std (Hz)", "Voiced ratio",
                  "Spectral centroid (Hz)", "ZCR"]

    print("\n" + "="*65)
    print(f"  {'Feature':<26} {'Italian':>12} {'Arabic':>12} {'p-value':>10}")
    print("="*65)
    for feat, name in zip(feats, feat_names):
        if feat not in df.columns:
            continue
        it_v = df[df["language"] == "Italian"][feat].dropna()
        ar_v = df[df["language"] == "Arabic"][feat].dropna()
        if len(it_v) == 0 or len(ar_v) == 0:
            continue
        _, p = stats.mannwhitneyu(it_v, ar_v, alternative="two-sided")
        sig = " *" if p < 0.05 else "  "
        print(f"  {name:<26} {it_v.mean():>12.2f} {ar_v.mean():>12.2f} "
              f"{p:>10.3f}{sig}")
    print("="*65)
    print("  * Mann-Whitney U p < 0.05")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  Direction 4: Italian → Arabic Asymmetry Analysis")
    print("="*60)

    # Part A
    print("\n[Part A] Quantifying asymmetry across all models...")
    cms = load_all_cms()
    asym_results = compute_asymmetry(cms)
    fig1_asymmetry_bars(asym_results)

    # Part B — two-tier loading to keep runtime manageable
    # Tier 1: Full acoustic features for Italian + Arabic only (violin plots)
    # Tier 2: MFCCs only for all 8 languages (profile comparison)
    print("\n[Part B] Extracting acoustic features from FLEURS test clips...")
    print(f"  Detailed (Italian + Arabic): up to {MAX_CLIPS_DETAIL} clips each")
    print(f"  MFCC-only (all 8 languages): up to {MAX_CLIPS_MFCC} clips each")
    print("  (First run downloads data; subsequent runs use local cache.)\n")

    detail_langs = [("it_it", "Italian"), ("ar_eg", "Arabic")]
    detail_data = []
    for lang_id, lang_label in detail_langs:
        clips = load_audio_features(lang_id, lang_label, max_clips=MAX_CLIPS_DETAIL)
        detail_data.extend(clips)

    print_summary(detail_data)

    # Tier 2: MFCCs only for all 8 languages
    print("\n  Extracting MFCCs for all 8 languages (profile comparison)...")
    mfcc_data = []
    for lang_id, lang_label in zip(LANGUAGES, LANG_LABELS):
        print(f"    {lang_label}...", end=" ", flush=True)
        ds = load_dataset("google/fleurs", lang_id, split="test", trust_remote_code=True)
        ds = ds.cast_column("audio", Audio(sampling_rate=SR))
        ds = ds.select(range(min(MAX_CLIPS_MFCC, len(ds))))
        for ex in ds:
            f = extract_mfcc_only(ex["audio"]["array"])
            if f is not None:
                f["language"] = lang_label
                mfcc_data.append(f)
        print(f"{len(ds)} clips")

    print("\nGenerating figures...")
    fig2_acoustic_violins(detail_data)
    fig3_mfcc_profiles(mfcc_data)

    print("\n" + "="*60)
    print("  Done. Figures saved to figures/:")
    print("    asymmetry_confusion_bars.png")
    print("    asymmetry_acoustic_violins.png")
    print("    asymmetry_mfcc_profiles.png")
    print("="*60)


if __name__ == "__main__":
    main()
