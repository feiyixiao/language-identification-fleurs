"""
eda.py — Exploratory Data Analysis for FLEURS Language Identification
======================================================================
Run this script BEFORE building the baseline model to understand your data.
It builds on load_data.py and extract_mfcc.py.

What this script checks:
1. Sample counts per language per split (are classes balanced?)
2. Audio duration distribution (how long are the clips?)
3. MFCC feature distributions (do languages look different in feature space?)
4. Language family breakdown (useful context for typological analysis later)

Usage:
    cd src
    python eda.py

Outputs:
    - Printed stats in terminal
    - Figures saved to ../figures/ folder (created automatically)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from extract_mfcc import extract_mfcc

# ── Language metadata ────────────────────────────────────────────────────────
# This maps HuggingFace language codes to human-readable names and language
# families. Useful for the typological analysis in the advanced phase.
LANGUAGE_META = {
    "cmn_hans_cn": {"name": "Mandarin",  "family": "Sino-Tibetan",    "branch": "Sinitic"},
    "ja_jp":       {"name": "Japanese",  "family": "Japonic",         "branch": "—"},
    "en_us":       {"name": "English",   "family": "Indo-European",   "branch": "Germanic"},
    "de_de":       {"name": "German",    "family": "Indo-European",   "branch": "Germanic"},
    "es_419":      {"name": "Spanish",   "family": "Indo-European",   "branch": "Romance"},
    "it_it":       {"name": "Italian",   "family": "Indo-European",   "branch": "Romance"},
    "ar_eg":       {"name": "Arabic",    "family": "Afro-Asiatic",    "branch": "Semitic"},
    "sw_ke":       {"name": "Swahili",   "family": "Niger-Congo",     "branch": "Bantu"},
}

LANGUAGES = list(LANGUAGE_META.keys())
SPLITS = ["train", "validation", "test"]
SAMPLES_PER_LANG = 1000  # must match load_data.py

# Output folder for figures
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


# ── 1. Sample counts per language per split ──────────────────────────────────
def analyze_splits():
    """
    Check how many samples exist per language in each split.
    
    Why this matters:
    - Class imbalance can bias the classifier toward majority classes.
    - FLEURS is designed to be balanced, but worth verifying for your subset.
    - You subsample 1000 train utterances — check val/test are untouched.
    """
    print("\n" + "="*60)
    print("1. SAMPLE COUNTS PER LANGUAGE PER SPLIT")
    print("="*60)
    print(f"{'Language':<12} {'Train':>8} {'Val':>8} {'Test':>8}")
    print("-"*40)

    train_counts = []
    for lang in LANGUAGES:
        counts = []
        for split in SPLITS:
            ds = load_dataset("google/fleurs", lang, split=split, trust_remote_code=True)
            counts.append(len(ds))
        name = LANGUAGE_META[lang]["name"]
        print(f"{name:<12} {counts[0]:>8} {counts[1]:>8} {counts[2]:>8}")
        train_counts.append(counts[0])

    # Flag imbalance if any language has >20% more samples than another
    if max(train_counts) / min(train_counts) > 1.2:
        print("\n⚠ WARNING: Class imbalance detected in training set.")
        print("  Consider reporting Macro-F1 in addition to accuracy.")
    else:
        print("\n✓ Classes are roughly balanced.")


# ── 2. Audio duration distribution ──────────────────────────────────────────
def analyze_durations(n_samples=100):
    """
    Plot the distribution of audio clip lengths per language.

    Why this matters:
    - FLEURS clips range from ~5-15 seconds.
    - Your MFCC extraction averages over time (np.mean), so very short
      clips might produce noisier features than longer ones.
    - This is directly relevant to the 'effect of utterance duration'
      research direction in your proposal.
    - n_samples: only check first N samples per language for speed.
    """
    print("\n" + "="*60)
    print("2. AUDIO DURATION DISTRIBUTION")
    print("="*60)

    all_durations = {}
    for lang in LANGUAGES:
        ds = load_dataset("google/fleurs", lang, split="train", trust_remote_code=True)
        ds = ds.select(range(min(n_samples, len(ds))))
        durations = [
            len(s["audio"]["array"]) / s["audio"]["sampling_rate"]
            for s in ds
        ]
        all_durations[lang] = durations
        name = LANGUAGE_META[lang]["name"]
        print(f"{name:<12}  mean={np.mean(durations):.1f}s  "
              f"min={np.min(durations):.1f}s  max={np.max(durations):.1f}s")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    names = [LANGUAGE_META[l]["name"] for l in LANGUAGES]
    data = [all_durations[l] for l in LANGUAGES]
    ax.boxplot(data, labels=names)
    ax.set_title("Audio Duration Distribution per Language")
    ax.set_ylabel("Duration (seconds)")
    ax.set_xlabel("Language")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "duration_distribution.png")
    plt.savefig(path)
    print(f"\n✓ Saved figure: {path}")
    plt.close()


# ── 3. MFCC feature distributions ───────────────────────────────────────────
def analyze_mfcc_distributions(n_samples=200):
    """
    Plot the mean value of each MFCC coefficient per language.

    Why this matters:
    - If languages have clearly different MFCC profiles, a simple
      classifier should work well.
    - If distributions overlap heavily, you may need more powerful features
      (e.g. delta MFCCs, mel spectrograms) in the advanced phase.
    - This is your first sanity check that the features are discriminative.
    """
    print("\n" + "="*60)
    print("3. MFCC FEATURE DISTRIBUTIONS")
    print("="*60)

    mean_mfccs = {}
    for lang in LANGUAGES:
        ds = load_dataset("google/fleurs", lang, split="train", trust_remote_code=True)
        ds = ds.select(range(min(n_samples, len(ds))))
        mfccs = np.array([
            extract_mfcc(s["audio"]["array"], s["audio"]["sampling_rate"])
            for s in ds
        ])
        mean_mfccs[lang] = mfccs.mean(axis=0)
        name = LANGUAGE_META[lang]["name"]
        print(f"{name:<12}  MFCC means: {np.round(mean_mfccs[lang], 1)}")

    # Plot mean MFCC profile per language
    fig, ax = plt.subplots(figsize=(12, 5))
    for lang in LANGUAGES:
        ax.plot(mean_mfccs[lang], label=LANGUAGE_META[lang]["name"], marker="o")
    ax.set_title("Mean MFCC Coefficients per Language")
    ax.set_xlabel("MFCC Coefficient Index")
    ax.set_ylabel("Mean Value")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "mfcc_profiles.png")
    plt.savefig(path)
    print(f"\n✓ Saved figure: {path}")
    plt.close()


# ── 4. Language family breakdown ─────────────────────────────────────────────
def analyze_language_families():
    """
    Print a summary of language families in your dataset.

    Why this matters:
    - Your advanced direction involves typological analysis using lang2vec.
    - Having 5 Indo-European languages out of 8 total means the classifier
      might struggle to distinguish within that family.
    - This is the linguistic framing you'll need for your final report.
    """
    print("\n" + "="*60)
    print("4. LANGUAGE FAMILY BREAKDOWN")
    print("="*60)
    print(f"{'Language':<12} {'Family':<20} {'Branch'}")
    print("-"*50)

    families = {}
    for lang, meta in LANGUAGE_META.items():
        print(f"{meta['name']:<12} {meta['family']:<20} {meta['branch']}")
        families[meta["family"]] = families.get(meta["family"], 0) + 1

    print("\nFamily counts:")
    for family, count in sorted(families.items(), key=lambda x: -x[1]):
        print(f"  {family}: {count} language(s)")

    # Key observation for your project
    indo_european = families.get("Indo-European", 0)
    total = len(LANGUAGE_META)
    print(f"\n⚠ Note: {indo_european}/{total} languages are Indo-European.")
    print("  English, German, Spanish, and Italian share Germanic/Romance roots.")
    print("  Expect higher confusion within this group — worth analyzing in M2/M3.")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("FLEURS Dataset — Exploratory Data Analysis")
    print("This may take a few minutes. Data loads from local cache.")

    analyze_language_families()  # Fast, no data loading needed
    analyze_splits()             # Loads all splits — takes ~1-2 min
    analyze_durations()          # Loads train split per language
    analyze_mfcc_distributions() # Extracts MFCCs — takes ~2-3 min

    print("\n" + "="*60)
    print("EDA complete. Check the ../figures/ folder for plots.")
    print("="*60)