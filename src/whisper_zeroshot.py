#!/usr/bin/env python3
# Zero-shot language ID using Whisper's built-in language detection — no fine-tuning.
# Whisper's encoder + one decoder step predicts a probability over its ~99 known
# languages; we restrict that to our 8 candidate FLEURS languages for a fair comparison.

import argparse
import os
import numpy as np
import torch
import whisper
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset, Audio, concatenate_datasets
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

LANGUAGES = [
    "cmn_hans_cn", "ja_jp", "en_us", "de_de",
    "es_419", "it_it", "ar_eg", "sw_ke",
]
LANG_LABELS = ["Mandarin", "Japanese", "English", "German",
               "Spanish", "Italian", "Arabic", "Swahili"]

# Whisper's internal language codes (ISO 639-1), not FLEURS codes
WHISPER_CODES = ["zh", "ja", "en", "de", "es", "it", "ar", "sw"]

SAMPLING_RATE = 16_000

label2id = {lang: idx for idx, lang in enumerate(LANGUAGES)}
code2id  = {code: idx for idx, code in enumerate(WHISPER_CODES)}


def main(model_size: str, n_per_lang: int):
    print("Loading test set...")
    parts = []
    for lang in LANGUAGES:
        print(f"  {lang}...")
        ds = load_dataset("google/fleurs", lang, split="test", trust_remote_code=True)
        ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
        if n_per_lang and len(ds) > n_per_lang:
            ds = ds.shuffle(seed=42).select(range(n_per_lang))
        ds = ds.add_column("label", [label2id[lang]] * len(ds))
        parts.append(ds)
    test_ds = concatenate_datasets(parts)
    print(f"Total test samples: {len(test_ds)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Whisper-{model_size} on {device}...")
    model = whisper.load_model(model_size, device=device)

    print("\nRunning zero-shot language detection...")
    all_preds, all_labels = [], []

    for i, sample in enumerate(test_ds):
        if i % 200 == 0:
            print(f"  {i}/{len(test_ds)}")

        array = np.array(sample["audio"]["array"], dtype=np.float32)
        audio = whisper.pad_or_trim(array)
        mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(device)

        _, probs = model.detect_language(mel)
        # restrict to our 8 candidate languages for a fair 8-way comparison
        restricted = {c: probs.get(c, 0.0) for c in WHISPER_CODES}
        pred_code = max(restricted, key=restricted.get)

        all_preds.append(code2id[pred_code])
        all_labels.append(sample["label"])

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    print(f"\nTest Accuracy : {acc:.4f}")
    print(f"Test Macro-F1 : {f1:.4f}")

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(8)))
    os.makedirs("../results", exist_ok=True)
    np.save("../results/confusion_matrix_whisper.npy", cm)
    print("Saved results/confusion_matrix_whisper.npy")

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LANG_LABELS, yticklabels=LANG_LABELS)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix — Whisper-{model_size} (zero-shot, test set)")
    plt.tight_layout()
    plt.savefig("../results/confusion_matrix_whisper.png", dpi=150)
    print("Saved results/confusion_matrix_whisper.png")

    with open("../results/results_whisper.txt", "w") as f:
        f.write(f"Method: Whisper-{model_size} (zero-shot language detection, no fine-tuning)\n")
        f.write(f"Languages: {', '.join(LANGUAGES)}\n")
        f.write(f"Test Accuracy: {acc:.4f}\n")
        f.write(f"Test Macro-F1: {f1:.4f}\n")
        f.write(f"N per language (test): {n_per_lang if n_per_lang else 'full'}\n")
    print("Saved results/results_whisper.txt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_size", type=str, default="base",
                     choices=["tiny", "base", "small", "medium", "large"])
    ap.add_argument("--n_per_lang", type=int, default=0,
                     help="Subsample N utterances per language for a quick test (0 = full test set)")
    args = ap.parse_args()
    main(args.model_size, args.n_per_lang)
