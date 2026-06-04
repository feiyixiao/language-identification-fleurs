#!/usr/bin/env python3
# Generate confusion matrix for the fine-tuned XLS-R model on the test set.

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset, Audio, concatenate_datasets
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

LANGUAGES = [
    "cmn_hans_cn", "ja_jp", "en_us", "de_de",
    "es_419", "it_it", "ar_eg", "sw_ke",
]
LANG_LABELS = ["Mandarin", "Japanese", "English", "German",
               "Spanish", "Italian", "Arabic", "Swahili"]

SAMPLING_RATE = 16_000
MAX_SAMPLES   = 10 * SAMPLING_RATE
MODEL_CKPT    = "facebook/wav2vec2-large-xlsr-53"
CHECKPOINT    = "/fs/scratch/users/xiaofg/xlsr-fleurs-lid/checkpoint-8000"  # best epoch

label2id = {lang: idx for idx, lang in enumerate(LANGUAGES)}

# ── Load test set ─────────────────────────────────────────────────────────────
print("Loading test set...")
parts = []
for lang in LANGUAGES:
    print(f"  {lang}...")
    ds = load_dataset("google/fleurs", lang, split="test", trust_remote_code=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
    ds = ds.add_column("label", [label2id[lang]] * len(ds))
    parts.append(ds)
test_ds = concatenate_datasets(parts)

# ── Load model ────────────────────────────────────────────────────────────────
print(f"\nLoading model from {CHECKPOINT}...")
extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_CKPT)
model     = Wav2Vec2ForSequenceClassification.from_pretrained(CHECKPOINT)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps"  if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)

# ── Run inference ─────────────────────────────────────────────────────────────
print("\nRunning inference on test set...")
all_preds, all_labels = [], []

for i, sample in enumerate(test_ds):
    if i % 200 == 0:
        print(f"  {i}/{len(test_ds)}")

    array = np.array(sample["audio"]["array"], dtype=np.float32)
    if len(array) > MAX_SAMPLES:
        array = array[:MAX_SAMPLES]
    elif len(array) < MAX_SAMPLES:
        array = np.pad(array, (0, MAX_SAMPLES - len(array)))

    inputs = extractor(array, sampling_rate=SAMPLING_RATE,
                       return_tensors="pt", padding=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
    pred = logits.argmax(dim=-1).item()

    all_preds.append(pred)
    all_labels.append(sample["label"])

# ── Metrics ───────────────────────────────────────────────────────────────────
acc = accuracy_score(all_labels, all_preds)
f1  = f1_score(all_labels, all_preds, average="macro")
print(f"\nTest Accuracy : {acc:.4f}")
print(f"Test Macro-F1 : {f1:.4f}")

# ── Confusion matrix ──────────────────────────────────────────────────────────
cm = confusion_matrix(all_labels, all_preds)

os.makedirs("../results", exist_ok=True)
np.save("../results/confusion_matrix_xlsr.npy", cm)
print("Saved results/confusion_matrix_xlsr.npy")

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=LANG_LABELS, yticklabels=LANG_LABELS)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix — XLS-R (wav2vec2-large-xlsr-53, test set)")
plt.tight_layout()
plt.savefig("../results/xlsr_confusion_matrix.png", dpi=150)
print("Saved results/xlsr_confusion_matrix.png")
