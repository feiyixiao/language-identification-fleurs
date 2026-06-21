#!/usr/bin/env python3
"""
Direction 3: Learning Curve — STEP 1: Training (run on HPC cluster)
=====================================================================
Fine-tunes XLS-R on exactly N samples per language and saves test-set
metrics to a small JSON file. Does NOT save model checkpoints to avoid
filling scratch quota.

Run once per data-size point you want on the curve:
    python learning_curve_train.py --samples_per_lang 100
    python learning_curve_train.py --samples_per_lang 250
    python learning_curve_train.py --samples_per_lang 500
    python learning_curve_train.py --samples_per_lang 1500

(We already have results for 1000 and ~2700 from earlier experiments.)

Output: results/learning_curve/<N>_samples.json
  Contains: overall accuracy, macro-F1, per-language recall, confusion matrix.
"""

import os
import json
import argparse
import numpy as np
import torch
from datasets import load_dataset, Audio, concatenate_datasets
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import f1_score, confusion_matrix
import evaluate

# ── Config ────────────────────────────────────────────────────────────────────

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
LANG_LABELS = ["Mandarin", "Japanese", "English", "German",
               "Spanish",  "Italian",  "Arabic",  "Swahili"]

SAMPLING_RATE = 16_000
MAX_AUDIO_LEN = 10 * SAMPLING_RATE      # 10 s, same as xlsr experiments
MODEL_CKPT    = "facebook/wav2vec2-large-xlsr-53"
SCRATCH_DIR   = "/fs/scratch/users/xiaofg/xlsr-learning-curve"
OUTPUT_DIR    = os.path.join(os.path.dirname(__file__), "..", "results", "learning_curve")

label2id = {lang: idx for idx, lang in enumerate(LANGUAGES)}
id2label = {idx: lang for idx, lang in enumerate(LANGUAGES)}

USE_FP16 = torch.cuda.is_available()

# ── Data ──────────────────────────────────────────────────────────────────────

def load_split(split: str, max_per_lang: int = None):
    parts = []
    for lang in LANGUAGES:
        ds = load_dataset("google/fleurs", lang, split=split)
        ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
        if max_per_lang and len(ds) > max_per_lang:
            ds = ds.shuffle(seed=42).select(range(max_per_lang))
        ds = ds.add_column("label", [label2id[lang]] * len(ds))
        parts.append(ds)
    return concatenate_datasets(parts).shuffle(seed=42)

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_CKPT)

def preprocess(example):
    arr = np.array(example["audio"]["array"], dtype=np.float32)
    if len(arr) > MAX_AUDIO_LEN:
        arr = arr[:MAX_AUDIO_LEN]
    elif len(arr) < MAX_AUDIO_LEN:
        arr = np.pad(arr, (0, MAX_AUDIO_LEN - len(arr)))
    inputs = feature_extractor(arr, sampling_rate=SAMPLING_RATE,
                               return_tensors="np", padding=False)
    return {"input_values": inputs.input_values[0], "label": example["label"]}

# ── Metrics ───────────────────────────────────────────────────────────────────

accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
    f1  = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "macro_f1": f1}

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples_per_lang", type=int, required=True,
                    help="Training samples per language (e.g. 100, 250, 500, 1500)")
    args = ap.parse_args()
    N = args.samples_per_lang

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{N}_samples.json")
    if os.path.exists(out_path):
        print(f"Results already exist: {out_path}. Delete to re-run.")
        return

    print(f"\n{'='*60}")
    print(f"  Learning Curve — {N} samples/language")
    print(f"{'='*60}\n")

    print("Loading data...")
    train_ds = load_split("train", max_per_lang=N)
    val_ds   = load_split("validation")
    test_ds  = load_split("test")
    print(f"  train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    print("Preprocessing...")
    cols = train_ds.column_names
    train_ds = train_ds.map(preprocess, remove_columns=cols)
    val_ds   = val_ds.map(preprocess,   remove_columns=val_ds.column_names)
    test_ds  = test_ds.map(preprocess,  remove_columns=test_ds.column_names)
    for ds in [train_ds, val_ds, test_ds]:
        ds.set_format("torch")

    print("Loading model...")
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        MODEL_CKPT,
        num_labels=len(LANGUAGES),
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )
    model.freeze_feature_encoder()

    # Temporary checkpoint dir — deleted after training to save scratch space
    tmp_ckpt = os.path.join(SCRATCH_DIR, f"tmp_{N}")
    os.makedirs(tmp_ckpt, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=tmp_ckpt,
        num_train_epochs=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=50,
        warmup_ratio=0.1,
        learning_rate=1e-5,
        fp16=USE_FP16,
        gradient_checkpointing=True,
        push_to_hub=False,
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    print("Training...")
    trainer.train()

    # ── Evaluate and collect per-language recall ──────────────────────────────
    print("Evaluating on test set...")
    preds_output = trainer.predict(test_ds)
    preds  = np.argmax(preds_output.predictions, axis=-1)
    labels = preds_output.label_ids

    overall_acc = float((preds == labels).mean())
    overall_f1  = float(f1_score(labels, preds, average="macro"))

    cm = confusion_matrix(labels, preds, labels=list(range(len(LANGUAGES))))
    # Per-language recall = diagonal / row sum
    row_sums = cm.sum(axis=1)
    per_lang_recall = {
        LANG_LABELS[i]: float(cm[i, i] / row_sums[i]) if row_sums[i] > 0 else 0.0
        for i in range(len(LANGUAGES))
    }

    results = {
        "samples_per_lang":  N,
        "total_train":       len(train_ds),
        "overall_accuracy":  overall_acc,
        "macro_f1":          overall_f1,
        "per_lang_recall":   per_lang_recall,
        "confusion_matrix":  cm.tolist(),
    }

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nOverall test accuracy : {overall_acc:.4f}")
    print(f"Macro-F1              : {overall_f1:.4f}")
    print("Per-language recall:")
    for lang, rec in per_lang_recall.items():
        flag = "  <-- watch this" if lang == "Japanese" else ""
        print(f"  {lang:<12}: {rec:.3f}{flag}")
    print(f"\nSaved: {out_path}")

    # Clean up temporary checkpoints to free scratch space
    import shutil
    shutil.rmtree(tmp_ckpt, ignore_errors=True)
    print(f"Cleaned up temp checkpoints: {tmp_ckpt}")


if __name__ == "__main__":
    main()
