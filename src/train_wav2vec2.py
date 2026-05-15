#!/usr/bin/env python3
# Fine-tune wav2vec2-base on 8 FLEURS languages for language identification.
# Run: python train_wav2vec2.py

import os
import numpy as np
import torch
from datasets import load_dataset, Audio, concatenate_datasets
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import f1_score
import evaluate

# --- config ---

LANGUAGES = [
    "cmn_hans_cn",  # Mandarin Chinese
    "ja_jp",        # Japanese
    "en_us",        # English
    "de_de",        # German
    "es_419",       # Spanish (Latin America)
    "it_it",        # Italian
    "ar_eg",        # Arabic
    "sw_ke",        # Swahili
]

SAMPLING_RATE           = 16_000
MAX_SAMPLES             = 10 * SAMPLING_RATE   # 10 s → 160 000 samples
MAX_TRAIN_PER_LANG      = 1000                 # match MLP baseline for fair comparison
MODEL_CKPT              = "facebook/wav2vec2-base"
OUTPUT_DIR              = "./wav2vec2-fleurs-lid"
# Set to "<your-hf-username>/wav2vec2-fleurs-lid" before running
HUB_MODEL_ID            = "wav2vec2-fleurs-lid"

label2id = {lang: idx for idx, lang in enumerate(LANGUAGES)}
id2label = {idx: lang for idx, lang in enumerate(LANGUAGES)}

# MPS (Apple Silicon) does not support fp16
USE_FP16 = torch.cuda.is_available()
if torch.backends.mps.is_available():
    print("MPS device detected — fp16 disabled, using MPS backend")
elif torch.cuda.is_available():
    print("CUDA device detected — fp16 enabled")
else:
    print("No GPU detected — training on CPU")

# ── Data loading ──────────────────────────────────────────────────────────────

def load_split(split: str, max_per_lang: int = None):
    """Load and concatenate all 8 languages for a given FLEURS split."""
    parts = []
    for lang in LANGUAGES:
        print(f"  {lang} [{split}]...")
        ds = load_dataset("google/fleurs", lang, split=split)
        ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
        if max_per_lang and len(ds) > max_per_lang:
            ds = ds.shuffle(seed=42).select(range(max_per_lang))
        ds = ds.add_column("label", [label2id[lang]] * len(ds))
        parts.append(ds)
    return concatenate_datasets(parts).shuffle(seed=42)


# ── Preprocessing ─────────────────────────────────────────────────────────────

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_CKPT)


def preprocess(example):
    """Truncate/zero-pad to MAX_SAMPLES, extract input_values."""
    array = np.array(example["audio"]["array"], dtype=np.float32)

    if len(array) > MAX_SAMPLES:
        array = array[:MAX_SAMPLES]
    elif len(array) < MAX_SAMPLES:
        array = np.pad(array, (0, MAX_SAMPLES - len(array)))

    inputs = feature_extractor(
        array,
        sampling_rate=SAMPLING_RATE,
        return_tensors="np",
        padding=False,
    )
    return {
        "input_values": inputs.input_values[0],
        "label": example["label"],
    }


# ── Metrics ───────────────────────────────────────────────────────────────────

accuracy_metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    f1  = f1_score(labels, predictions, average="macro")
    return {"accuracy": acc, "macro_f1": f1}


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Load data
    print("=== Loading datasets ===")
    train_ds = load_split("train",      max_per_lang=MAX_TRAIN_PER_LANG)
    val_ds   = load_split("validation")
    test_ds  = load_split("test")
    print(f"\nSizes — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    # 2. Preprocess (fixed-length feature extraction)
    print("\n=== Preprocessing ===")
    train_ds = train_ds.map(preprocess, remove_columns=train_ds.column_names)
    val_ds   = val_ds.map(preprocess,   remove_columns=val_ds.column_names)
    test_ds  = test_ds.map(preprocess,  remove_columns=test_ds.column_names)

    train_ds.set_format("torch")
    val_ds.set_format("torch")
    test_ds.set_format("torch")

    # 3. Load model
    print("\n=== Loading model ===")
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        MODEL_CKPT,
        num_labels=len(LANGUAGES),
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )
    model.freeze_feature_encoder()

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params — total: {total:,} | trainable: {trainable:,}")

    # 4. Train
    print("\n=== Training ===")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_dir="./logs",
        logging_steps=50,
        warmup_ratio=0.1,
        learning_rate=3e-5,
        fp16=USE_FP16,
        push_to_hub=False,          # set True + set HUB_MODEL_ID to push
        hub_model_id=HUB_MODEL_ID,
        report_to="none",
        dataloader_num_workers=0,   # 0 for MPS/CPU compatibility
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # 5. Evaluate on test set
    print("\n=== Test set evaluation ===")
    val_results  = trainer.evaluate(val_ds)
    test_results = trainer.evaluate(test_ds)

    val_acc  = val_results["eval_accuracy"]
    val_f1   = val_results["eval_macro_f1"]
    test_acc = test_results["eval_accuracy"]
    test_f1  = test_results["eval_macro_f1"]

    print(f"Val  accuracy : {val_acc:.4f} | macro-F1 : {val_f1:.4f}")
    print(f"Test accuracy : {test_acc:.4f} | macro-F1 : {test_f1:.4f}")

    # 6. Save results
    os.makedirs("../results", exist_ok=True)
    with open("../results/results_wav2vec2.txt", "w") as f:
        f.write("Method: Wav2Vec2ForSequenceClassification (fine-tuned)\n")
        f.write(f"Languages: {', '.join(LANGUAGES)}\n")
        f.write(f"Val Accuracy: {val_acc:.4f}\n")
        f.write(f"Val Macro-F1: {val_f1:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test Macro-F1: {test_f1:.4f}\n")
        f.write("Epochs: 10\n")
        f.write("Learning rate: 3e-5\n")
        f.write("Batch size: 8\n")
    print("Saved results/results_wav2vec2.txt")

    # 7. Push to Hub (uncomment when HUB_MODEL_ID is configured)
    # trainer.push_to_hub()
    # print(f"Pushed to: https://huggingface.co/{HUB_MODEL_ID}")
