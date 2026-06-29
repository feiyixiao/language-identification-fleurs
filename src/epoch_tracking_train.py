#!/usr/bin/env python3
"""
Direction 5: Confusion Matrix Evolution — Training Script (run on HPC cluster)
===============================================================================
Re-runs XLS-R fine-tuning (no augmentation) with a custom callback that saves
a confusion matrix on the test set after every epoch. No full checkpoints are
kept — only the per-epoch confusion matrices (tiny .npy files).

Run twice to get both conditions:
    python epoch_tracking_train.py --condition plain
    python epoch_tracking_train.py --condition augmented

Output: results/epoch_tracking/<condition>_epoch_<N>.npy  (one per epoch)
        results/epoch_tracking/<condition>_meta.json

Transfer results/epoch_tracking/ to your local machine, then run:
    python epoch_tracking_visualize.py
"""

import os
import json
import argparse
import random
import numpy as np
import torch
from datasets import load_dataset, Audio, concatenate_datasets
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from sklearn.metrics import f1_score, confusion_matrix
import evaluate

# ── Config ────────────────────────────────────────────────────────────────────

LANGUAGES = [
    "cmn_hans_cn", "ja_jp", "en_us", "de_de",
    "es_419",      "it_it", "ar_eg", "sw_ke",
]
LANG_LABELS = ["Mandarin", "Japanese", "English", "German",
               "Spanish",  "Italian",  "Arabic",  "Swahili"]

SAMPLING_RATE    = 16_000
MAX_SAMPLES      = 10 * SAMPLING_RATE
MAX_TRAIN_PER_LANG = None            # full official split (~2700/lang)
MODEL_CKPT       = "facebook/wav2vec2-large-xlsr-53"
SCRATCH_BASE     = "/fs/scratch/users/xiaofg"
NUM_EPOCHS       = 10
SEED             = 42

label2id = {lang: idx for idx, lang in enumerate(LANGUAGES)}
id2label = {idx: lang for idx, lang in enumerate(LANGUAGES)}

random.seed(SEED)
np.random.seed(SEED)
USE_FP16 = torch.cuda.is_available()

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "epoch_tracking")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Augmentation (only used in --condition augmented) ─────────────────────────

def augment(array: np.ndarray, sr: int = SAMPLING_RATE) -> np.ndarray:
    """Same augmentation pipeline as train_xlsr_augmented.py."""
    # Gain perturbation
    gain = random.uniform(0.7, 1.3)
    array = array * gain

    # Additive Gaussian noise
    if random.random() < 0.5:
        noise = np.random.randn(len(array)).astype(np.float32)
        array = array + random.uniform(0.001, 0.015) * noise

    # Time masking (zero out a random segment)
    if random.random() < 0.5:
        mask_len = int(random.uniform(0.05, 0.20) * len(array))
        start = random.randint(0, max(0, len(array) - mask_len))
        array[start:start + mask_len] = 0.0

    # Speed perturbation via resampling
    if random.random() < 0.5:
        factor = random.uniform(0.9, 1.1)
        old_len = len(array)
        new_len = int(old_len / factor)
        indices = np.linspace(0, old_len - 1, new_len)
        array = np.interp(indices, np.arange(old_len), array).astype(np.float32)
        # Pad or trim back to original length
        if len(array) < old_len:
            array = np.pad(array, (0, old_len - len(array)))
        else:
            array = array[:old_len]

    return np.clip(array, -1.0, 1.0)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_split(split: str, max_per_lang=None):
    parts = []
    for lang in LANGUAGES:
        ds = load_dataset("google/fleurs", lang, split=split)
        ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
        if max_per_lang and len(ds) > max_per_lang:
            ds = ds.shuffle(seed=SEED).select(range(max_per_lang))
        ds = ds.add_column("label", [label2id[lang]] * len(ds))
        parts.append(ds)
    return concatenate_datasets(parts).shuffle(seed=SEED)


feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_CKPT)


def preprocess(example):
    """Static, deterministic preprocessing for val/test — no augmentation,
    regardless of condition. Keeps evaluation comparable across conditions
    and consistent with train_xlsr_augmented.py."""
    arr = np.array(example["audio"]["array"], dtype=np.float32)
    if len(arr) > MAX_SAMPLES:
        arr = arr[:MAX_SAMPLES]
    elif len(arr) < MAX_SAMPLES:
        arr = np.pad(arr, (0, MAX_SAMPLES - len(arr)))
    inputs = feature_extractor(arr, sampling_rate=SAMPLING_RATE,
                               return_tensors="np", padding=False)
    return {"input_values": inputs.input_values[0], "label": example["label"]}


def make_train_transform(use_augmentation: bool):
    """Dynamic transform for train — re-applied fresh every epoch via
    set_transform, so augmentation differs each pass instead of being
    baked in once via .map()."""
    def train_transform(batch):
        out_values = []
        for arr in batch["audio"]:
            a = np.array(arr["array"], dtype=np.float32)
            if len(a) > MAX_SAMPLES:
                a = a[:MAX_SAMPLES]
            elif len(a) < MAX_SAMPLES:
                a = np.pad(a, (0, MAX_SAMPLES - len(a)))
            if use_augmentation:
                a = augment(a)
            inputs = feature_extractor(a, sampling_rate=SAMPLING_RATE,
                                       return_tensors="np", padding=False)
            out_values.append(inputs.input_values[0])
        return {"input_values": out_values, "label": batch["label"]}
    return train_transform


# ── Metrics ───────────────────────────────────────────────────────────────────

accuracy_metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
    f1  = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "macro_f1": f1}


# ── Epoch callback ────────────────────────────────────────────────────────────

class EpochConfusionCallback(TrainerCallback):
    """After every epoch, run full inference on the test set and save a confusion matrix."""

    def __init__(self, trainer_ref, test_dataset, condition: str):
        self.trainer_ref  = trainer_ref
        self.test_dataset = test_dataset
        self.condition    = condition
        self.epoch_num    = 0

    def on_epoch_end(self, args, state, control, **kwargs):
        self.epoch_num += 1
        print(f"\n[EpochConfusionCallback] Epoch {self.epoch_num} — computing test confusion matrix...")

        output = self.trainer_ref.predict(self.test_dataset)
        preds  = np.argmax(output.predictions, axis=-1)
        labels = output.label_ids

        cm = confusion_matrix(labels, preds, labels=list(range(len(LANGUAGES))))
        path = os.path.join(OUT_DIR, f"{self.condition}_epoch_{self.epoch_num:02d}.npy")
        np.save(path, cm)

        # Quick per-language recall printout
        row_totals = cm.sum(axis=1)
        recalls = {LANG_LABELS[i]: float(cm[i, i] / row_totals[i])
                   if row_totals[i] > 0 else 0.0
                   for i in range(len(LANGUAGES))}
        overall = float(np.diag(cm).sum() / cm.sum())
        print(f"  Overall acc: {overall:.3f}  |  "
              f"Japanese: {recalls['Japanese']:.3f}  |  "
              f"Spanish: {recalls['Spanish']:.3f}")
        print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=["plain", "augmented"],
                    help="'plain' = no augmentation (to observe collapse); "
                         "'augmented' = with augmentation (to see if collapse is prevented)")
    args = ap.parse_args()
    condition = args.condition
    use_aug   = (condition == "augmented")

    print(f"\n{'='*60}")
    print(f"  Direction 5: Epoch Tracking — condition: {condition}")
    print(f"{'='*60}\n")

    print("Loading data...")
    train_ds = load_split("train", max_per_lang=MAX_TRAIN_PER_LANG)
    val_ds   = load_split("validation")
    test_ds  = load_split("test")
    print(f"  train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    print("Preprocessing...")
    # train: dynamic augmentation applied fresh every epoch via set_transform
    # (only perturbs audio when condition == "augmented")
    train_ds.set_transform(make_train_transform(use_augmentation=use_aug))
    # val/test: static, deterministic, NEVER augmented — keeps evaluation
    # comparable between the plain and augmented conditions
    val_ds  = val_ds.map(preprocess,  remove_columns=val_ds.column_names)
    test_ds = test_ds.map(preprocess, remove_columns=test_ds.column_names)
    val_ds.set_format("torch")
    test_ds.set_format("torch")

    print("Loading model...")
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        MODEL_CKPT,
        num_labels=len(LANGUAGES),
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )
    model.freeze_feature_encoder()

    tmp_ckpt = os.path.join(SCRATCH_BASE, f"xlsr-epoch-tracking-{condition}")
    os.makedirs(tmp_ckpt, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=tmp_ckpt,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
        save_strategy="no",         # no checkpoints saved — only our confusion matrices
        logging_steps=50,
        warmup_ratio=0.1,
        learning_rate=1e-5,
        fp16=USE_FP16,
        gradient_checkpointing=True,
        push_to_hub=False,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,  # needed: set_transform keeps raw "audio"/"label" columns
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    # Attach the callback (needs trainer reference for .predict())
    callback = EpochConfusionCallback(trainer, test_ds, condition)
    trainer.add_callback(callback)

    print(f"\nTraining ({NUM_EPOCHS} epochs, saving confusion matrix after each)...\n")
    trainer.train()

    # Save metadata
    meta = {
        "condition":    condition,
        "augmented":    use_aug,
        "num_epochs":   NUM_EPOCHS,
        "languages":    LANG_LABELS,
        "num_langs":    len(LANGUAGES),
        "model_ckpt":   MODEL_CKPT,
    }
    meta_path = os.path.join(OUT_DIR, f"{condition}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved metadata: {meta_path}")

    print(f"\nDone. Transfer results/epoch_tracking/ to your local machine")
    print(f"and run: python epoch_tracking_visualize.py")


if __name__ == "__main__":
    main()
