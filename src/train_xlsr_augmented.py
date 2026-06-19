#!/usr/bin/env python3
# Fine-tune XLS-R on 8 FLEURS languages, with on-the-fly audio augmentation.
#
# Rationale: we found that XLS-R collapses Japanese into Spanish (ja_jp -> es_419,
# 64% of the time) under limited fine-tuning data, while a zero-shot model (Whisper)
# with much richer pretraining does not make this mistake. We hypothesize XLS-R is
# relying on coarse acoustic cues (vowel quality, speech rhythm/tempo) that happen
# to be similar between Japanese and Spanish, rather than more robust phonological
# features. Augmentation perturbs exactly those coarse cues (amplitude, missing
# segments, tempo) so the model can't over-rely on them.
#
# Augmentation is applied uniformly to ALL 8 languages (not just Japanese) — applying
# it only to Japanese would create a new spurious correlation between "perturbed audio"
# and the Japanese label, which is the opposite of what we want.
#
# Run: python train_xlsr_augmented.py

import os
import random
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
MAX_SAMPLES             = 10 * SAMPLING_RATE   # 10 s -> 160 000 samples
MAX_TRAIN_PER_LANG      = None                 # full official train split (~2700-2800/lang)
MODEL_CKPT              = "facebook/wav2vec2-large-xlsr-53"
OUTPUT_DIR              = "/fs/scratch/users/xiaofg/xlsr-fleurs-lid-augmented"
HUB_MODEL_ID            = "xlsr-fleurs-lid-augmented"
SEED                    = 42

label2id = {lang: idx for idx, lang in enumerate(LANGUAGES)}
id2label = {idx: lang for idx, lang in enumerate(LANGUAGES)}

random.seed(SEED)
np.random.seed(SEED)

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
            ds = ds.shuffle(seed=SEED).select(range(max_per_lang))
        ds = ds.add_column("label", [label2id[lang]] * len(ds))
        parts.append(ds)
    return concatenate_datasets(parts).shuffle(seed=SEED)


# ── Augmentation (train split only, applied uniformly to all languages) ────────

def augment_waveform(array: np.ndarray, sr: int = SAMPLING_RATE) -> np.ndarray:
    """
    Randomly perturb amplitude, missing-segment robustness, and tempo.
    Each augmentation targets a coarse acoustic cue the model might be
    shortcutting on, rather than learning robust phonological structure.
    """
    array = array.copy()

    # 1. Random gain
    if random.random() < 0.5:
        array = array * random.uniform(0.7, 1.3)

    # 2. Additive Gaussian noise at a random SNR
    if random.random() < 0.5:
        snr_db = random.uniform(10, 30)
        sig_power = np.mean(array ** 2) + 1e-10
        noise_power = sig_power / (10 ** (snr_db / 10))
        array = array + np.random.normal(0, np.sqrt(noise_power), size=array.shape).astype(np.float32)

    # 3. Time masking — zero out 1-2 short random segments
    if random.random() < 0.5:
        for _ in range(random.randint(1, 2)):
            mask_len = random.randint(int(0.05 * sr), int(0.3 * sr))
            if len(array) > mask_len:
                start = random.randint(0, len(array) - mask_len)
                array[start:start + mask_len] = 0.0

    # 4. Speed perturbation — directly perturbs tempo/rhythm, the cue we
    #    hypothesize XLS-R may be over-relying on for the ja_jp/es_419 pair
    if random.random() < 0.5:
        speed = random.uniform(0.9, 1.1)
        orig_len = len(array)
        new_len = max(1, int(orig_len / speed))
        array = np.interp(
            np.linspace(0, orig_len - 1, new_len),
            np.arange(orig_len),
            array,
        ).astype(np.float32)

    return array.astype(np.float32)


def fit_length(array: np.ndarray) -> np.ndarray:
    if len(array) > MAX_SAMPLES:
        return array[:MAX_SAMPLES]
    elif len(array) < MAX_SAMPLES:
        return np.pad(array, (0, MAX_SAMPLES - len(array)))
    return array


feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_CKPT)


def preprocess(example):
    """Static preprocessing for val/test (no augmentation, deterministic)."""
    array = fit_length(np.array(example["audio"]["array"], dtype=np.float32))
    inputs = feature_extractor(array, sampling_rate=SAMPLING_RATE, return_tensors="np", padding=False)
    return {"input_values": inputs.input_values[0], "label": example["label"]}


def train_transform(batch):
    """
    Dynamic, on-the-fly transform for the train split — re-applied (with fresh
    random augmentation) every time examples are fetched, so each epoch sees a
    different perturbed version rather than one fixed augmented copy baked in once.
    """
    input_values = []
    for audio in batch["audio"]:
        array = np.array(audio["array"], dtype=np.float32)
        array = augment_waveform(array)
        array = fit_length(array)
        feat = feature_extractor(array, sampling_rate=SAMPLING_RATE, return_tensors="np", padding=False)
        input_values.append(feat.input_values[0])
    return {"input_values": input_values, "label": batch["label"]}


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
    print("=== Loading datasets ===")
    train_ds = load_split("train",      max_per_lang=MAX_TRAIN_PER_LANG)
    val_ds   = load_split("validation")
    test_ds  = load_split("test")
    print(f"\nSizes — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    print("\n=== Preprocessing ===")
    # train: dynamic augmentation applied fresh every epoch via set_transform
    train_ds.set_transform(train_transform)
    # val/test: static, deterministic, no augmentation — keeps evaluation comparable
    # to the non-augmented runs
    val_ds  = val_ds.map(preprocess,  remove_columns=val_ds.column_names)
    test_ds = test_ds.map(preprocess, remove_columns=test_ds.column_names)
    val_ds.set_format("torch")
    test_ds.set_format("torch")

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

    print("\n=== Training ===")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_dir="/fs/scratch/users/xiaofg/logs_xlsr_augmented",
        logging_steps=50,
        warmup_ratio=0.1,
        learning_rate=1e-5,
        fp16=USE_FP16,
        gradient_checkpointing=True,
        push_to_hub=False,
        hub_model_id=HUB_MODEL_ID,
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

    trainer.train()

    print("\n=== Test set evaluation ===")
    val_results  = trainer.evaluate(val_ds)
    test_results = trainer.evaluate(test_ds)

    val_acc  = val_results["eval_accuracy"]
    val_f1   = val_results["eval_macro_f1"]
    test_acc = test_results["eval_accuracy"]
    test_f1  = test_results["eval_macro_f1"]

    print(f"Val  accuracy : {val_acc:.4f} | macro-F1 : {val_f1:.4f}")
    print(f"Test accuracy : {test_acc:.4f} | macro-F1 : {test_f1:.4f}")

    os.makedirs("../results", exist_ok=True)
    with open("../results/results_xlsr_augmented.txt", "w") as f:
        f.write("Method: XLS-R (wav2vec2-large-xlsr-53, fine-tuned, full train split + audio augmentation)\n")
        f.write(f"Languages: {', '.join(LANGUAGES)}\n")
        f.write(f"Val Accuracy: {val_acc:.4f}\n")
        f.write(f"Val Macro-F1: {val_f1:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test Macro-F1: {test_f1:.4f}\n")
        f.write("Epochs: 10\n")
        f.write("Learning rate: 1e-5\n")
        f.write("Batch size: 8\n")
        f.write("Train samples per language: full official split (~2700-2800)\n")
        f.write("Augmentation: random gain, additive noise, time masking, speed perturb (applied to all languages)\n")
    print("Saved results/results_xlsr_augmented.txt")
