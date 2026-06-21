#!/usr/bin/env python3
"""
Direction 2: Layer-wise Probing — STEP 1: Feature Extraction
=============================================================
Run this script ON THE HPC CLUSTER where the fine-tuned checkpoints live.
It loads each fine-tuned model, runs the FLEURS test set through it with
output_hidden_states=True, and saves mean-pooled per-layer representations
as numpy arrays.

The saved arrays are small enough (~50-300 MB each) to transfer to your local
machine, where layer_probe_visualize.py trains the probes and makes figures.

Usage (on cluster, from the src/ directory):
    python layer_probe_extract.py --model xlsr_augmented
    python layer_probe_extract.py --model xlsr
    python layer_probe_extract.py --model wav2vec2

Output (saved to ../results/layer_probes/):
    <model>_layer_features.npy   shape: (n_layers+1, n_test_samples, hidden_dim)
    <model>_labels.npy           shape: (n_test_samples,)   integer class labels
    <model>_meta.json            n_layers, hidden_dim, label_names
"""

import os
import json
import argparse
import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
)

# ── Model checkpoint paths (edit if yours differ) ─────────────────────────────

CHECKPOINTS = {
    "wav2vec2":       "./wav2vec2-fleurs-lid/checkpoint-4000",
    "xlsr":           "/fs/scratch/users/xiaofg/xlsr-fleurs-lid/checkpoint-8000",
    "xlsr_fulldata":  "/fs/scratch/users/xiaofg/xlsr-fleurs-lid-fulldata/checkpoint-22128",
    "xlsr_augmented": "/fs/scratch/users/xiaofg/xlsr-fleurs-lid-augmented/checkpoint-13830",
}
BASE_MODELS = {
    "wav2vec2":       "facebook/wav2vec2-base",
    "xlsr":           "facebook/wav2vec2-large-xlsr-53",
    "xlsr_augmented": "facebook/wav2vec2-large-xlsr-53",
}

LANGUAGES = ["cmn_hans_cn", "ja_jp", "en_us", "de_de",
             "es_419",      "it_it", "ar_eg", "sw_ke"]
LANG_LABELS = ["Mandarin", "Japanese", "English", "German",
               "Spanish",  "Italian",  "Arabic",  "Swahili"]

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "layer_probes")
os.makedirs(RESULTS_DIR, exist_ok=True)

BATCH_SIZE = 8
MAX_DURATION_S = 20.0   # clip test samples longer than this
SAMPLE_RATE = 16_000


# ── Data loading ──────────────────────────────────────────────────────────────

def load_test_data():
    """Load and concatenate the official FLEURS test split for all 8 languages."""
    print("Loading FLEURS test splits...")
    splits = []
    for lang_id, label_name in zip(LANGUAGES, LANG_LABELS):
        ds = load_dataset("google/fleurs", lang_id, split="test", trust_remote_code=True)
        ds = ds.map(lambda ex: {"label_int": LANGUAGES.index(lang_id)}, batched=False)
        splits.append(ds)
    combined = concatenate_datasets(splits)
    print(f"  Total test samples: {len(combined)}")
    return combined


# ── Extraction ────────────────────────────────────────────────────────────────

def extract_hidden_states(model_name: str):
    ckpt = CHECKPOINTS[model_name]
    base = BASE_MODELS[model_name]

    if not os.path.isdir(ckpt):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt}\n"
            "Make sure you're running this on the cluster where the model was saved."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    print(f"Loading feature extractor from {base}...")
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(base)

    print(f"Loading fine-tuned model from {ckpt}...")
    model = Wav2Vec2ForSequenceClassification.from_pretrained(ckpt)
    model.eval()
    model.to(device)

    n_transformer_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    # hidden_states includes the CNN output (index 0) + one per transformer layer
    n_total_layers = n_transformer_layers + 1
    print(f"  Architecture: {n_transformer_layers} transformer layers, hidden_dim={hidden_dim}")

    dataset = load_test_data()

    all_features = []  # will be shape (n_total_layers, N, hidden_dim)
    all_labels   = []

    # Pre-allocate per-layer lists
    layer_buffers = [[] for _ in range(n_total_layers)]

    print(f"\nExtracting hidden states for {len(dataset)} samples...")
    for start in range(0, len(dataset), BATCH_SIZE):
        batch = dataset.select(range(start, min(start + BATCH_SIZE, len(dataset))))

        # Clip extremely long audio to avoid OOM
        audios = []
        for ex in batch:
            arr = np.array(ex["audio"]["array"], dtype=np.float32)
            max_len = int(MAX_DURATION_S * SAMPLE_RATE)
            audios.append(arr[:max_len])

        inputs = extractor(
            audios,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
            )

        # outputs.hidden_states: tuple of (n_total_layers,) tensors
        # each tensor shape: (batch, time_steps, hidden_dim)
        # We mean-pool over time to get (batch, hidden_dim)
        hidden_states = outputs.hidden_states  # len = n_total_layers
        for layer_idx, hs in enumerate(hidden_states):
            # hs: (batch, T, D) — mask out padding (attention_mask for wav2vec2
            # is on the raw waveform level, not aligned to hidden states,
            # so we use simple mean over all time steps as approximation)
            pooled = hs.mean(dim=1).cpu().numpy()  # (batch, D)
            layer_buffers[layer_idx].append(pooled)

        labels = [ex["label_int"] for ex in batch]
        all_labels.extend(labels)

        if (start // BATCH_SIZE) % 10 == 0:
            print(f"  Processed {start + len(batch)}/{len(dataset)} samples...")

    # Stack per layer: (n_total_layers, N, hidden_dim)
    all_features = np.stack(
        [np.concatenate(buf, axis=0) for buf in layer_buffers],
        axis=0
    )
    all_labels = np.array(all_labels)

    print(f"\nExtracted features shape: {all_features.shape}")
    print(f"Labels shape: {all_labels.shape}")

    # Save
    feat_path  = os.path.join(RESULTS_DIR, f"{model_name}_layer_features.npy")
    label_path = os.path.join(RESULTS_DIR, f"{model_name}_labels.npy")
    meta_path  = os.path.join(RESULTS_DIR, f"{model_name}_meta.json")

    np.save(feat_path,  all_features)
    np.save(label_path, all_labels)
    with open(meta_path, "w") as f:
        json.dump({
            "model_name":          model_name,
            "checkpoint":          ckpt,
            "n_transformer_layers": n_transformer_layers,
            "n_total_layers":      n_total_layers,
            "hidden_dim":          hidden_dim,
            "n_samples":           int(all_labels.shape[0]),
            "label_names":         LANG_LABELS,
        }, f, indent=2)

    print(f"\n  Saved: {feat_path}")
    print(f"  Saved: {label_path}")
    print(f"  Saved: {meta_path}")
    print("\nDone. Transfer the results/layer_probes/ folder to your local machine,")
    print("then run layer_probe_visualize.py to generate the figures.")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model", required=True,
        choices=sorted(CHECKPOINTS.keys()),
        help="Which fine-tuned model to probe."
    )
    args = ap.parse_args()
    extract_hidden_states(args.model)
