# Language Identification on FLEURS

Spoken language identification across **8 languages** using the [FLEURS](https://huggingface.co/datasets/google/fleurs) dataset. The repo contains two systems: a classical MFCC-based baseline and a fine-tuned [Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base) model trained with the HuggingFace Trainer API.

---

## Supported Languages

| Code | Language |
|------|----------|
| `cmn_hans_cn` | Mandarin Chinese |
| `ja_jp` | Japanese |
| `en_us` | English |
| `de_de` | German |
| `es_419` | Spanish (Latin America) |
| `it_it` | Italian |
| `ar_eg` | Arabic |
| `sw_ke` | Swahili |

---

## Model Architecture

### Baseline — MFCC + MLP / LinearSVC
- **Features**: 13 MFCC coefficients (C1–C13), mean-pooled over time frames using torchaudio
- **MLP**: two hidden layers [256, 128] with ReLU + Dropout(0.3), trained with Adam + StepLR
- **LinearSVC**: standard SVM on z-scored MFCC features

### Fine-tuned — Wav2Vec2
- **Backbone**: `facebook/wav2vec2-base` (94 M parameters)
- **Fine-tuning**: CNN feature encoder frozen; 12 transformer layers + classification head trained
- **Input**: raw waveform, truncated/padded to 10 s (160 000 samples at 16 kHz)
- **Head**: linear classifier over the mean-pooled last hidden state
- **Training**: HuggingFace `Trainer`, 10 epochs, lr=3e-5, warmup 10%, batch size 8

---

## Results

> Numbers below are on the FLEURS **test** split. Update after training.

| Method | Accuracy | Macro-F1 |
|--------|----------|----------|
| MFCC + LinearSVC (baseline) | — | — |
| MFCC + MLP (baseline) | — | — |
| Wav2Vec2 fine-tuned | — | — |

---

## Training Data

All data is streamed directly from the [google/fleurs](https://huggingface.co/datasets/google/fleurs) HuggingFace dataset — no local downloads required. Standard FLEURS train / validation / test splits are used.

---

## Reproduction

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the MFCC + MLP baseline

```bash
cd src
python nn_baseline.py
```

### 3. Run the MFCC + LinearSVC baseline

```bash
cd src
python svc.py
```

### 4. Fine-tune Wav2Vec2

Log in to HuggingFace first (required for `push_to_hub`):

```bash
huggingface-cli login
```

Then edit `HUB_MODEL_ID` in `src/train_wav2vec2.py` to `<your-username>/wav2vec2-fleurs-lid`, then:

```bash
cd src
python train_wav2vec2.py
```

Training takes ~2–4 hours on a single GPU. Checkpoints are saved to `src/wav2vec2-fleurs-lid/`.

### 5. Inference example

```python
import torch
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

model_id = "<your-username>/wav2vec2-fleurs-lid"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id)
model.eval()

# audio_array: np.ndarray at 16 kHz, mono
def predict_language(audio_array: np.ndarray) -> str:
    inputs = feature_extractor(
        audio_array, sampling_rate=16000, return_tensors="pt", padding=True
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_id = logits.argmax().item()
    return model.config.id2label[predicted_id]
```

---

## HuggingFace Hub

Trained model: [wav2vec2-fleurs-lid](https://huggingface.co/wav2vec2-fleurs-lid) *(link updated after training)*

---

## Project Structure

```
language-identification-fleurs/
├── src/
│   ├── load_data.py          # FLEURS dataset loading utilities
│   ├── extract_mfcc.py       # MFCC feature extraction (torchaudio)
│   ├── nn_baseline.py        # MLP baseline: MFCC → MLP
│   ├── svc.py                # LinearSVC baseline: MFCC → SVC
│   ├── pipeline.py           # Logistic Regression (single-language sanity check)
│   └── train_wav2vec2.py     # Wav2Vec2 fine-tuning via HuggingFace Trainer
├── requirements.txt
└── README.md
```

---

## Citation

```bibtex
@inproceedings{conneau2022fleurs,
  title     = {FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech},
  author    = {Conneau, Alexis and Ma, Min and Khanuja, Simran and Zhang, Yu and
               Axelrod, Vera and Dalmia, Siddharth and Riesa, Jason and
               Rivera, Clara and Bapna, Ankur},
  booktitle = {IEEE Spoken Language Technology Workshop (SLT)},
  year      = {2022}
}
```
