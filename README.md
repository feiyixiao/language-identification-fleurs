# Language Identification on FLEURS

Course project for Intro to Deep Learning (Stuttgart, WS 2025). The task is to identify which of 8 languages a spoken audio clip is in, using the [FLEURS](https://huggingface.co/datasets/google/fleurs) dataset.

Two approaches are compared: a simple MFCC feature baseline and fine-tuning [wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base).

## Languages

| Code | Language |
|------|----------|
| `cmn_hans_cn` | Mandarin Chinese |
| `ja_jp` | Japanese |
| `en_us` | English |
| `de_de` | German |
| `es_419` | Spanish |
| `it_it` | Italian |
| `ar_eg` | Arabic |
| `sw_ke` | Swahili |

## Approach

**Baseline** (`nn_baseline.py`, `svc.py`): extract 13 MFCC coefficients from each audio clip, then train either an MLP or a LinearSVC on top.

**Main model** (`train_wav2vec2.py`): fine-tune `facebook/wav2vec2-base` directly on raw audio. The CNN feature encoder is frozen; only the transformer layers and classification head are trained.

## Results

Evaluated on the official FLEURS test split.

| Method | Accuracy | Macro-F1 |
|--------|----------|----------|
| MFCC + LinearSVC | — | — |
| MFCC + MLP | 16.85% | 15.71% |
| wav2vec2 fine-tuned | — | — |

*The MLP baseline overfits to training speakers — it gets 91% on an internal split but drops to ~17% on unseen speakers. wav2vec2 results coming after training finishes.*

## How to run

```bash
pip install -r requirements.txt
```

MFCC baseline:
```bash
cd src
python nn_baseline.py   # MLP
python svc.py           # LinearSVC
```

Fine-tune wav2vec2 (~2 hours on MPS/GPU):
```bash
cd src
python train_wav2vec2.py
```

Inference:
```python
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
import torch

model_id = "feiyixiao/wav2vec2-fleurs-lid"
extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id)
model.eval()

inputs = extractor(audio_array, sampling_rate=16000, return_tensors="pt")
with torch.no_grad():
    pred = model(**inputs).logits.argmax().item()
print(model.config.id2label[pred])
```

## Files

```
src/
├── load_data.py        load FLEURS subsets
├── extract_mfcc.py     MFCC feature extraction
├── nn_baseline.py      MLP classifier
├── svc.py              LinearSVC classifier
└── train_wav2vec2.py   wav2vec2 fine-tuning
```
