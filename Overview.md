# Language Identification on FLEURS — Project Context

## Project overview
8-way spoken language identification (LID) using the FLEURS dataset.
Course: Team Lab Phonetics, University of Stuttgart (IMS), Summer 2026.
Team name: thangvusproteges
Members: Kiran Belgal (st199665) and Feixiang Xiao (st199507)

## Languages
8 typologically diverse languages:
- Mandarin (cmn_hans_cn) — Sino-Tibetan
- Japanese (ja_jp) — Japonic
- English (en_us) — Indo-European, Germanic
- German (de_de) — Indo-European, Germanic
- Spanish (es_419) — Indo-European, Romance
- Italian (it_it) — Indo-European, Romance
- Arabic (ar_eg) — Afro-Asiatic, Semitic
- Swahili (sw_ke) — Niger-Congo, Bantu

## Dataset
FLEURS (Few-shot Learning Evaluation of Universal Representations of Speech)
- 1,000 subsampled training utterances per language (~8,000 total)
- Official val/test splits used in full (speaker-disjoint across all splits)
- Loaded via HuggingFace datasets library
- Key versions: datasets==2.18.0, pyarrow==15.0.0
- All audio: 16kHz mono, 5-15 seconds

## Environment
- Conda environment: teamlab (Python 3.11)
- Key packages: torch, torchaudio, transformers, datasets, librosa, 
  scikit-learn, lang2vec, seaborn, matplotlib, scipy, numpy
- All scripts run from the src/ subdirectory

## Repository structure
Personal code repo: feiyixiao/language-identification-fleurs (GitHub)
- Kiran and Fei are direct collaborators (no forking needed)
- Branch naming: kiran/feature-name or fei/feature-name
- PRs merged into main

Course submission repo: github.tik.uni-stuttgart.de (Tik GitHub)
- Weekly reports submitted as PRs from fork
- Branch naming: thangvusproteges/wN
- PR titles: wN: brief description or M2/M3: brief description
- IMPORTANT: always git checkout main && git pull before creating a new branch
  to avoid merge conflicts (learned from TA feedback)

## File structure (src/)
- load_data.py — loads FLEURS for all 8 languages, 1000 samples per language
- extract_mfcc.py — extracts 13 MFCC coefficients per audio clip using librosa, averaged across time
- pipeline.py — early test pipeline (mostly superseded)
- nn_baseline.py — MLP baseline (PyTorch, torchaudio MFCCs)
- svc.py — LinearSVC baseline (scikit-learn); saves confusion matrix to results/
- train_wav2vec2.py — fine-tunes facebook/wav2vec2-base on all 8 languages
- wav2vec2_confusion_matrix.py — runs inference on test set, saves confusion matrix PNG and .npy
- lang2vec_analysis.py — typological distance vs confusion rate analysis
  - takes --matrix argument: python lang2vec_analysis.py --matrix ../results/confusion_matrix_wav2vec2.npy
  - uses syntax_knn features from lang2vec/URIEL
  - computes Spearman correlation between typological distance and symmetric confusion rate
- attractor_analysis.py — bar chart showing which languages act as misclassification attractors
- EDA.py — exploratory data analysis (sample counts, duration distributions, MFCC profiles)

## Results so far
| Model | Dev Accuracy | Test Accuracy | Macro-F1 |
|-------|-------------|---------------|----------|
| MLP (MFCC) | 91.3% (internal) | 16.85% | 15.71% |
| LinearSVC (MFCC) | 15.09% | 15.42% | 15.00% |
| wav2vec2-base | ~80% | TBD | TBD |

Key finding: MFCC baselines suffer severe speaker overfitting — 91% on internal
split but ~15-17% on official test set with unseen speakers. wav2vec2 generalizes
much better because it operates on raw audio with pretrained speech representations.

## Lang2vec analysis results
LinearSVC confusion matrix:
- Spearman rho = +0.24, p = 0.21 — no significant correlation
- Confusions appear random (model is basically guessing)

wav2vec2-base confusion matrix:
- Spearman rho = -0.32, p = 0.09 — trending negative correlation
- Stronger model begins to reflect linguistic structure in errors
- Still not significant (p > 0.05), likely due to small sample size (only 28 language pairs)

## Key findings and observations
1. Speaker overfitting is the central problem for MFCC baselines
2. Italian→Arabic asymmetric confusion: 192 errors in one direction, ~0 in reverse
   - Very interesting because Italian and Arabic are typologically distant
   - Hypothesis: Arabic and Italian may act as acoustic "attractors"
3. Spanish→Italian confusion (165 errors) makes linguistic sense (both Romance)
4. English→German confusion makes sense (both Germanic)
5. The attractor effect: Italian and Arabic attract many misclassifications from other languages

## Division of labor
- Kiran owns: lang2vec_analysis.py, EDA.py, attractor_analysis.py
- Fei owns: load_data.py, extract_mfcc.py, nn_baseline.py, svc.py, 
  train_wav2vec2.py, wav2vec2_confusion_matrix.py

## Current priorities (as of June 2026)
1. Investigate the Italian–Arabic asymmetric confusion phenomenon
2. Run lang2vec analysis on XLS-R confusion matrix when available
3. Compare typological analysis across all three models (LinearSVC, wav2vec2, XLS-R)
4. Prepare M3 advanced methods report (due June 29)
5. Prepare poster (due July 6) and final report (due July 28)

## Upcoming deadlines
- M3 advanced methods report: June 29, 14:00 (GitHub PR)
- Poster draft: July 6, 14:00 (ILIAS)
- Poster presentation: July 14, in person
- Final written report: July 28, 23:59 (ILIAS, 4-page Interspeech format)

## Notes for Claude Code
- Always activate the teamlab conda environment before running scripts
- Run scripts from the src/ directory
- Data is cached in ~/.cache/huggingface/ — no need to re-download
- results/ and figures/ folders are gitignored for large files
- When generating confusion matrices, always save both PNG and .npy versions
- The .npy file is needed for lang2vec_analysis.py
- lang2vec uses ISO 639-3 codes, not HuggingFace language codes (see HF_TO_ISO dict in lang2vec_analysis.py)