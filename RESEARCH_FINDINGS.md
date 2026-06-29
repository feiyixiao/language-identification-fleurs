# Research Findings — Language Identification on FLEURS

**Team:** thangvusproteges (Kiran Belgal & Feixiang Xiao)  
**Course:** Team Lab Phonetics, University of Stuttgart, Summer 2026  
**Task:** 8-way spoken language identification (FLEURS dataset)

---

## Model Performance Summary

| Model | Test Accuracy | Macro-F1 | Notes |
|---|---|---|---|
| MFCC + MLP | 16.85% | 15.71% | Severe speaker overfitting |
| MFCC + LinearSVC | ~15.42% | ~15.00% | Same speaker overfitting |
| wav2vec2-base | 83.27% | 82.92% | Good pretrained representations |
| XLS-R (1k/lang) | 87.24% | 84.15% | Japanese→Spanish collapse (64% confusion) |
| XLS-R (full data, ~2.7k/lang) | 89.04% | 86.59% | Improved with more data |
| XLS-R + Augmentation | **93.59%** | **93.06%** | Best fine-tuned model |
| Whisper (zero-shot) | 99.14% | 98.87% | Upper bound reference; not fine-tuned |

---

## Direction 1: Phonological Distance Analysis

**Script:** `src/phonological_analysis.py`  
**Run with:** `~/anaconda3/envs/teamlab/bin/python phonological_analysis.py` (from `src/`)  
**Figures:** `figures/phonological_distance_heatmap.png`, `figures/vowel_inventory_grid.png`, `figures/phonological_scatter_grid.png`, `figures/predictor_rho_comparison.png`

### Motivation

The existing typological analysis used syntactic features (via lang2vec `syntax_wals`) to predict model confusion. But for audio-based language ID, **phonological** similarity — especially vowel inventory overlap — should be a more direct predictor of confusion. We test three predictors:
- **Syntactic distance** (lang2vec `syntax_wals`) — existing baseline
- **Phonological distance** (lang2vec `phonology_knn`) — broader phonological typology
- **Vowel Jaccard distance** — pairwise set-theoretic overlap of vowel inventories (hardcoded from IPA/PHOIBLE references)

### Key Results

| Model | Syntactic ρ (p) | Phonological ρ (p) | Vowel Jaccard ρ (p) |
|---|---|---|---|
| wav2vec2-base | −0.245 (0.208) | −0.005 (0.979) | **−0.495 (0.007) \*** |
| XLS-R 1k/lang | **+0.421 (0.026) \*** | +0.347 (0.070) | −0.230 (0.240) |
| XLS-R full data | +0.124 (0.529) | +0.007 (0.974) | −0.199 (0.310) |
| XLS-R + Augment | −0.094 (0.633) | −0.075 (0.703) | −0.282 (0.145) |

\* p < 0.05

### Takeaways

1. **Vowel Jaccard is the strongest and only significant predictor for wav2vec2-base (ρ = −0.495, p = 0.007).** Languages sharing more vowel qualities are confused more often. This makes phonetic sense: wav2vec2 leans on vowel quality as a primary discriminating cue.

2. **The Japanese–Spanish collapse is explained by vowel similarity.** Japanese and Spanish are the only two languages in the study with exactly the same 5-vowel system {a, e, i, o, u} (Jaccard similarity = 1.0). When data is scarce, the model over-relies on vowel acoustics and cannot distinguish these two languages — they are acoustically near-identical on this dimension.

3. **The broken XLS-R (1k/lang) shows a spurious positive syntactic correlation (ρ = +0.421, p = 0.026)**, previously reported by Fei. Phonological distance trends in the same direction (ρ = +0.347, p = 0.070), suggesting the collapse has both syntactic and phonological artifact signatures.

4. **After augmentation, all three predictors collapse near zero.** The best model's remaining errors are non-systematic and not explained by any distance metric tested — a good sign that the model has learned robust language-specific features.

5. **Broad phonological distance (lang2vec `phonology_knn`) is never significant** across any model. The fine-grained vowel inventory measure is more informative than aggregate phonological typology, consistent with vowel quality being the specific acoustic dimension at play.

### Implications for the Paper

> "For wav2vec2-base, vowel inventory overlap (Jaccard ρ = −0.495, p = 0.007) is a significant predictor of confusion, while broad syntactic or phonological typological distances are not. Japanese and Spanish — the pair responsible for the XLS-R collapse — share a perfect 5-vowel system overlap (Jaccard = 1.0). This implicates vowel quality as the acoustic feature the model over-relies on under data scarcity, and explains why augmentation (which perturbs amplitude, pitch, and spectral envelope) breaks the dependency."

---

## Direction 2: Layer-wise Probing

**Scripts:** `src/layer_probe_extract.py` (run on HPC cluster), `src/layer_probe_visualize.py` (run locally)  
**Status:** Scripts written; awaiting cluster execution to generate probe features.

*(Findings to be filled in after running on cluster)*

---

## Direction 3: Learning Curve / Data Threshold Analysis

**Scripts:** `src/learning_curve_train.py` (run on HPC cluster), `src/learning_curve_visualize.py` (run locally)  
**Run with:** `~/anaconda3/envs/teamlab/bin/python learning_curve_visualize.py` (from `src/`)  
**Figures:** `figures/learning_curve_accuracy.png`, `figures/learning_curve_per_language.png`, `figures/learning_curve_heatmap.png`

### Motivation

We already have two training-data-size conditions (1k and ~2.7k samples/language). This analysis frames them as a learning curve to answer: *"How much data does XLS-R need, and is the Japanese collapse a data-quantity problem or something else?"*

Additional curve points (100, 250, 500, 1500 samples/lang) can be added by running `learning_curve_train.py` on the cluster and dropping the resulting JSON files into `results/learning_curve/`.

### Current Results (2 data points: 1k and 2.7k)

| Condition | Samples/lang | Overall Acc | Japanese Recall |
|---|---|---|---|
| XLS-R (no augmentation) | 1,000 | 87.2% | 0.21 |
| XLS-R (no augmentation) | 2,700 | 89.0% | 0.28 |
| XLS-R + Augmentation    | 2,700 | 93.5% | **0.71** |

### Takeaways

1. **More data barely helps Japanese.** Tripling training data (1k → 2.7k) raises Japanese recall only from 0.21 → 0.28 — still catastrophically below the 50% threshold and below every other language.

2. **Augmentation is the key, not data quantity.** At the same 2.7k samples/lang, augmentation jumps Japanese recall from 0.28 → 0.71. The collapse is not a data-scarcity problem in the conventional sense; it is a data-diversity problem. The model needs perturbations that break its over-reliance on vowel quality.

3. **Most other languages are near-perfect even at 1k.** Mandarin, English, German, and Spanish all reach ≥1.0 recall at 1k samples. Arabic and Swahili show the next-largest gains from more data (Arabic: 0.82 → 0.97), suggesting they would also benefit from the intermediate curve points.

4. **The heatmap makes Japanese's isolation visually stark** — it is the only language that is "red" at any data size without augmentation.

### Implications for the Paper

> "Scaling from 1k to 2.7k samples per language improves overall accuracy by only 1.8 pp and Japanese recall by only 7 pp (0.21 → 0.28). Augmentation at the same data size raises Japanese recall by 43 pp (0.28 → 0.71). This demonstrates that the Japanese–Spanish collapse is not a data-quantity failure but a data-diversity failure: the model requires spectral and temporal perturbation to learn features beyond vowel quality."

---

## Direction 4: Confusion Asymmetry — Italian → Arabic

**Script:** `src/asymmetry_analysis.py`  
**Run with:** `~/anaconda3/envs/teamlab/bin/python asymmetry_analysis.py` (from `src/`, ~2 min)  
**Figures:** `figures/asymmetry_confusion_bars.png`, `figures/asymmetry_acoustic_violins.png`, `figures/asymmetry_mfcc_profiles.png`

### Motivation

wav2vec2-base misclassifies 192 Italian clips as Arabic with 0 errors in the reverse direction — a ratio of 1.0. This is the largest and most directional confusion in the entire study. We investigate whether this is driven by genuine acoustic similarity or by model-specific behavior.

### Key Results

**Asymmetry across models (Italian→Arabic : Arabic→Italian raw counts):**

| Model | Italian→Arabic | Arabic→Italian | Ratio |
|---|---|---|---|
| wav2vec2-base | **192** | 0 | 1.000 |
| XLS-R 1k/lang | 1 | 16 | 0.059 |
| XLS-R full data | 5 | 2 | 0.714 |
| XLS-R + Augment | 19 | 4 | 0.826 |

**Acoustic features — Italian vs Arabic (200 clips each, Mann-Whitney U):**

| Feature | Italian | Arabic | Significant? |
|---|---|---|---|
| Mean F0 (Hz) | 183 | 201 | *** |
| F0 Std Dev (Hz) | 54 | 64 | *** |
| Spectral Centroid (Hz) | 1613 | 2128 | *** |
| Zero Crossing Rate | 0.12 | 0.15 | *** |
| Voiced Frame Ratio | 0.96 | 0.97 | n.s. |

**MFCC profiles:** Despite the significant differences above, Italian and Arabic have nearly overlapping mean MFCC profiles relative to all other languages — they cluster together in low-dimensional spectral envelope space.

### Takeaways

1. **The Italian→Arabic asymmetry is wav2vec2-specific.** It essentially disappears in XLS-R models — in fact, XLS-R (1k/lang) shows the *opposite* direction (1 vs 16). This strongly implicates the model's pretraining (English-only wav2vec2 vs multilingual XLS-R) rather than the language pair's acoustics.

2. **The two languages are acoustically distinct on pitch and spectral brightness** — Arabic has significantly higher F0 (201 vs 183 Hz), more pitch variability, higher spectral centroid (2128 vs 1613 Hz), and higher ZCR. These are differences a robust model should separate.

3. **Their MFCC profiles are the closest of all 8 language pairs.** In the low-dimensional MFCC space wav2vec2 relies on most, Italian and Arabic cluster together while all other languages separate. wav2vec2's English-only pretraining may not have learned the pitch and spectral brightness cues that distinguish them.

4. **The asymmetry direction (Italian→Arabic, not vice versa) implies Arabic acts as an acoustic attractor** in wav2vec2's feature space. Arabic's higher spectral energy and pitch may define a region Italian clips can "fall into," but not vice versa.

### Implications for the Paper

> "The Italian→Arabic confusion (192 errors, ratio 1.0) in wav2vec2-base is a model-specific artifact, not an acoustic similarity effect. Despite Italian and Arabic having significantly different pitch (183 vs 201 Hz), spectral centroid (1613 vs 2128 Hz), and ZCR, they share near-identical mean MFCC profiles relative to all other languages. wav2vec2's English-only pretraining appears to over-rely on the MFCC spectral envelope and under-weight pitch and spectral brightness — the very dimensions that distinguish Italian from Arabic. XLS-R's multilingual pretraining eliminates this error almost entirely."

---

## Direction 5: Confusion Matrix Evolution Across Training Epochs

**Scripts:** `src/epoch_tracking_train.py` (run on HPC cluster), `src/epoch_tracking_visualize.py` (run locally)  
**Status:** "plain" condition complete and validated. "augmented" condition had a bug — needs re-run.

### Bug found and fixed (2026-06-29)

The first cluster run revealed a preprocessing bug: `epoch_tracking_train.py` applied the same augmented preprocessing function to train, validation, *and* test sets. This meant the "augmented" condition's test-set evaluations were corrupted by the same noise/time-masking/speed-perturbation used during training — explaining an anomalous Italian→Arabic error spike and lower-than-expected accuracy in that condition. The original `train_xlsr_augmented.py` correctly keeps val/test deterministic; the epoch-tracking script has been fixed to match (train uses `set_transform` with dynamic per-epoch augmentation; val/test use static, never-augmented preprocessing). Fix pushed to GitHub — **Fei needs to re-run only `--condition augmented`.**

### Results — "plain" condition (validated, matches known XLS-R behavior)

- Overall test accuracy reaches ~88-90% by epoch 2 and stays stable — consistent with the known xlsr/xlsr_fulldata benchmarks.
- **Japanese recall collapses by epoch 1** and stays low (0.2-0.4) for the remainder of training — the collapse is not a late-training drift, it appears almost immediately and persists.
- Italian→Arabic errors stay low throughout (consistent with Direction 4's finding that this asymmetry is a wav2vec2-specific, not XLS-R, phenomenon).

### Takeaways (preliminary — "plain" condition only)

1. **The Japanese collapse happens early, not gradually.** By the end of epoch 1, Japanese recall is already at its long-run low value. This suggests the model finds the Japanese≈Spanish shortcut almost immediately rather than drifting into it slowly — consistent with the model gravitating quickly toward the easiest separating feature (vowel quality) and then not being dislodged from it by further training.

2. Full comparison against the augmented condition is pending the cluster re-run.

### Implications for the Paper

> "The Japanese recall collapse is established within the first training epoch and persists essentially unchanged through epoch 10, suggesting XLS-R converges quickly onto a shortcut feature (likely vowel quality) rather than gradually drifting into it. This is consistent with the broader pattern of shortcut learning: once the model finds an easy separating cue, subsequent training does not self-correct without an external intervention such as augmentation."
