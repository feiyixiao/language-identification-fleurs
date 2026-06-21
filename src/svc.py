import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from load_data import load_fleurs_subset
from extract_mfcc import extract_mfcc
from datasets import load_dataset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os
import seaborn as sns

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

def extract_features(dataset, lang_label):
    """Extract MFCC features from a HuggingFace dataset split."""
    X, y = [], []
    for sample in dataset:
        mfcc = extract_mfcc(sample["audio"]["array"], sample["audio"]["sampling_rate"])
        X.append(mfcc)
        y.append(lang_label)
    return np.array(X), np.array(y)

def load_split(lang, split):
    """Load a single language split directly from FLEURS."""
    return load_dataset("google/fleurs", lang, split=split, trust_remote_code=True)

if __name__ == "__main__":
    # ── 1. Load train split for all 8 languages (subsampled) ──────────────
    print("Loading training data for all 8 languages...")
    train_data = load_fleurs_subset()  # 1000 samples per language

    # ── 2. Load val and test splits for all 8 languages (full) ────────────
    print("Loading validation and test splits...")
    X_val_all,  y_val_all  = [], []
    X_test_all, y_test_all = [], []

    for lang in LANGUAGES:
        print(f"  {lang} val/test...")
        val_ds  = load_split(lang, "validation")
        test_ds = load_split(lang, "test")

        X_v, y_v = extract_features(val_ds,  lang)
        X_t, y_t = extract_features(test_ds, lang)

        X_val_all.append(X_v);   y_val_all.append(y_v)
        X_test_all.append(X_t);  y_test_all.append(y_t)

    X_val  = np.concatenate(X_val_all);  y_val  = np.concatenate(y_val_all)
    X_test = np.concatenate(X_test_all); y_test = np.concatenate(y_test_all)

    # ── 3. Extract train features ──────────────────────────────────────────
    print("Extracting training MFCC features...")
    X_train_all, y_train_all = [], []
    for lang, ds in train_data.items():
        print(f"  {lang}...")
        X, y = extract_features(ds, lang)
        X_train_all.append(X)
        y_train_all.append(y)

    X_train = np.concatenate(X_train_all)
    y_train = np.concatenate(y_train_all)

    print(f"\nDataset sizes:")
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # ── 4. Standardize features ────────────────────────────────────────────
    print("\nStandardizing features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # ── 5. Train LinearSVC classifier ──────────────────────────────────────
    # LinearSVC avoids the numerical overflow issues seen with LogisticRegression
    # on this dataset, while remaining a linear baseline comparable in complexity.
    print("\nTraining LinearSVC classifier...")
    clf = LinearSVC(max_iter=2000, C=1.0, random_state=42)
    clf.fit(X_train, y_train)

    # ── 6. Evaluate on validation set ──────────────────────────────────────
    y_val_pred = clf.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"\nValidation Accuracy: {val_acc:.4f}")

    # ── 7. Evaluate on test set ─────────────────────────────────────────────
    y_test_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"Test Accuracy:       {test_acc:.4f}")
    print("\nClassification Report (Test):")
    print(classification_report(y_test, y_test_pred))

    # ── 8. Save confusion matrix ────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_test_pred, labels=LANGUAGES)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=LANGUAGES, yticklabels=LANGUAGES, cmap="Blues")
    plt.title("Confusion Matrix (Official Test Split)")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    os.makedirs("../results", exist_ok=True)
    np.save("../results/confusion_matrix.npy", cm)
    plt.savefig("../results/svc_confusion_matrix.png")
    print("Saved confusion matrix to results/")
    plt.close()