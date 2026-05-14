import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from datasets import load_dataset
from extract_mfcc import extract_mfcc
 
def extract_features(dataset):
    """Extract MFCC features from a HuggingFace dataset split."""
    X, y = [], []
    for sample in dataset:
        mfcc = extract_mfcc(sample["audio"]["array"], sample["audio"]["sampling_rate"])
        X.append(mfcc)
        y.append(sample["language"])
    return np.array(X), np.array(y)
 
if __name__ == "__main__":
    # ── 1. Load English train / validation / test splits ──────────────────
    print("Loading English data (train, validation, test)...")
    en_train = load_dataset("google/fleurs", "en_us", split="train",      trust_remote_code=True)
    en_val   = load_dataset("google/fleurs", "en_us", split="validation", trust_remote_code=True)
    en_test  = load_dataset("google/fleurs", "en_us", split="test",       trust_remote_code=True)
 
    # Subsample 1000 training utterances (same strategy as load_data.py)
    en_train = en_train.shuffle(seed=42).select(range(min(1000, len(en_train))))
 
    # ── 2. Extract MFCC features ──────────────────────────────────────────
    print("Extracting MFCC features...")
    X_train, y_train = extract_features(en_train)
    X_val,   y_val   = extract_features(en_val)
    X_test,  y_test  = extract_features(en_test)
 
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
 
    # ── 3. Train Logistic Regression classifier ───────────────────────────
    print("Training Logistic Regression...")
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
 
    # ── 4. Evaluate on validation set ─────────────────────────────────────
    y_val_pred = clf.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"\nValidation Accuracy: {val_acc:.4f}")
 
    # ── 5. Evaluate on test set ────────────────────────────────────────────
    y_test_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"Test Accuracy:       {test_acc:.4f}")
    print("\nClassification Report (Test):")
    print(classification_report(y_test, y_test_pred))
