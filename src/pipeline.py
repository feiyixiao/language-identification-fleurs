import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from load_data import load_fleurs_subset
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
    # Load one language to test pipeline
    print("Loading English data...")
    data = load_fleurs_subset()
    en_data = data["en_us"]

    # Extract features
    print("Extracting MFCCs...")
    X, y = extract_features(en_data)
    print(f"Features shape: {X.shape}")