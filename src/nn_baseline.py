import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset

from load_data import load_fleurs_subset, LANGUAGES
from extract_mfcc import extract_mfcc

N_MFCC      = 13
HIDDEN_DIMS = [256, 128]
DROPOUT     = 0.3
LR          = 1e-3
EPOCHS      = 30
BATCH_SIZE  = 64
SEED        = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


def build_features(split_dict):
    X, y = [], []
    for lang, ds in split_dict.items():
        print(f"  {lang}...")
        for sample in ds:
            mfcc = extract_mfcc(
                sample["audio"]["array"],
                sample["audio"]["sampling_rate"],
                n_mfcc=N_MFCC,
            )
            X.append(mfcc)
            y.append(lang)
    return np.array(X, dtype=np.float32), np.array(y)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, n_classes, dropout):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


def evaluate(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            out = model(X_batch.to(DEVICE)).argmax(dim=1).cpu().numpy()
            preds.extend(out)
            targets.extend(y_batch.numpy())
    acc = accuracy_score(targets, preds)
    f1  = f1_score(targets, preds, average="macro")
    return acc, f1, np.array(targets), np.array(preds)


def make_loader(X, y, shuffle=False):
    ds = TensorDataset(torch.tensor(X), torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)


if __name__ == "__main__":
    print("\n=== Loading data ===")
    train_raw = load_fleurs_subset()

    print("\n=== Extracting features ===")
    X_all, y_all_str = build_features(train_raw)
    input_dim = X_all.shape[1]
    print(f"Feature matrix: {X_all.shape}")

    le    = LabelEncoder()
    y_all = le.fit_transform(y_all_str)
    print(f"Classes: {list(le.classes_)}")

    mean  = X_all.mean(axis=0)
    std   = X_all.std(axis=0) + 1e-8
    X_all = (X_all - mean) / std

    n   = len(X_all)
    idx = np.random.permutation(n)
    cut = int(0.8 * n)
    X_tr, y_tr = X_all[idx[:cut]], y_all[idx[:cut]]
    X_va, y_va = X_all[idx[cut:]], y_all[idx[cut:]]

    train_loader = make_loader(X_tr, y_tr, shuffle=True)
    val_loader   = make_loader(X_va, y_va)

    n_classes = len(le.classes_)
    model     = MLP(input_dim, HIDDEN_DIMS, n_classes, DROPOUT).to(DEVICE)
    print(f"\n{model}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    print("\n=== Training ===")
    history = {"loss": [], "val_acc": [], "val_f1": []}

    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion)
        val_acc, val_f1, _, _ = evaluate(model, val_loader)
        scheduler.step()
        history["loss"].append(loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | loss={loss:.4f} | val_acc={val_acc:.4f} | macro-F1={val_f1:.4f}")

    val_acc, val_f1, y_true, y_pred = evaluate(model, val_loader)
    print(f"\nVal accuracy : {val_acc:.4f}")
    print(f"Macro-F1     : {val_f1:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history["loss"],    label="train loss")
    axes[0].plot(history["val_acc"], label="val acc")
    axes[0].plot(history["val_f1"],  label="macro-F1")
    axes[0].set_xlabel("Epoch")
    axes[0].set_title("Training curves")
    axes[0].legend()
    axes[0].grid(True)

    sns.heatmap(
        cm,
        annot=True, fmt="d",
        xticklabels=le.classes_,
        yticklabels=le.classes_,
        cmap="Blues",
        ax=axes[1],
    )
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    axes[1].set_title("Confusion matrix (val)")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("Saved confusion_matrix.png")

    torch.save(
        {
            "model_state":   model.state_dict(),
            "label_encoder": le,
            "mean":          mean,
            "std":           std,
            "input_dim":     input_dim,
        },
        "mlp_lid.pt",
    )
    print("Saved mlp_lid.pt")

    # ── Evaluate on official FLEURS val and test splits ───────────────────
    print("\n=== Evaluating on FLEURS validation and test splits ===")

    def eval_fleurs_split(split_name):
        X_s, y_s_str = [], []
        for lang in LANGUAGES:
            ds = load_dataset("google/fleurs", lang, split=split_name, trust_remote_code=True)
            print(f"  {lang} [{split_name}]: {len(ds)} samples")
            for sample in ds:
                mfcc = extract_mfcc(
                    sample["audio"]["array"],
                    sample["audio"]["sampling_rate"],
                    n_mfcc=N_MFCC,
                )
                X_s.append(mfcc)
                y_s_str.append(lang)
        X_s = np.array(X_s, dtype=np.float32)
        y_s = le.transform(y_s_str)
        X_s = (X_s - mean) / std
        loader = make_loader(X_s, y_s)
        acc, f1, _, _ = evaluate(model, loader)
        return acc, f1

    fleurs_val_acc,  fleurs_val_f1  = eval_fleurs_split("validation")
    fleurs_test_acc, fleurs_test_f1 = eval_fleurs_split("test")

    print(f"\nFLEURS Val  — accuracy: {fleurs_val_acc:.4f} | macro-F1: {fleurs_val_f1:.4f}")
    print(f"FLEURS Test — accuracy: {fleurs_test_acc:.4f} | macro-F1: {fleurs_test_f1:.4f}")

    os.makedirs("../results", exist_ok=True)
    with open("../results/results_baseline.txt", "w") as f:
        f.write("Method: MFCC + MLP\n")
        f.write(f"Val Accuracy: {fleurs_val_acc:.4f}\n")
        f.write(f"Val Macro-F1: {fleurs_val_f1:.4f}\n")
        f.write(f"Test Accuracy: {fleurs_test_acc:.4f}\n")
        f.write(f"Test Macro-F1: {fleurs_test_f1:.4f}\n")
    print("Saved results/results_baseline.txt")