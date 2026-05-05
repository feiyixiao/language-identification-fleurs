"""
MFCC feature extraction using torchaudio.
Returns a 1-D numpy array (mean-pooled over time frames).
"""

import numpy as np
import torch
import torchaudio.transforms as T


def extract_mfcc(audio_array, sample_rate, n_mfcc=13):
    """
    Args:
        audio_array : np.ndarray or list, raw waveform (mono, any dtype)
        sample_rate : int, e.g. 16000
        n_mfcc      : int, number of MFCC coefficients to keep (excl. C0)

    Returns:
        np.ndarray of shape (n_mfcc,)  — mean over time
    """
    # Convert to float32 tensor, shape (1, T)
    waveform = torch.tensor(audio_array, dtype=torch.float32).unsqueeze(0)

    # torchaudio MelSpectrogram → MFCC pipeline
    # n_mfcc+1 so we can drop C0 (energy term, very large magnitude)
    transform = T.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc + 1,          # extract one extra, drop C0 below
        melkwargs={
            "n_fft": 400,            # 25 ms at 16 kHz
            "hop_length": 160,       # 10 ms at 16 kHz
            "n_mels": 40,
            "f_min": 20.0,
            "f_max": sample_rate / 2,
        },
    )

    with torch.no_grad():
        mfcc = transform(waveform)   # (1, n_mfcc+1, T)

    mfcc = mfcc.squeeze(0)           # (n_mfcc+1, T)
    mfcc = mfcc[1:]                  # drop C0  → (n_mfcc, T)
    mfcc = mfcc.mean(dim=1)          # mean over time → (n_mfcc,)

    return mfcc.numpy()


if __name__ == "__main__":
    from datasets import load_dataset

    print("Loading one sample from English FLEURS...")
    ds = load_dataset("google/fleurs", "en_us", split="train", trust_remote_code=True)
    sample = ds[0]

    audio = sample["audio"]
    mfcc  = extract_mfcc(audio["array"], audio["sampling_rate"])

    print(f"Sample     : {sample['transcription'][:60]}...")
    print(f"MFCC shape : {mfcc.shape}")   # should be (13,)
    print(f"MFCC values: {mfcc.round(3)}")