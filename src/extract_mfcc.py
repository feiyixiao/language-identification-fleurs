import numpy as np
import librosa
from datasets import load_dataset

def extract_mfcc(audio_array, sample_rate, n_mfcc=13):
    mfcc = librosa.feature.mfcc(
        y=audio_array.astype(np.float32),
        sr=sample_rate,
        n_mfcc=n_mfcc
    )
    return np.mean(mfcc, axis=1)  

if __name__ == "__main__":
    print("Loading one sample from English...")
    ds = load_dataset("google/fleurs", "en_us", split="train", trust_remote_code=True)
    sample = ds[0]

    audio = sample["audio"]
    mfcc = extract_mfcc(audio["array"], audio["sampling_rate"])

    print(f"Sample: {sample['transcription']}")
    print(f"MFCC shape: {mfcc.shape}")
    print(f"MFCC values: {mfcc}")