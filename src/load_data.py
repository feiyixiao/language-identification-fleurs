from datasets import load_dataset

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

SAMPLES_PER_LANG = 1000

def load_fleurs_subset():
    subsets = {}
    for lang in LANGUAGES:
        print(f"Loading {lang}...")
        ds = load_dataset("google/fleurs", lang, split="train")
        ds = ds.shuffle(seed=42).select(range(SAMPLES_PER_LANG))
        subsets[lang] = ds
        print(f"  {lang}: {len(ds)} samples loaded")
    return subsets

if __name__ == "__main__":
    data = load_fleurs_subset()
    print("\nDone! Languages loaded:", list(data.keys()))