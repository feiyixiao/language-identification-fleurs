import numpy as np

cm = np.array([
    [943, 0, 2, 0, 0, 0, 0, 0],    # Mandarin
    [31, 500, 2, 0, 85, 4, 5, 23],  # Japanese
    [0, 0, 647, 0, 0, 0, 0, 0],     # English
    [2, 2, 97, 698, 0, 15, 47, 1],  # German
    [6, 3, 3, 8, 573, 165, 129, 21],# Spanish
    [14, 0, 16, 2, 4, 617, 192, 20],# Italian
    [8, 0, 0, 0, 15, 0, 405, 0],    # Arabic
    [2, 5, 0, 0, 0, 0, 40, 440],    # Swahili
])

np.save("../results/confusion_matrix_wav2vec2.npy", cm)
print("Saved!")