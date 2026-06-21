import numpy as np
import matplotlib.pyplot as plt

LANG_LABELS = ["Mandarin", "Japanese", "English", "German",
               "Spanish", "Italian", "Arabic", "Swahili"]

cm = np.load("../results/confusion_matrix_wav2vec2.npy")

# Sum each column (excluding diagonal) = how many times others were misclassified AS this language
attracted = []
for j in range(8):
    total = sum(cm[i][j] for i in range(8) if i != j)
    attracted.append(total)

plt.figure(figsize=(10, 5))
plt.bar(LANG_LABELS, attracted, color="steelblue")
plt.title("Misclassification Attractor Effect\n(how many times other languages were predicted as this one)")
plt.ylabel("Number of misclassifications attracted")
plt.xlabel("Language")
plt.tight_layout()
plt.savefig("../figures/attractor_effect.png")
plt.show()
print("Saved!")