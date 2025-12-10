import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

thresholds = [0.005, 0.01, 0.02]
variations = [0.1, 0.2, 0.3, 0.4, 0.5]

counter_examples = np.array([
    [13,24,30,33,33], # Per variation, threshold = 0.005
    [42,73,90,96,99], # Per variation, threshold = 0.01
    [73,97,99,99,99] # Per variation, threshold = 0.02
])

plt.figure(figsize=(8,4))
sb.heatmap(
    counter_examples,
    annot=True,
    fmt='d',
    cmap='viridis_r',
    xticklabels=variations,
    yticklabels=thresholds,
    cbar_kws={"label": "Number counterexamples found with Vehicle"}
)

plt.xlabel("PCA Unimportant Feature Variation")
plt.ylabel("PCA Unimportant Feature Threshold (<)")
plt.title("Heatmap of Counterexample Results")
plt.tight_layout()
plt.show()