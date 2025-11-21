import os
import matplotlib.pyplot as plt

def plot_semantic_bar(results, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)

    base = results["base_stability"]
    adv  = results["adv_stability"]

    labels = ["Base", "Adversarial"]
    vals = [base, adv]

    plt.figure()
    plt.bar(labels, vals)
    plt.ylim(0, 1)
    plt.ylabel("Semantic Stability Rate")
    plt.title("Semantic robustness (2376 semantic HRs)")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    out_path = os.path.join(save_dir, "semantic_stability_bar.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()

    print(f"[PLOT] Saved bar chart to {out_path}")
