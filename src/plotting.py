import os
import matplotlib.pyplot as plt

def plot_semantic_bar(results, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)

    base = results["base_stability"]
    adv  = results["adv_stability"]

    labels = ["Base", "Adversarial"]
    vals = [base, adv]

    print("[PLOT] Creating semantic stability bar chart...")
    plt.figure(figsize=(5.5, 4))

    colors = ["#4C78A8", "#F58518"]  # calm blue + orange
    bars = plt.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.8)

    plt.ylim(0, 1)
    plt.ylabel("Semantic Stability Rate", fontsize=11)
    plt.title("Semantic Robustness on Semantic Hyperrectangles\n(N = 2376 HRs)", fontsize=12)

    # Light grid for readability
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.gca().set_axisbelow(True)

    # Add numeric labels (percentage + raw)
    for bar, v in zip(bars, vals):
        plt.text(
            bar.get_x() + bar.get_width()/2,
            v + 0.02,
            f"{v:.3f}\n({v*100:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10
        )

    # tidy up spines 
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out_path = os.path.join(save_dir, "semantic_stability_bar.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    print(f"[PLOT] Saved semantic stability bar chart to: {out_path}")
