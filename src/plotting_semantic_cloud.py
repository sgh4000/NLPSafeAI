import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle as pk

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # NLPSafeAI/


def plot_semantic_cloud(
    dataset_name="depression",
    encoding_model_name="sbert22M",
    path="datasets",
    hyperrectangles_name="semantic",
    max_hr_to_draw=50,   # draw first N HR boxes to avoid clutter
    alpha_points=0.35
):
    """
    Single PCA scatter plot showing:
      1) Original points colored by POS/NEG
      2) Semantic perturbations colored by POS/NEG
      3) If C1–C9 rule labels exist, semantic points also get marker styles per rule
      4) Semantic hyperrectangles overlaid (first max_hr_to_draw)
    """

    base_path = PROJECT_ROOT / path / dataset_name / "embeddings" / encoding_model_name
    orig_dir = base_path / "original"
    sem_dir  = base_path / "semantic"

    # --------------------------
    # Load PCA transformer
    # --------------------------
    pca_path = PROJECT_ROOT / path / dataset_name / "embeddings" / encoding_model_name / "pca.pkl"
    with open(pca_path, "rb") as f:
        pca = pk.load(f)

    # --------------------------
    # Load original embeddings (already PCA trained on these)
    # --------------------------
    X_train_pos_o = np.load(orig_dir / "X_train_pos.npy")
    X_train_neg_o = np.load(orig_dir / "X_train_neg.npy")

    # project to PCA space
    Xp_o = pca.transform(X_train_pos_o)
    Xn_o = pca.transform(X_train_neg_o)

    # --------------------------
    # Load semantic embeddings
    # --------------------------
    X_train_pos_s = np.load(sem_dir / "X_train_pos.npy")
    X_train_neg_s = np.load(sem_dir / "X_train_neg.npy")

    Xp_s = pca.transform(X_train_pos_s)
    Xn_s = pca.transform(X_train_neg_s)

    # --------------------------
    # Optional: load C1–C9 rule labels if they exist
    # Expected file names (you can create later):
    #   perturbations/semantic/rules/train_pos_rules.npy
    #   perturbations/semantic/rules/train_neg_rules.npy
    # --------------------------
    rules_dir = PROJECT_ROOT / path / dataset_name / "perturbations" / "semantic" / "rules"
    pos_rules_path = rules_dir / "train_pos_rules.npy"
    neg_rules_path = rules_dir / "train_neg_rules.npy"

    pos_rules = None
    neg_rules = None
    if pos_rules_path.exists() and neg_rules_path.exists():
        pos_rules = np.load(pos_rules_path)
        neg_rules = np.load(neg_rules_path)

    # Marker map for C1–C9
    rule_markers = {
        1: "o", 2: "s", 3: "^", 4: "v", 5: "D",
        6: "P", 7: "X", 8: "*", 9: "<"
    }

    # --------------------------
    # Plot
    # --------------------------
    plt.figure(figsize=(8, 7))

    # Original clouds
    plt.scatter(Xn_o[:, 0], Xn_o[:, 1], s=10, alpha=alpha_points, label="Original NEG")
    plt.scatter(Xp_o[:, 0], Xp_o[:, 1], s=10, alpha=alpha_points, label="Original POS")

    # Semantic clouds
    if pos_rules is None:
        # no rule labels → normal scatter
        plt.scatter(Xn_s[:, 0], Xn_s[:, 1], s=8, alpha=alpha_points, label="Semantic NEG")
        plt.scatter(Xp_s[:, 0], Xp_s[:, 1], s=8, alpha=alpha_points, label="Semantic POS")
    else:
        # rule labels exist → marker depends on rule id
        for r in range(1, 10):
            m = rule_markers[r]
            mask_p = (pos_rules == r)
            mask_n = (neg_rules == r)
            if np.any(mask_p):
                plt.scatter(Xp_s[mask_p, 0], Xp_s[mask_p, 1],
                            s=10, alpha=alpha_points, marker=m,
                            label=f"Semantic POS (C{r})")
            if np.any(mask_n):
                plt.scatter(Xn_s[mask_n, 0], Xn_s[mask_n, 1],
                            s=10, alpha=alpha_points, marker=m,
                            label=f"Semantic NEG (C{r})")

    # --------------------------
    # Overlay semantic hyperrectangles (in PCA space already)
    # --------------------------
    hr_path = PROJECT_ROOT / path / dataset_name / "hyperrectangles" / encoding_model_name / f"{hyperrectangles_name}.npy"
    if hr_path.exists():
        hrs = np.load(hr_path)   # (N, 30, 2)
        n_draw = min(len(hrs), max_hr_to_draw)

        for i in range(n_draw):
            mins = hrs[i, :, 0]
            maxs = hrs[i, :, 1]
            # draw only first 2 PCA dims as rectangle
            x0, x1 = mins[0], maxs[0]
            y0, y1 = mins[1], maxs[1]
            plt.plot([x0, x1, x1, x0, x0],
                     [y0, y0, y1, y1, y0],
                     linewidth=0.8, alpha=0.6)

        plt.text(0.02, 0.98, f"HR boxes shown: {n_draw}/{len(hrs)}",
                 transform=plt.gca().transAxes,
                 va="top", fontsize=9)

    # Cosmetics
    plt.title("PCA cloud: Original vs Semantic perturbations (+ HR boxes)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(markerscale=1.5, fontsize=8, loc="best")
    plt.grid(alpha=0.3)

    out_dir = PROJECT_ROOT / "src" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)  # parents=True in case src/results doesn't exist
    out_path = out_dir / "semantic_pca_cloud.png"
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.show()
    plt.close()
    
    print(f"[PLOT] Saved PCA cloud to: {out_path}")
