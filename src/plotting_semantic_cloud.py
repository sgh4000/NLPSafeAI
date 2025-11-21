import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_semantic_cloud(dataset_name, encoding_model_name, path="datasets"):

    print("[PCA] Loading embeddings...")

    # Base path to original embeddings
    base_path = os.path.join(path, dataset_name, "embeddings", encoding_model_name)

    # ---------- ORIGINAL EMBEDDINGS ----------
    orig_path = os.path.join(base_path, "original")
    train_pos = np.load(os.path.join(orig_path, "X_train_pos.npy"))
    train_neg = np.load(os.path.join(orig_path, "X_train_neg.npy"))

    original = np.vstack([train_pos, train_neg])

    print("[PCA] Original shape:", original.shape)

    # ---------- SEMANTIC EMBEDDINGS ----------
    sem_path = os.path.join(base_path, "semantic")

    sem_train_pos = np.load(os.path.join(sem_path, "X_train_pos.npy"))
    sem_train_neg = np.load(os.path.join(sem_path, "X_train_neg.npy"))

    semantic = np.vstack([sem_train_pos, sem_train_neg])
    print("[PCA] Semantic shape:", semantic.shape)

    # ---------- PCA ----------
    print("[PCA] Computing PCA down to 2D...")
    pca = PCA(n_components=2)
    all_points = np.vstack([original, semantic])
    pca.fit(all_points)

    orig_2d = pca.transform(original)
    sem_2d = pca.transform(semantic)

    # ---------- PLOT ----------
    plt.figure(figsize=(8, 6))
    plt.scatter(orig_2d[:, 0], orig_2d[:, 1], s=5, alpha=0.5, label="Original", color="blue")
    plt.scatter(sem_2d[:, 0], sem_2d[:, 1], s=5, alpha=0.5, label="Semantic Perturbations", color="red")

    plt.title("PCA Projection:\nOriginal vs Semantic Perturbations")
    plt.legend()
    plt.grid(alpha=0.3)

    save_path = os.path.join("results", "semantic_cloud.png")
    plt.savefig(save_path, dpi=200)
    plt.show()
    plt.close()

    print(f"[PCA] Saved semantic cloud plot to {save_path}")
