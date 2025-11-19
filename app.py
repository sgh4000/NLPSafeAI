# app.py
# Streamlit app for SafeNLP: NLP Robustness & Cosine Filter

import os, sys
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import shutil
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F


# repo imports
from sentence_transformers import util, SentenceTransformer
from data import load_data, load_embeddings, load_align_mat
from perturbations import create_perturbations

# NEW: SafeNLP hyperrectangles API
from hyperrectangles import load_hyperrectangles, contained

pos_lable_1_neg_lable_0 = True
test_train_split_rate = 0.7

# --------------------------- NLTK bootstrap ---------------------------
def _ensure_nltk():
    try:
        import nltk
        needed = [
            ("tokenizers/punkt", "punkt"),
            ("tokenizers/punkt_tab", "punkt_tab"),  # newer NLTK split
            ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
            ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ]
        for path_key, pkg in needed:
            try:
                nltk.data.find(path_key)
            except LookupError:
                try:
                    nltk.download(pkg, quiet=True)
                except Exception:
                    pass
    except Exception:
        # don't crash the app if offline, etc.
        pass

_ensure_nltk()


# --------------------------- Utility helpers ---------------------------
def cos_manual(a, b):
    at = torch.as_tensor(a, dtype=torch.float32)
    bt = torch.as_tensor(b, dtype=torch.float32)
    if at.ndim == 1: at = at.unsqueeze(0)
    if bt.ndim == 1: bt = bt.unsqueeze(0)

    at = F.normalize(at, dim=-1)
    bt = F.normalize(bt, dim=-1)
    return (at @ bt.T)  # same as cos_sim

def cos_sim_np(a, b):
    """Cosine sim that safely casts NumPy arrays to float32 tensors."""
    at = torch.as_tensor(a, dtype=torch.float32)
    bt = torch.as_tensor(b, dtype=torch.float32)
    if at.ndim == 1: at = at.unsqueeze(0)
    if bt.ndim == 1: bt = bt.unsqueeze(0)
    return util.cos_sim(at, bt)


def find_datasets(root="datasets"):
    if not os.path.isdir(root):
        return []
    out = []
    for d in sorted(os.listdir(root)):
        data_dir = os.path.join(root, d, "data")
        if os.path.isdir(data_dir):
            out.append(d)
    return out


def default_models():
    # Map HF model id -> folder name used in your repo structure
    return {
        "sentence-transformers/all-MiniLM-L6-v2": "sbert22M",
        "sentence-transformers/all-mpnet-base-v2": "sbert-mpnet",
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": "sbert-multi-mpnet",
    }


def _emb_dir(path, dataset, enc_name, pert):
    return os.path.join(path, dataset, "embeddings", enc_name, pert)


def _pert_dir(path, dataset, pert, sub):
    return os.path.join(path, dataset, "perturbations", pert, sub)


def embeddings_present(path, dataset, enc_name, pert) -> bool:
    d = _emb_dir(path, dataset, enc_name, pert)
    needed = [
        "X_train_pos.npy", "X_train_neg.npy", "y_train_pos.npy", "y_train_neg.npy",
        "X_test_pos.npy",  "X_test_neg.npy",  "y_test_pos.npy",  "y_test_neg.npy",
    ]
    return os.path.isdir(d) and all(os.path.isfile(os.path.join(d, f)) for f in needed)


def perturbation_sentences_present(path, dataset, pert) -> bool:
    sdir = _pert_dir(path, dataset, pert, "sentences")
    needed = [
        "X_train_pos.npy", "X_train_neg.npy", "y_train_pos.npy", "y_train_neg.npy",
        "X_test_pos.npy",  "X_test_neg.npy",  "y_test_pos.npy",  "y_test_neg.npy",
    ]
    return os.path.isdir(sdir) and all(os.path.isfile(os.path.join(sdir, f)) for f in needed)


def perturbation_indexes_present(path, dataset, pert) -> bool:
    idir = _pert_dir(path, dataset, pert, "indexes")
    needed = ["train_pos_indexes.npy", "train_neg_indexes.npy"]
    return os.path.isdir(idir) and all(os.path.isfile(os.path.join(idir, f)) for f in needed)


def _clean_embeddings_dir(path, dataset, enc_name, pert):
    d = _emb_dir(path, dataset, enc_name, pert)
    if os.path.isdir(d):
        shutil.rmtree(d)


def _safe_load(path, allow_empty=False, dtype=object):
    if os.path.isfile(path):
        return np.load(path, allow_pickle=True)
    if allow_empty:
        return np.array([], dtype=dtype)
    raise FileNotFoundError(path)


def _encode_list(model: SentenceTransformer, arr):
    if arr is None or len(arr) == 0:
        # build an empty (0, dim) array, dim is the model’s embedding size (e.g., 384 for MiniLM, 768 for mpnet).
        dim = model.get_sentence_embedding_dimension()
        return np.empty((0, dim), dtype=np.float32)
    return model.encode(
        list(arr),
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
        device="cpu",  # device = "cuda" if torch.cuda.is_available() else "cpu"
    )


def _encode_and_save_embeds_from_pert_sentences(dataset_name, enc_key, enc_name, pert, path="datasets"):
    """
    Build embeddings for a perturbation by reading sentences from:
      datasets/<dataset>/perturbations/<pert>/sentences/{X_train_*, X_test_*}.npy
    Then encode and save to:
      datasets/<dataset>/embeddings/<enc_name>/<pert>/{X_train_*, X_test_*}.npy
    Also copies y_* labels. If test files are missing, we encode empty arrays.
    """
    sent_dir = _pert_dir(path, dataset_name, pert, "sentences")
    idx_dir  = _pert_dir(path, dataset_name, pert, "indexes")
    out_dir  = _emb_dir(path, dataset_name, enc_name, pert)
    os.makedirs(out_dir, exist_ok=True)

    # ---- Train sentences + labels (required)
    Xtr_p = _safe_load(os.path.join(sent_dir, "X_train_pos.npy"))
    Xtr_n = _safe_load(os.path.join(sent_dir, "X_train_neg.npy"))
    ytr_p = _safe_load(os.path.join(sent_dir, "y_train_pos.npy"))
    ytr_n = _safe_load(os.path.join(sent_dir, "y_train_neg.npy"))

    # ---- Test sentences + labels (optional -> allow empty)
    Xte_p = _safe_load(os.path.join(sent_dir, "X_test_pos.npy"), allow_empty=True)
    Xte_n = _safe_load(os.path.join(sent_dir, "X_test_neg.npy"), allow_empty=True)
    yte_p = _safe_load(os.path.join(sent_dir, "y_test_pos.npy"), allow_empty=True)
    yte_n = _safe_load(os.path.join(sent_dir, "y_test_neg.npy"), allow_empty=True)

    # ---- Index lengths must match train sentences
    pos_idx = _safe_load(os.path.join(idx_dir, "train_pos_indexes.npy"))
    neg_idx = _safe_load(os.path.join(idx_dir, "train_neg_indexes.npy"))
    if len(Xtr_p) != len(pos_idx) or len(Xtr_n) != len(neg_idx):
        # Sentences / indexes inconsistent on disk -> regenerate them
        raise RuntimeError(
            "Perturbation sentences and indexes are inconsistent before encoding: "
            f"train pos_sent={len(Xtr_p)} vs pos_idx={len(pos_idx)} | "
            f"train neg_sent={len(Xtr_n)} vs neg_idx={len(neg_idx)}"
        )

    # ---- Encode
    model = SentenceTransformer(enc_key, device="cpu")
    Xtr_p_emb = _encode_list(model, Xtr_p)
    Xtr_n_emb = _encode_list(model, Xtr_n)
    Xte_p_emb = _encode_list(model, Xte_p)
    Xte_n_emb = _encode_list(model, Xte_n)
    
    align_mat = load_align_mat(dataset_name, enc_name, Xte_p, load_saved_align_mat = True, path='datasets')
    Xtr_p_emb = np.matmul(Xtr_p_emb, align_mat)
    Xtr_n_emb = np.matmul(Xtr_n_emb, align_mat)
    Xte_p_emb = np.matmul(Xte_p_emb, align_mat)
    Xte_n_emb = np.matmul(Xte_n_emb, align_mat)

    # ---- Save embeddings + labels
    np.save(os.path.join(out_dir, "X_train_pos.npy"), Xtr_p_emb)
    np.save(os.path.join(out_dir, "X_train_neg.npy"), Xtr_n_emb)
    np.save(os.path.join(out_dir, "y_train_pos.npy"), ytr_p)
    np.save(os.path.join(out_dir, "y_train_neg.npy"), ytr_n)

    np.save(os.path.join(out_dir, "X_test_pos.npy"), Xte_p_emb)
    np.save(os.path.join(out_dir, "X_test_neg.npy"), Xte_n_emb)
    np.save(os.path.join(out_dir, "y_test_pos.npy"), yte_p)
    np.save(os.path.join(out_dir, "y_test_neg.npy"), yte_n)


def get_perturbations(dataset_name, pert, mode, data_obj, path="datasets"):
    """
    mode ∈ {"auto", "existing", "rebuild"}:
      - auto: use existing if present, else build
      - existing: require existing; error if missing
      - rebuild: build (overwrite) then use
    We require BOTH: sentences and indexes for the perturbation.
    """
    have_sent = perturbation_sentences_present(path, dataset_name, pert)
    have_idx  = perturbation_indexes_present(path, dataset_name, pert)

    if mode == "existing":
        if not (have_sent and have_idx):
            st.error(f"Perturbation '{pert}' files are missing (sentences or indexes). Switch to 'Auto' or 'Rebuild'.")
            st.stop()
        return

    need_build = (mode == "rebuild") or (mode == "auto" and not (have_sent and have_idx))
    if need_build:
        with st.spinner(f"Generating perturbations '{pert}' (sentences + indexes)…"):
            create_perturbations(dataset_name, pert, data_obj, path=path)


def get_embeddings(dataset_name, enc_key, enc_name, pert, mode, data_obj,
                   path="datasets", clean_before_rebuild=True):
    """
    Returns the tuple from load_embeddings. When building:
      - original: build via load_embeddings(..., data=data_obj)
      - perturbation: encode from sentences into embeddings (train+test) and then load
    """
    have = embeddings_present(path, dataset_name, enc_name, pert)

    def build_original():
        return load_embeddings(dataset_name, enc_key, enc_name,
                               perturbation_name="original",
                               load_saved_embeddings=False,
                               load_saved_align_mat=False,
                               path=path, data=data_obj)

    def build_perturbed():
        # Ensure sentences+indexes exist; build if missing
        if not (perturbation_sentences_present(path, dataset_name, pert) and
                perturbation_indexes_present(path, dataset_name, pert)):
            create_perturbations(dataset_name, pert, data_obj, path=path)
        # Encode train+test from sentences into embeddings
        _encode_and_save_embeds_from_pert_sentences(dataset_name, enc_key, enc_name, pert, path=path)
        # Now load what we just saved
        return load_embeddings(dataset_name, enc_key, enc_name,
                               perturbation_name=pert,
                               load_saved_embeddings=True,
                               load_saved_align_mat=True,
                               path=path)

    def _build():
        return build_original() if pert == "original" else build_perturbed()

    if mode == "existing":
        if not have:
            st.error(f"Embeddings missing for {enc_name}/{pert}. Switch to 'Auto' or 'Rebuild'.")
            st.stop()
        return load_embeddings(dataset_name, enc_key, enc_name,
                               perturbation_name=pert, load_saved_embeddings=True,
                               load_saved_align_mat=True,
                               path=path)

    if mode == "rebuild" or (mode == "auto" and not have):
        if clean_before_rebuild:
            _clean_embeddings_dir(path, dataset_name, enc_name, pert)
        return _build()

    return load_embeddings(dataset_name, enc_key, enc_name,
                           perturbation_name=pert, load_saved_embeddings=True,
                           load_saved_align_mat=True,
                           path=path)


def ensure_consistency_or_repair(dataset_name, enc_key, enc_name, pert, mode, data_obj,
                                 path="datasets", clean_before_rebuild=True):
    """
    Ensures that perturbed embeddings length matches perturbation indexes length.
    If mismatch and mode allows rebuild, it cleans & rebuilds once. If the root cause
    is inconsistent SENTENCES vs INDEXES, we regenerate perturbations, then encode.
    """
    # Must have indexes
    if not perturbation_indexes_present(path, dataset_name, pert):
        # regenerate perturbations
        create_perturbations(dataset_name, pert, data_obj, path=path)

    pos_idx = _safe_load(os.path.join(path, dataset_name, "perturbations", pert, "indexes", "train_pos_indexes.npy"))
    neg_idx = _safe_load(os.path.join(path, dataset_name, "perturbations", pert, "indexes", "train_neg_indexes.npy"))

    # If sentences missing -> regenerate
    if not perturbation_sentences_present(path, dataset_name, pert):
        create_perturbations(dataset_name, pert, data_obj, path=path)

    # Before touching embeddings, confirm sentences align with indexes
    sdir = _pert_dir(path, dataset_name, pert, "sentences")
    Xtr_p = _safe_load(os.path.join(sdir, "X_train_pos.npy"))
    Xtr_n = _safe_load(os.path.join(sdir, "X_train_neg.npy"))
    if len(Xtr_p) != len(pos_idx) or len(Xtr_n) != len(neg_idx):
        # Regenerate sentences+indexes fully
        _ = shutil.rmtree(_pert_dir(path, dataset_name, pert, "sentences"), ignore_errors=True)
        _ = shutil.rmtree(_pert_dir(path, dataset_name, pert, "indexes"), ignore_errors=True)
        create_perturbations(dataset_name, pert, data_obj, path=path)
        # reload after regeneration
        pos_idx = _safe_load(os.path.join(path, dataset_name, "perturbations", pert, "indexes", "train_pos_indexes.npy"))
        neg_idx = _safe_load(os.path.join(path, dataset_name, "perturbations", pert, "indexes", "train_neg_indexes.npy"))
        Xtr_p = _safe_load(os.path.join(sdir, "X_train_pos.npy"))
        Xtr_n = _safe_load(os.path.join(sdir, "X_train_neg.npy"))

    # Now ensure embeddings exist & match
    Xp, Xn, *_ = get_embeddings(dataset_name, enc_key, enc_name, pert, mode, data_obj,
                                path=path, clean_before_rebuild=clean_before_rebuild)

    if len(Xp) == len(pos_idx) and len(Xn) == len(neg_idx):
        return

    if mode == "existing":
        st.error(
            f"Inconsistent on-disk files for {enc_name}/{pert}.\n\n"
            f"pos_emb={len(Xp)} vs pos_idx={len(pos_idx)} | "
            f"neg_emb={len(Xn)} vs neg_idx={len(neg_idx)}\n\n"
            f"Please switch Data mode to 'Rebuild now'."
        )
        st.stop()

    # Clean embeddings, re-encode from sentences, reload
    _clean_embeddings_dir(path, dataset_name, enc_name, pert)
    _encode_and_save_embeds_from_pert_sentences(dataset_name, enc_key, enc_name, pert, path=path)
    Xp2, Xn2, *_ = get_embeddings(dataset_name, enc_key, enc_name, pert, "existing", data_obj,
                                  path=path, clean_before_rebuild=False)

    if len(Xp2) != len(pos_idx) or len(Xn2) != len(neg_idx):
        st.error(
            f"After rebuild, shapes still mismatch:\n"
            f"pos_emb={len(Xp2)} vs pos_idx={len(pos_idx)} | "
            f"neg_emb={len(Xn2)} vs neg_idx={len(neg_idx)}.\n\n"
            f"Please delete BOTH folders and rerun:\n"
            f"• 'datasets/{dataset_name}/perturbations/{pert}'\n"
            f"• 'datasets/{dataset_name}/embeddings/{enc_name}/{pert}'"
        )
        st.stop()

def _ensure_embeddings_dtype(path_root, dataset_name, encoder_name, perts, to_dtype=np.float64):
    """
    Ensure embeddings .npy files under datasets/<dataset>/embeddings/<encoder>/<pert>/*
    are stored in a single dtype (default float64). This prevents dtype mismatches
    when SafeNLP builds hyperrectangles and calls util.cos_sim.
    """
    import os, numpy as np
    target = np.dtype(to_dtype)

    def _maybe_cast_file(fp):
        if not os.path.isfile(fp):
            return
        try:
            arr = np.load(fp, allow_pickle=True)
            if arr.dtype != target:
                np.save(fp, arr.astype(target, copy=False))
        except Exception:
            # Skip silently if a file can't be read/written
            pass

    emb_root = os.path.join(path_root, dataset_name, "embeddings", encoder_name)
    names = ["X_train_pos.npy", "X_train_neg.npy", "X_test_pos.npy", "X_test_neg.npy"]
    for pert in perts:
        for name in names:
            _maybe_cast_file(os.path.join(emb_root, pert, name))


def _safe_load_hrects(*, dataset_name, encoder_name, hyperrectangles_name,
                      path="datasets", pad_eps=1e-6, cosine_threshold=0.6, rebuild_now=False):
    """
    Non-breaking wrapper that prepares dtypes for a build if needed and then
    calls SafeNLP's load_hyperrectangles. If the file doesn't exist, it rebuilds once.
    """
    # decide whether we need to build
    import os
    target_file = os.path.join(path, dataset_name, "hyperrectangles",
                               encoder_name, f"{hyperrectangles_name}.npy")
    need_build = rebuild_now or (not os.path.isfile(target_file))

    if need_build:
        # upcast embeddings used by the builder so util.cos_sim sees one dtype
        _ensure_embeddings_dtype(path, dataset_name, encoder_name,
                                 perts=["original", hyperrectangles_name],
                                 to_dtype=np.float64)

    try:
        return load_hyperrectangles(
            dataset_name=dataset_name,
            encoding_model_name=encoder_name,
            hyperrectangles_name=hyperrectangles_name,
            load_saved_hyperrectangles=(not need_build),
            eps=float(pad_eps),
            cosine_threshold=float(cosine_threshold),
            path=path,
        )
    except FileNotFoundError:
        # If load failed because nothing is saved yet, build now
        _ensure_embeddings_dtype(path, dataset_name, encoder_name,
                                 perts=["original", hyperrectangles_name],
                                 to_dtype=np.float64)
        return load_hyperrectangles(
            dataset_name=dataset_name,
            encoding_model_name=encoder_name,
            hyperrectangles_name=hyperrectangles_name,
            load_saved_hyperrectangles=False,
            eps=float(pad_eps),
            cosine_threshold=float(cosine_threshold),
            path=path,
        )

# ====================== BEGIN: Threshold sweep helpers ======================
def _to_f32(x):
    return np.asarray(x, dtype=np.float32)

def _unit_normalize(mat):
    """Row-wise L2 normalize (float32)."""
    mat = _to_f32(mat)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return mat / norms

def _scores_true_pairs(orig_norm, pert_norm, src_idx):
    """Cosine for (orig[src[j]], pert[j]) for all j (vectorized)."""
    src_idx = np.asarray(src_idx, dtype=np.int64)
    sel = orig_norm[src_idx]                    # (N, d)
    return np.sum(sel * pert_norm, axis=1)     # (N,)

def _scores_neg_pairs(orig_norm, pert_norm, src_idx, k_neg=1, seed=42):
    """
    For each perturbed j, sample k_neg wrong originals and take the max impostor score.
    Returns one negative score per j (hard-negative among the k).
    """
    rng = np.random.default_rng(seed)
    n_orig = orig_norm.shape[0]
    src_idx = np.asarray(src_idx, dtype=np.int64)
    N = len(src_idx)

    # Edge case: if there's 0 or 1 original, there are no wrong originals to sample.
    if n_orig <= 1:
        return np.full(N, -np.inf, dtype=np.float32)

    # Cap k_neg to the maximum number of available wrong originals (M - 1)
    k_eff = min(k_neg, n_orig - 1)

    neg_scores = np.empty(N, dtype=np.float32)
    for j in range(N):
        true_i = src_idx[j]
        cand = []
        while len(cand) < k_eff:
            candidate = int(rng.integers(0, n_orig))
            if candidate != true_i:
                cand.append(candidate)
        cands = orig_norm[cand]
        s = np.dot(cands, pert_norm[j])
        neg_scores[j] = np.max(s)
    return neg_scores

def _confusion_from_scores(pos_scores, neg_scores, tau):
    tp = int((pos_scores >= tau).sum())
    fn = int((pos_scores <  tau).sum())
    fp = int((neg_scores >= tau).sum())
    tn = int((neg_scores <  tau).sum())
    return tp, fp, tn, fn

def _metrics_from_conf(tp, fp, tn, fn):
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tpr
    f1 = (2*prec*rec)/(prec+rec) if (prec+rec) else 0.0
    J = tpr - fpr
    return tpr, fpr, prec, rec, f1, J

def _auc_trapz(x, y):
    """Standard trapezoidal AUC; assumes x is sorted ascending."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    order = np.argsort(x)
    x = x[order]; y = y[order]
    return float(np.trapz(y, x))

def run_threshold_sweep_for_split(
    X_orig, X_pert, src_idx, *,
    negs_per_pert=1, tau_min=0.30, tau_max=0.95, tau_step=0.01, seed=42):
    """
    Build positive & negative cosine score sets, then sweep thresholds.
    Returns (sweep_df, summary_dict).
    """
    X_orig_n = _unit_normalize(X_orig)
    X_pert_n = _unit_normalize(X_pert)

    pos_scores = _scores_true_pairs(X_orig_n, X_pert_n, src_idx)
    neg_scores = _scores_neg_pairs(X_orig_n, X_pert_n, src_idx, k_neg=negs_per_pert, seed=seed)

    taus = np.arange(tau_min, tau_max + 1e-9, tau_step, dtype=np.float32)
    rows = []
    roc_x, roc_y = [], []
    pr_x, pr_y = [], []

    for tau in taus:
        tp, fp, tn, fn = _confusion_from_scores(pos_scores, neg_scores, tau)
        tpr, fpr, prec, rec, f1, J = _metrics_from_conf(tp, fp, tn, fn)
        rows.append({
            "tau": float(tau), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "tpr": tpr, "fpr": fpr, "precision": prec, "recall": rec, "f1": f1, "J": J
        })
        roc_x.append(fpr); roc_y.append(tpr)
        pr_x.append(rec);  pr_y.append(prec)

    auc_roc = _auc_trapz(roc_x, roc_y)
    auc_pr  = _auc_trapz(pr_x, pr_y)

    rows_np = np.array([[r["tau"], r["f1"], r["J"]] for r in rows], dtype=np.float64)
    i_f1 = int(np.argmax(rows_np[:, 1]))
    i_J  = int(np.argmax(rows_np[:, 2]))
    best_tau_f1, best_f1 = float(rows[i_f1]["tau"]), float(rows[i_f1]["f1"])
    best_tau_J,  best_J  = float(rows[i_J]["tau"]),  float(rows[i_J]["J"])

    sweep_df = pd.DataFrame(rows)
    summary = {
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "best_tau_f1": best_tau_f1,
        "best_f1": best_f1,
        # "best_tau_J": best_tau_J,
        # "best_J": best_J,
        "n_pos": int(len(pos_scores)),
        "n_neg": int(len(neg_scores)),
    }
    return sweep_df, summary

def run_threshold_sweep(
    dataset_name, enc_key, enc_name, perturbation, *,
    mode_key, data_obj, path="datasets",
    tau_min=0.30, tau_max=0.95, tau_step=0.01,
    negs_per_pert=1, seed=42
):
    """
    Runs sweep for both splits (positive & negative) and returns:
      combined_sweep_df, summary_df
    Also writes CSVs under: runs/<dataset>/<encoder>/<pert>/
    """
    ensure_consistency_or_repair(dataset_name, enc_key, enc_name, perturbation,
                                 mode_key, data_obj, path=path, clean_before_rebuild=True)

    Xp_o, Xn_o, *_ = get_embeddings(dataset_name, enc_key, enc_name, "original", mode_key, data_obj, path=path)
    Xp_p, Xn_p, *_ = get_embeddings(dataset_name, enc_key, enc_name, perturbation, mode_key, data_obj, path=path)

    pos_idx = _safe_load(os.path.join(path, dataset_name, "perturbations", perturbation, "indexes", "train_pos_indexes.npy"))
    neg_idx = _safe_load(os.path.join(path, dataset_name, "perturbations", perturbation, "indexes", "train_neg_indexes.npy"))

    sweep_pos, sum_pos = run_threshold_sweep_for_split(
        X_orig=Xp_o, X_pert=Xp_p, src_idx=pos_idx,
        negs_per_pert=negs_per_pert,
        tau_min=tau_min, tau_max=tau_max, tau_step=tau_step, seed=seed
    )
    sweep_pos.insert(0, "split", "positive")

    sweep_neg, sum_neg = run_threshold_sweep_for_split(
        X_orig=Xn_o, X_pert=Xn_p, src_idx=neg_idx,
        negs_per_pert=negs_per_pert,
        tau_min=tau_min, tau_max=tau_max, tau_step=tau_step, seed=seed
    )
    sweep_neg.insert(0, "split", "negative")

    combined = pd.concat([sweep_pos, sweep_neg], ignore_index=True)

    summary_df = pd.DataFrame([
        {
            "dataset": dataset_name,
            "encoder": enc_name,
            "perturbation": perturbation,
            "split": "positive",
            **sum_pos
        },
        {
            "dataset": dataset_name,
            "encoder": enc_name,
            "perturbation": perturbation,
            "split": "negative",
            **sum_neg
        },
    ])

    out_dir = os.path.join("runs", dataset_name, enc_name, perturbation)
    os.makedirs(out_dir, exist_ok=True)
    combined.to_csv(os.path.join(out_dir, "sweep.csv"), index=False)
    summary_df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    return combined, summary_df
# ====================== END: Threshold sweep helpers ======================


# ====================== BEGIN: SafeNLP-compatible G & F helpers ======================
def _ensure_pca_pickle(dataset_name, encoder_name, X_pos_orig, X_neg_orig, *, path="datasets"):
    """
    Ensure datasets/<dataset>/embeddings/<encoder_name>/pca.pkl exists.
    If missing, fit PCA on ORIGINAL (pos+neg) embeddings provided and save it.
    """
    import os, pickle
    try:
        from sklearn.decomposition import PCA
    except Exception:
        st.error("scikit-learn is required to auto-create PCA. Please `pip install scikit-learn`.")
        st.stop()

    pca_path = os.path.join(path, dataset_name, "embeddings", encoder_name, "pca.pkl")
    if os.path.isfile(pca_path):
        return

    X_list = []
    if X_pos_orig is not None and len(X_pos_orig) > 0:
        X_list.append(np.asarray(X_pos_orig))
    if X_neg_orig is not None and len(X_neg_orig) > 0:
        X_list.append(np.asarray(X_neg_orig))

    if not X_list:
        st.error("No ORIGINAL embeddings available to fit PCA. Run step (1) Prepare artifacts first.")
        st.stop()

    X_all = np.vstack(X_list)
    n_comp = min(X_all.shape[0], X_all.shape[1])
    pca = PCA(n_components=n_comp)
    pca.fit(X_all)

    os.makedirs(os.path.dirname(pca_path), exist_ok=True)
    with open(pca_path, "wb") as f:
        pickle.dump(pca, f)


def _load_pca_transform(dataset_name, encoder_name, *, path="datasets"):
    """Load the PCA object saved by SafeNLP and return a .transform function."""
    import pickle
    pca_path = os.path.join(path, dataset_name, "embeddings", encoder_name, "pca.pkl")
    if not os.path.isfile(pca_path):
        raise FileNotFoundError(pca_path)
    with open(pca_path, "rb") as f:
        pca = pickle.load(f)
    return pca.transform


def compute_generalisability_and_embedding_error(
    X_pos_orig, X_pos_pert, pos_src_idx,
    X_neg_pert=None,
    use_filtered=True,            # kept for API compatibility (not used by SafeNLP builder)
    pos_accept_mask=None,         # kept for API compatibility (not used by SafeNLP builder)
    include_orig=True, pad_eps=1e-6,
    *,
    dataset_name=None,
    encoder_name=None,
    hyperrectangles_name=None,    # e.g., "character" | "word" | "eps_cube"
    path="datasets",
    cosine_threshold=0.6,         # forwarded to SafeNLP builder
    rebuild_now=False             # if True, force a rebuild of rectangles with given threshold
):
    """
    Compute G & F using SafeNLP's hyper-rectangles + PCA rotation.
    - Ensures PCA exists (fits on ORIGINAL embeddings if missing via app helper).
    - Loads or rebuilds hyper-rectangles via _safe_load_hrects (handles dtype consistency).
    - Transforms perturbations into PCA space and checks membership with `contained`.
    """
    if dataset_name is None or encoder_name is None or hyperrectangles_name is None:
        raise ValueError("dataset_name, encoder_name, and hyperrectangles_name must be provided.")

    # 1) Ensure PCA is available for this dataset/encoder (uses originals you pass)
    #    If you've already called _ensure_pca_pickle before invoking this function, this is a no-op.
    _ensure_pca_pickle(dataset_name, enc_name, Xp_o, Xn_o, path="datasets")
    # _ensure_pca_pickle(dataset_name, encoder_name, X_pos_orig, None, path=path)
    pca_transform = _load_pca_transform(dataset_name, encoder_name, path=path)

    # 2) Load (or build) hyperrectangles safely. This wrapper upcasts embeddings on disk
    #    when a build is needed, avoiding torch dtype mismatches inside SafeNLP.
    hyperrects = _safe_load_hrects(
        dataset_name=dataset_name,
        encoder_name=encoder_name,
        hyperrectangles_name=hyperrectangles_name,
        path=path,
        pad_eps=pad_eps,
        cosine_threshold=cosine_threshold,
        rebuild_now=rebuild_now
    )
    hyperrects = np.asarray(hyperrects)  # shape: (M, d, 2)

    # 3) Rotate perturbations into PCA space for evaluation (SafeNLP builds rects in PCA space)
    X_pos_pert_rot = pca_transform(np.asarray(X_pos_pert)) if len(X_pos_pert) else np.empty_like(X_pos_pert)
    X_neg_pert_rot = None
    if X_neg_pert is not None and len(X_neg_pert) > 0:
        X_neg_pert_rot = pca_transform(np.asarray(X_neg_pert))

    # 4) Compute G: coverage of ALL positive perturbations by union of rectangles
    G = 0.0
    n_pos_pts = int(len(X_pos_pert_rot))
    n_pos_cov = 0
    if n_pos_pts > 0 and hyperrects.size > 0:
        for p in X_pos_pert_rot:
            hit = False
            for h in hyperrects:
                if contained(p, h):
                    hit = True
                    break
            if hit:
                n_pos_cov += 1
        G = float(n_pos_cov) / max(1, n_pos_pts)

    # 5) Compute F: % of rectangles that contain at least one negative perturbation
    F = 0.0
    n_rects = int(hyperrects.shape[0]) if hyperrects.ndim >= 3 else 0
    rects_hit = 0
    if n_rects > 0 and X_neg_pert_rot is not None and len(X_neg_pert_rot) > 0:
        for h in hyperrects:
            any_neg = False
            for q in X_neg_pert_rot:
                if contained(q, h):
                    any_neg = True
                    break
            if any_neg:
                rects_hit += 1
        F = float(rects_hit) / max(1, n_rects)

    detail = {
        "n_rects": n_rects,
        "n_pos_points": n_pos_pts,
        "n_pos_covered": n_pos_cov,
        "n_neg_points": int(0 if X_neg_pert is None else len(X_neg_pert)),
        "n_rects_with_neg": rects_hit,
    }
    return G, F, detail

# ====================== END: SafeNLP-compatible G & F helpers ======================


# --------------------------- Streamlit UI ---------------------------
st.set_page_config(
    page_title="SafeNLP – NLP Robustness & Cosine Filter",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    st.markdown("## ⚙️ Controls")

    datasets = find_datasets()
    if not datasets:
        st.error("No datasets found under `datasets/`. Expected `datasets/<name>/data/...`")
        st.stop()

    dataset_name = st.selectbox("Dataset", datasets, index=0)

    models_map = default_models()
    enc_key = st.selectbox("Encoding model ", list(models_map.keys()), index=0)
    enc_name = models_map[enc_key]

    st.markdown("---")
    st.markdown("### Perturbations")
    all_perturbations = ["character", "word"]
    selected_perturbations = st.multiselect(
        "Choose one or more",
        all_perturbations,
        default=["character"],
    )

    st.markdown("---")
    data_mode_label = st.radio(
        "Data mode",
        ["Auto (use if exist else build)", "Always use existing", "Rebuild now"],
        index=0,
        help=(
            "Auto: reuse files if present; otherwise generate and save.\n"
            "Always use existing: error if missing.\n"
            "Rebuild now: clean embedding folder and recompute this run."
        ),
    )
    mode_key = {
        "Auto (use if exist else build)": "auto",
        "Always use existing": "existing",
        "Rebuild now": "rebuild",
    }[data_mode_label]

    clean_before_rebuild = st.checkbox(
        "Clean embedding folder before rebuild",
        value=True,
        help="Prevents stale/partial files causing shape mismatches.",
    )

    threshold = st.slider("Cosine threshold", 0.0, 1.0, 0.25, 0.01)
    sample_preview = st.slider("Preview matches (per split)", 1, 20, 5, 1)

st.markdown(
    """
    # 🧮 SafeNLP: Robustness Cosine Filter
    Generate character/word perturbations and measure cosine similarity vs. originals.
    """
)

# --- Load dataset
with st.spinner("Loading dataset…"):
    data_obj = load_data(dataset_name, pos_lable_1_neg_lable_0, test_train_split_rate, path="datasets")
st.success(f"Loaded dataset **{dataset_name}**")

# --- Prepare artifacts
colA, colB = st.columns([1, 1], gap="large")
with colA:
    st.markdown("#### 1) Prepare artifacts")
    if selected_perturbations:
        for pert in selected_perturbations:
            get_perturbations(dataset_name, pert, mode_key, data_obj, path="datasets")
        st.success("Perturbations ready.")
    else:
        st.info("Select at least one perturbation in the sidebar.")

    # Ensure ORIGINAL embeddings are ready
    _ = get_embeddings(dataset_name, enc_key, enc_name, "original",
                       mode_key, data_obj, path="datasets",
                       clean_before_rebuild=clean_before_rebuild)
    st.success(f"Embeddings ready for **{enc_name}/original**")

with colB:
    st.markdown("#### 2) Compute cosine matches")
    go = st.button("Run cosine filtering")


# --------------------------- Core compute ---------------------------
def compute_for_one(dataset_name, enc_key, enc_name, perturbation, thr,
                    mode_key, data_obj, path="datasets",
                    clean_before_rebuild=True, sample_preview=5):
    # Ensure consistency between indexes and perturbed embeddings (repair if needed)
    ensure_consistency_or_repair(dataset_name, enc_key, enc_name, perturbation,
                                 mode_key, data_obj, path=path,
                                 clean_before_rebuild=clean_before_rebuild)

    # Load originals & perturbed embeddings according to mode
    X_train_pos, X_train_neg, *_ = get_embeddings(dataset_name, enc_key, enc_name,
                                                  "original", mode_key, data_obj,
                                                  path=path,
                                                  clean_before_rebuild=clean_before_rebuild)
    X_train_pos_p, X_train_neg_p, *_ = get_embeddings(dataset_name, enc_key, enc_name,
                                                      perturbation, mode_key, data_obj,
                                                      path=path,
                                                      clean_before_rebuild=clean_before_rebuild)

    # Index mapping arrays for who-perturbed-from-who
    train_pos_indexes = _safe_load(
        os.path.join(path, dataset_name, "perturbations", perturbation, "indexes", "train_pos_indexes.npy")
    )
    train_neg_indexes = _safe_load(
        os.path.join(path, dataset_name, "perturbations", perturbation, "indexes", "train_neg_indexes.npy")
    )

    # Also load sentence arrays for showing examples
    sdir = _pert_dir(path, dataset_name, perturbation, "sentences")
    X_train_pos_sent = _safe_load(os.path.join(sdir, "X_train_pos.npy"))
    X_train_neg_sent = _safe_load(os.path.join(sdir, "X_train_neg.npy"))

    p_total = len(X_train_pos_p)
    n_total = len(X_train_neg_p)
    pcount = 0
    ncount = 0

    pos_examples = []
    neg_examples = []

    # Positive split
    for i in range(len(X_train_pos)):
        idxs = np.where(train_pos_indexes == i)[0]
        for idx in idxs:
            if idx < p_total:
                cs = cos_sim_np(X_train_pos[i], X_train_pos_p[idx]).item()
                if cs > thr:
                    pcount += 1
                    if len(pos_examples) < sample_preview:
                        # recover original and perturbed text
                        orig_txt = None
                        pert_txt = None
                        try:
                            # originals come from load_data -> data_obj[0]
                            orig_txt = data_obj[0][i]
                        except Exception:
                            pass
                        try:
                            pert_txt = X_train_pos_sent[idx]
                        except Exception:
                            pass
                        pos_examples.append({
                            "orig_idx": int(i),
                            "pert_idx": int(idx),
                            "cosine": round(cs, 4),
                            "original": orig_txt,
                            "perturbed": pert_txt,
                        })

    # Negative split
    for i in range(len(X_train_neg)):
        idxs = np.where(train_neg_indexes == i)[0]
        for idx in idxs:
            if idx < n_total:
                cs = cos_sim_np(X_train_neg[i], X_train_neg_p[idx]).item()
                if cs > thr:
                    ncount += 1
                    if len(neg_examples) < sample_preview:
                        orig_txt = None
                        pert_txt = None
                        try:
                            orig_txt = data_obj[1][i]
                        except Exception:
                            pass
                        try:
                            pert_txt = X_train_neg_sent[idx]
                        except Exception:
                            pass
                        neg_examples.append({
                            "orig_idx": int(i),
                            "pert_idx": int(idx),
                            "cosine": round(cs, 4),
                            "original": orig_txt,
                            "perturbed": pert_txt,
                        })

    p_pct = (pcount / p_total * 100.0) if p_total else 0.0
    n_pct = (ncount / n_total * 100.0) if n_total else 0.0

    return {
        "Dataset": dataset_name,
        "Encoding Model": enc_name,
        "Perturbation": perturbation,
        "Positive": f"{pcount}/{p_total} ({p_pct:.2f}%)",
        "Negative": f"{ncount}/{n_total} ({n_pct:.2f}%)",
        "_p_count": pcount, "_p_total": p_total, "_p_pct": p_pct,
        "_n_count": ncount, "_n_total": n_total, "_n_pct": n_pct,
        "_pos_examples": pos_examples,
        "_neg_examples": neg_examples,
    }


# --------------------------- Run & render ---------------------------
if go and selected_perturbations:
    with st.spinner("Computing cosine scores…"):
        results = []
        for pert in selected_perturbations:
            # Ensure consistency (repairs automatically if allowed)
            ensure_consistency_or_repair(dataset_name, enc_key, enc_name, pert,
                                         mode_key, data_obj, path="datasets",
                                         clean_before_rebuild=clean_before_rebuild)
            # Compute
            results.append(
                compute_for_one(dataset_name, enc_key, enc_name, pert, threshold,
                                mode_key, data_obj, path="datasets",
                                clean_before_rebuild=clean_before_rebuild,
                                sample_preview=sample_preview)
            )

    # KPIs
    st.markdown("### Results")
    kcols = st.columns(len(results))
    for k, res in enumerate(results):
        with kcols[k]:
            st.metric(
                label=f"{res['Perturbation']} • Pos %",
                value=f"{res['_p_pct']:.2f}%",
                delta=f"{res['_p_count']}/{res['_p_total']}"
            )
            st.metric(
                label=f"{res['Perturbation']} • Neg %",
                value=f"{res['_n_pct']:.2f}%",
                delta=f"{res['_n_count']}/{res['_n_total']}"
            )

    table_df = pd.DataFrame([
        {
            "Dataset": r["Dataset"],
            "Encoding Model": r["Encoding Model"],
            "Perturbation": r["Perturbation"],
            "Positive": r["Positive"],
            "Negative": r["Negative"],
        } for r in results
    ])
    st.dataframe(table_df, width="stretch", height=140)

    # Example pairs
    st.markdown("### 🔎 Preview a few matching examples")
    for r in results:
        st.markdown(f"**{r['Perturbation']} – Positive examples**")
        if r["_pos_examples"]:
            pos_df = pd.DataFrame(r["_pos_examples"])
            st.dataframe(pos_df[["orig_idx", "pert_idx", "cosine", "original", "perturbed"]],
                         width="stretch", height=220)
        else:
            st.caption("None above threshold.")

        st.markdown(f"**{r['Perturbation']} – Negative examples**")
        if r["_neg_examples"]:
            neg_df = pd.DataFrame(r["_neg_examples"])
            st.dataframe(neg_df[["orig_idx", "pert_idx", "cosine", "original", "perturbed"]],
                         width="stretch", height=220)
        else:
            st.caption("None above threshold.")

    # Download
    csv_bytes = table_df.to_csv(index=False).encode()
    st.download_button(
        "Download results as CSV",
        data=csv_bytes,
        file_name=f"cosine_results_{dataset_name}_{enc_name}.csv",
        mime="text/csv",
        width="stretch",
    )

else:
    st.info("Choose perturbations and click **Run cosine filtering** to compute results.")



# ---- Generalisability & Embedding Error (per perturbation) ----
with st.expander("📏 Generalisability & Embedding Error (per perturbation)", expanded=True):
    st.caption(
        "We use SafeNLP's PCA rotation and hyper-rectangle builder. "
        "Rectangles are built in PCA space from each original + its perturbations "
        "filtered by the chosen cosine threshold (below). "
        "Then we compute:  \n"
        "• **Generalisability (G)**: % of positive perturbations covered by the union of rectangles  \n"
        "• **Embedding Error (F)**: % of rectangles that contain at least one *negative* perturbation"
    )
    use_filtered = st.checkbox("(Kept for compatibility) Build rectangles from τ-accepted points", value=True, key="gf_use_filtered")
    include_orig = st.checkbox("Include original embedding inside each rectangle (builder default)", value=True, key="gf_include_orig")
    pad_eps = st.number_input("Rectangle padding ε (for 'eps_cube' mode)", min_value=0.0, max_value=1e-2, value=1e-6, step=1e-6, format="%.6f", key="gf_pad_eps")

    # SafeNLP builder controls
    gf_cos_thr = st.slider("Cosine threshold for building rectangles (SafeNLP)", 0.0, 1.0, 0.60, 0.01, key="gf_cos_thr")
    gf_rebuild = st.checkbox("Rebuild rectangles now with this threshold", value=True, key="gf_rebuild")

    if st.button("Compute G & F", key="btn_gf"):
        gf_rows = []
        for pert in (selected_perturbations or []):
            # Ensure consistency (reuses your repair logic)
            ensure_consistency_or_repair(dataset_name, enc_key, enc_name, pert,
                                         mode_key, data_obj, path="datasets",
                                         clean_before_rebuild=clean_before_rebuild)

            # Load embeddings for ORIGINAL and this PERTURBATION
            Xp_o, Xn_o, *_ = get_embeddings(dataset_name, enc_key, enc_name,
                                            "original", mode_key, data_obj, path="datasets",
                                            clean_before_rebuild=clean_before_rebuild)
            Xp_p, Xn_p, *_ = get_embeddings(dataset_name, enc_key, enc_name,
                                            pert, mode_key, data_obj, path="datasets",
                                            clean_before_rebuild=clean_before_rebuild)

            # Index mappings (not used by SafeNLP builder, but kept for API parity)
            pos_idx = _safe_load(os.path.join("datasets", dataset_name, "perturbations", pert, "indexes", "train_pos_indexes.npy"))

            # SafeNLP-compatible G & F (rotated + rectangles)
            G, F, detail = compute_generalisability_and_embedding_error(
                X_pos_orig=Xp_o,
                X_pos_pert=Xp_p,
                pos_src_idx=pos_idx,
                X_neg_pert=Xn_p,
                use_filtered=use_filtered,
                pos_accept_mask=None,
                include_orig=include_orig,
                pad_eps=float(pad_eps),
                dataset_name=dataset_name,
                encoder_name=enc_name,
                hyperrectangles_name=pert,     # build rectangles for this perturbation family
                path="datasets",
                cosine_threshold=gf_cos_thr,
                rebuild_now=gf_rebuild
            )

            gf_rows.append({
                "Dataset": dataset_name,
                "Encoder": enc_name,
                "Perturbation": pert,
                "G (pos coverage %)": round(100.0 * G, 2),
                "F (rects w/ neg %)": round(100.0 * F, 3),
                "#Rects": detail["n_rects"],
                "#Pos perts": detail["n_pos_points"],
                "#Pos covered": detail["n_pos_covered"],
                "#Neg perts": detail["n_neg_points"],
                "#Rects hit by neg": detail["n_rects_with_neg"],
            })

        if gf_rows:
            gf_df = pd.DataFrame(gf_rows)
            st.dataframe(gf_df, width="stretch", height=160)
            st.download_button(
                "Download G&F as CSV",
                data=gf_df.to_csv(index=False).encode(),
                file_name=f"G_F_{dataset_name}_{enc_name}.csv",
                mime="text/csv",
                width="stretch",
            )
        else:
            st.info("No perturbations selected.")


# ---- Threshold sweep UI ----
with st.expander("🔬 Threshold sweep (ROC / PR)", expanded=False):
    enable_sweep = st.checkbox("Enable threshold sweep", value=True)
    negs_per_pert = st.number_input("Negatives per perturbed (hard impostors)", min_value=1, max_value=10, value=1, step=1)
    col_tau1, col_tau2, col_tau3 = st.columns(3)
    with col_tau1:
        tau_min = st.number_input("τ min", min_value=0.0, max_value=1.0, value=0.30, step=0.01, format="%.2f")
    with col_tau2:
        tau_max = st.number_input("τ max", min_value=0.0, max_value=1.0, value=0.95, step=0.01, format="%.2f")
    with col_tau3:
        tau_step = st.number_input("τ step", min_value=0.001, max_value=0.2, value=0.01, step=0.001, format="%.3f")
    seed = st.number_input("Random seed (neg sampling)", min_value=0, max_value=1_000_000, value=42, step=1)
    run_sweep_btn = st.button("Run threshold sweep")


# --------------------------- Threshold sweep runner ---------------------------
if 'enable_sweep' in locals() and enable_sweep and selected_perturbations and run_sweep_btn:
    all_summaries = []
    with st.spinner("Running threshold sweep(s)…"):
        for pert in selected_perturbations:
            comb_df, sum_df = run_threshold_sweep(
                dataset_name, enc_key, enc_name, pert,
                mode_key=mode_key, data_obj=data_obj, path="datasets",
                tau_min=tau_min, tau_max=tau_max, tau_step=tau_step,
                negs_per_pert=negs_per_pert, seed=seed
            )
            st.success(f"Sweep done for **{pert}**  •  rows: {len(comb_df)}")
            # Show key numbers
            st.markdown(f"**{pert} — summary**")
            st.dataframe(sum_df, width="stretch", height=120)

            # Downloads for this perturbation
            sweep_csv = comb_df.to_csv(index=False).encode()
            sum_csv = sum_df.to_csv(index=False).encode()
            c1, c2 = st.columns(2)
            with c1:
                st.download_button(
                    f"Download sweep CSV ({pert})",
                    data=sweep_csv,
                    file_name=f"sweep_{dataset_name}_{enc_name}_{pert}.csv",
                    mime="text/csv",
                    width="stretch",
                )
            with c2:
                st.download_button(
                    f"Download summary CSV ({pert})",
                    data=sum_csv,
                    file_name=f"summary_{dataset_name}_{enc_name}_{pert}.csv",
                    mime="text/csv",
                    width="stretch",
                )
            all_summaries.append(sum_df)
