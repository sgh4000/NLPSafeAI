# Inspired from: https://github.com/ANTONIONLP/ANTONIO/blob/main/src/property_parser.py
import os
import numpy as np
import onnxruntime as ort
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_onnx_session(onnx_path: str) -> ort.InferenceSession:
    onnx_path = str(onnx_path)
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX model not found at: {onnx_path}")
    return ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])


def _onnx_predict_label(sess: ort.InferenceSession, x: np.ndarray) -> int:
    """Predict label for a single PCA point x shape (30,)"""
    input_name = sess.get_inputs()[0].name
    logits = sess.run(None, {input_name: x.astype(np.float32)[None, :]})[0][0]
    return int(np.argmax(logits))

def _vehicle_box_formula(mins, maxs):
    """Builds ((x!0>=min0 and x!0<=max0) and ... )"""
    clauses = []
    for j in range(len(mins)):
        clauses.append(f"(x ! {j} >= {mins[j]} and x ! {j} <= {maxs[j]})")
    return "(" + " and ".join(clauses) + ")"

def _vehicle_predicate_name(expected_label: int):
    """Map label to predicate name."""
    # label 1 = depressed, label 0 = non-depressed
    return "isClassifiedDepression" if expected_label == 1 else "isClassifiedNonDepression"

def parse_semantic_properties_marabou(
    dataset_name="depression",
    encoding_model_name="sbert22M",
    hyperrectangles_name="semantic",
    path="datasets",
    onnx_path="results/base.onnx",   # use base OR adversarial here
    model_tag="base",               
    out_dir=None,
    max_hr=None,                     # for quick sanity check, e.g. 200
):
    """
    Create Marabou-style property files from semantic hyperrectangles.

    For each HR:
      - Take HR center as representative point.
      - Use ONNX model to decide expected label at center.
      - Write Marabou constraints:
            x j >= min_j
            x j <= max_j
            y_expected >= y_other

    This matches semantic robustness notion:
      "Prediction should not change inside the semantic HR".
    """

    # -------------------------
    # 1) Load semantic HRs
    # -------------------------
    hr_path = PROJECT_ROOT / path / dataset_name / "hyperrectangles" / encoding_model_name / f"{hyperrectangles_name}.npy"
    hr_path = str(hr_path)

    if not os.path.exists(hr_path):
        raise FileNotFoundError(f"Semantic HR file not found at: {hr_path}")

    hyperrectangles = np.load(hr_path)
    n_hr, dim, _ = hyperrectangles.shape
    print(f"[PROP] Loaded {n_hr} semantic HRs from {hr_path}")

    if max_hr is not None:
        n_hr = min(n_hr, max_hr)
        hyperrectangles = hyperrectangles[:n_hr]
        print(f"[PROP] Limiting to first {n_hr} HRs (sanity check).")

    # -------------------------
    # 2) Load ONNX model
    # -------------------------
    onnx_path = PROJECT_ROOT / onnx_path
    sess = _load_onnx_session(onnx_path)
    print(f"[PROP] Using model: {onnx_path}")

    # -------------------------
    # 3) Output directory
    # -------------------------
    if out_dir is None:
        out_dir = PROJECT_ROOT / path / dataset_name / "properties" / "marabou" / encoding_model_name / hyperrectangles_name / model_tag
    out_dir = str(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # -------------------------
    # 4) Write each property
    # -------------------------
    for i, cube in enumerate(hyperrectangles):
        mins = cube[:, 0]
        maxs = cube[:, 1]
        center = (mins + maxs) / 2.0

        expected_label = _onnx_predict_label(sess, center)
        other_label = 1 - expected_label

        label_tag = "depressed" if expected_label == 1 else "nondepressed"
        prop_path = os.path.join(out_dir, f"{hyperrectangles_name}@{i}_{label_tag}.vcl")

       
        with open(prop_path, "w") as f:
            # header comments for clarity
            f.write(f"-- semantic HR {i}\n")
            f.write(f"-- expected label at center = {expected_label} "
                f"({'depressed' if expected_label==1 else 'non-depressed'})\n\n")
    
        box_formula = _vehicle_box_formula(mins, maxs)
        pred_name = _vehicle_predicate_name(expected_label)
        # final Vehicle property
        f.write(f"{box_formula} => {pred_name} x\n")
        
    print(f"[PROP] Done. Wrote {n_hr} Marabou properties to:\n       {out_dir}")
    return out_dir
