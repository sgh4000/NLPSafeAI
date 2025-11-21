import os
import numpy as np
import onnxruntime as ort


def _load_onnx_session(onnx_path: str) -> ort.InferenceSession:
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX model not found at: {onnx_path}")
    # Basic CPU session
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    return sess


def _onnx_predict(sess: ort.InferenceSession, X: np.ndarray) -> np.ndarray:
    """
    Run ONNX model on a batch of inputs X with shape (N, 30).
    Returns predicted class indices (argmax over logits).
    """
    # Get input name (assumes single input)
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: X.astype(np.float32)})
    logits = outputs[0]  # shape (N, 2)
    preds = np.argmax(logits, axis=1)
    return preds


def evaluate_semantic_stability(
    dataset_name: str,
    encoding_model_name: str,
    path: str = "datasets",
    hyperrectangles_name: str = "semantic",
    base_onnx_path: str = "results/base.onnx",
    adv_onnx_path: str = "results/adversarial.onnx",
    num_samples_per_hr: int = 10,
    random_seed: int = 42,
):
    """
    Empirical semantic robustness evaluation:

    For each semantic hyperrectangle:
      - Sample 'num_samples_per_hr' points uniformly inside it.
      - For each model (base, adversarial), check if predictions
        are constant across all sampled points.
      - Count how many hyperrectangles are 'stable' for each model.

    Prints stability rates for base and adversarial models.
    """

    rng = np.random.default_rng(random_seed)

    # -------------------------------------------------
    # 1) Load semantic hyperrectangles
    # -------------------------------------------------
    hr_path = os.path.join(path, dataset_name, "hyperrectangles", encoding_model_name, f"{hyperrectangles_name}.npy")
    if not os.path.exists(hr_path):
        raise FileNotFoundError(f"Semantic hyperrectangles file not found at: {hr_path}")

    hyperrectangles = np.load(hr_path)  # shape: (N_hr, 30, 2)
    if hyperrectangles.ndim != 3 or hyperrectangles.shape[2] != 2:
        raise ValueError(
            f"Expected hyperrectangles of shape (N, 30, 2), got {hyperrectangles.shape}"
        )

    n_hr, dim, _ = hyperrectangles.shape
    print(f"[EVAL] Loaded {n_hr} semantic hyperrectangles of dimension {dim} from: {hr_path}")

    # -------------------------------------------------
    # 2) Load ONNX models
    # -------------------------------------------------
    print(f"[EVAL] Loading base model from: {base_onnx_path}")
    base_sess = _load_onnx_session(base_onnx_path)

    print(f"[EVAL] Loading adversarial model from: {adv_onnx_path}")
    adv_sess = _load_onnx_session(adv_onnx_path)

    # -------------------------------------------------
    # 3) Loop over hyperrectangles and sample points
    # -------------------------------------------------
    base_stable = 0
    adv_stable = 0

    for i in range(n_hr):
        # Each HR is [min, max] in each dimension
        mins = hyperrectangles[i, :, 0]
        maxs = hyperrectangles[i, :, 1]

        # If any dimension is degenerate (min == max), sampling is trivial
        # but this still works.
        width = maxs - mins

        # Sample points: mins + U(0,1)*width
        U = rng.random((num_samples_per_hr, dim))
        samples = mins + U * width  # shape (num_samples_per_hr, dim)

        # Get predictions from both models
        base_preds = _onnx_predict(base_sess, samples)
        adv_preds = _onnx_predict(adv_sess, samples)

        # Check if all predictions are the same in this HR
        if np.all(base_preds == base_preds[0]):
            base_stable += 1
        if np.all(adv_preds == adv_preds[0]):
            adv_stable += 1

    # -------------------------------------------------
    # 4) Compute stability rates
    # -------------------------------------------------
    base_stability = base_stable / n_hr
    adv_stability = adv_stable / n_hr

    print("\n[EVAL] Semantic stability (empirical, via sampling):")
    print(f"       Base model:         {base_stable}/{n_hr} hyperrectangles stable "
          f"({base_stability:.3f})")
    print(f"       Adversarial model:  {adv_stable}/{n_hr} hyperrectangles stable "
          f"({adv_stability:.3f})")

    # return them if you want to use them elsewhere
    return {
        "n_hyperrectangles": n_hr,
        "base_stable": base_stable,
        "adv_stable": adv_stable,
        "base_stability": base_stability,
        "adv_stability": adv_stability,
    }
