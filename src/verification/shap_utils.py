import shap
import numpy as np
import matplotlib.pyplot as plt

def feature_importance_graph(X_train, X_test, onnx_session):
   
    input_name = onnx_session.get_inputs()[0].name

    # Take a small subset of the training data to explain using SHAP
    background = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]
    to_explain = X_test[:100]

    # Define a predict function for SHAP
    def predict_fn(X):
        X_float = X.astype(np.float32)
        preds = onnx_session.run(None, {input_name: X_float})[0]
        return preds

    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(to_explain)

    # Measure how features push towards class 1
    shap_diff = shap_values[..., 1] - shap_values[..., 0]

    shap.summary_plot(
        shap_diff, to_explain,
        feature_names=[f"f{i}" for i in range(X_train.shape[1])]
    )
    return shap_diff

def predict_labels_onnx(onnx_session, X):
    input_name = onnx_session.get_inputs()[0].name
    X_batch = X.astype(np.float32)
    logits = onnx_session.run(None, {input_name: X_batch})[0]
    return np.argmax(logits, axis=1)


def generate_feature_importance_driven_hyperrectangles(X_train, X_test, y_test, onnx_session, n_background=10, n_explain=100,n_points=20,unimportant_threshold=0.0,unimportant_width=0.0):

    input_name = onnx_session.get_inputs()[0].name

    # Take a small subset of the training data to explain using SHAP
    background = X_train[np.random.choice(X_train.shape[0], n_background, replace=False)]
    to_explain = X_test[:n_explain]

    # Define a predict function for SHAP
    def predict_fn(X):
        X_float = X.astype(np.float32)
        preds = onnx_session.run(None, {input_name: X_float})[0]
        return preds

    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(to_explain)
    print("shap values")
    print(shap_values)

    pca_train_min = np.min(X_train, axis=0)
    pca_train_max = np.max(X_train, axis=0)

    y_pred = predict_labels_onnx(onnx_session, X_test)
    mis_idx = np.where(y_pred != y_test)[0]
    correct_idx = np.where(y_pred == y_test)[0]

    # Only want to be checking the robustness for the correctly classified inputs
    X_val = X_test[correct_idx]
    y_val = y_test[correct_idx]

    rectangles = []

    for i, (z0, y0) in enumerate(zip(X_val, y_val)):

        target_class = int(y0)
        phi = shap_values[i, :, target_class]

        importance = np.abs(phi)
        print("importance: ", importance)
        unimportant_mask = (importance <= unimportant_threshold)

        w = np.zeros_like(importance, dtype=float)
        w[unimportant_mask] = unimportant_width

        L = z0 - w
        U = z0 + w

        # Clip to training range per feature
        L = np.maximum(L, pca_train_min)
        U = np.minimum(U, pca_train_max)

        rectangles.append({
            "index": i,
            "label": target_class,
            "center": z0,
            "L": L,
            "U": U
        })

    return rectangles

def hyperrect_to_vehicle(L, U, variable_name="x", precision=6):
    lines = []

    for i, (l, u) in enumerate(zip(L,U)):
        l_str = f"{l:.{precision}f}"
        u_str = f"{u:.{precision}f}"
        lines.append(f"({variable_name} ! {i} >= {l_str} and {variable_name} ! {i} <= {u_str})")

    return " and\n      ".join(lines)

def rectangle_to_vehicle_property(r):

    idx = r["index"]

    label = r["label"]
    if(label) == 1:
        class_query = "isClassifiedDepression x"
    else:
        class_query = "not (isClassifiedDepression x)"
    L = r["L"]
    U = r["U"]

    print("max |L-U|: ", np.max(np.abs(L-U)))
    print("max |L-center|: ", np.max(np.abs(L - r["center"])))
    print(r)
    property_prefix = "property"
    variable_name = "x"

    property_spec = hyperrect_to_vehicle(L,U,variable_name)
    property_name = f"{property_prefix}{idx}"

    prop_str = f"""
    {property_name} : Bool
    {property_name} =
        forall {variable_name} .
            ({property_spec})
            => {class_query}
    """

    return prop_str.strip()