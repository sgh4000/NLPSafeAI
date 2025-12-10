import shap
import numpy as np
import matplotlib.pyplot as plt

# This function is creating a plot of global SHAP feature importance
def feature_importance_graph(X_train, X_test, onnx_session, to_explain=100):
   
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

# Return prediction
def predict_labels_onnx(onnx_session, X):
    input_name = onnx_session.get_inputs()[0].name
    X_batch = X.astype(np.float32)
    logits = onnx_session.run(None, {input_name: X_batch})[0]
    return np.argmax(logits, axis=1)

# Generating the hyperrectangles, using global SHAP importance values
def generate_feature_importance_driven_hyperrectangles(X_train, X_test, y_test, onnx_session, model, n_background=10, n_explain=100,unimportant_threshold=0,unimportant_width=0.2):

    input_name = onnx_session.get_inputs()[0].name

    # Take a small subset of the training data to explain using SHAP
    background = X_train[np.random.choice(X_train.shape[0], n_background, replace=False)]
    to_explain = X_train[:n_explain]

    # Define a predict function for SHAP
    def predict_fn(X):
        X_float = X.astype(np.float32)
        preds = onnx_session.run(None, {input_name: X_float})[0]
        return preds

    shap_file = f"shap_values{n_background}-{model}.npy"

    # Check if SHAP values were already computed
    try:
        shap_values = np.load(shap_file)
        print("Loaded SHAP values from disk.")
    except FileNotFoundError:
        print("Computing SHAP values...")
        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(to_explain)
        
        # Save for later use
        np.save(shap_file, shap_values)
        print(f"Saved SHAP values to {shap_file}")

    global_importance = np.mean(np.abs(shap_values), axis=0)

    pca_train_min = np.min(X_train, axis=0)
    pca_train_max = np.max(X_train, axis=0)

    y_pred = predict_labels_onnx(onnx_session, X_test)
    # Find where classification is correct
    correct_idx = np.where(y_pred == y_test)[0]

    # Only want to be checking the robustness for the correctly classified inputs
    X_val = X_test[correct_idx]
    X_val = X_val[:n_explain]
    y_val = y_test[correct_idx]
    y_val = y_val[:n_explain]

    rectangles = []

    n_features = 30

    for i, (z0, y0) in enumerate(zip(X_val, y_val)):

        target_class = int(y0)

        # For the given target class, work out which features are globally unimportant, given the specified threshold
        importance_for_class = global_importance[:, target_class]
        unimportant_mask = (importance_for_class <= unimportant_threshold)

        w = np.zeros(n_features)
        w[unimportant_mask] = unimportant_width

        # Vary according to the unimportance width value
        L = z0 - w
        U = z0 + w

        # Clip to training range per feature, want to ensure each feature value stays within the range observed during training
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
    

# Helper function, turning each Lower and Upper bound value into a vehicle bound, for an individual PCA feature
def hyperrect_to_vehicle(L, U, variable_name="x", precision=6):
    lines = []

    for i, (l, u) in enumerate(zip(L,U)):
        l_str = f"{l:.{precision}f}"
        u_str = f"{u:.{precision}f}"
        lines.append(f"({variable_name} ! {i} >= {l_str} and {variable_name} ! {i} <= {u_str})")

    return " and\n      ".join(lines)

# Helper function, to be used to create property specification which is compatible with Vehicle
def rectangle_to_vehicle_property(r):

    idx = r["index"]

    label = r["label"]
    if(label) == 1:
        class_query = "isClassifiedDepression x"
    else:
        class_query = "not (isClassifiedDepression x)"
    L = r["L"]
    U = r["U"]

    property_prefix = "property"
    variable_name = "x"

    property_spec = hyperrect_to_vehicle(L,U,variable_name)
    property_name = f"{property_prefix}{idx}"

    prop_str = f"""
    @property \n{property_name} : Bool \n{property_name} =
        forall {variable_name} .
            ({property_spec})
            => {class_query}
    """
    return prop_str.strip()

    