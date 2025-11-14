import os
os.environ["TRANSFORMERS_NO_TF"] = "1"  # Prevent TF import

import pickle
import numpy as np
import onnxruntime as ort
from shap_utils import feature_importance_graph, generate_feature_importance_driven_hyperrectangles, rectangle_to_vehicle_property

# Load data
with open("../data/train_test_data.pkl", "rb") as f:
    data = pickle.load(f)

X_train = data["X_train"]
X_test = data["X_test"]
y_test = data["y_test"]

# Load ONNX model
onnx_path = "../results/adversarial.onnx"
session = ort.InferenceSession(onnx_path)

# SHAP feature importance
feature_importance_graph(X_train, X_test, session)

rects = generate_feature_importance_driven_hyperrectangles(X_train, X_test, y_test, session)

vehicle_properties = [rectangle_to_vehicle_property(r) for r in rects]

for x in vehicle_properties:
    print("\n\n--------")
    print(x)
    print("\n\n--------")