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

n_explain_graph = 2000

# feature_importance_graph(X_train, X_test, session, n_explain_graph)

# Generating the explainability-driven hyperrectangles, which can then be used for Vehicle verification

n_background = 5000
n_explain = 100
unimportant_threshold = 0.02
unimportant_width = 0.2

rects = generate_feature_importance_driven_hyperrectangles(X_train, X_test, y_test, session, n_background=n_background, n_explain=n_explain,unimportant_threshold=unimportant_threshold,unimportant_width=unimportant_width)

vehicle_properties = [rectangle_to_vehicle_property(r) for r in rects]

for x in vehicle_properties:
    print(x)

filename = f"hyperrectangle_properties/{n_background}_{n_explain}_{unimportant_threshold}_{unimportant_width}.txt"

with open(filename, "w") as f:
  for x in vehicle_properties:
    f.write(x+"\n")
