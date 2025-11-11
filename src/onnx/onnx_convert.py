from tensorflow import keras

import tf2onnx
import onnx

output_path = "my_model.onnx"

model = keras.models.load_model("model_base.h5", compile=False)


model_proto, _ = tf2onnx.convert.from_keras(model, opset=13)
onnx.save(model_proto, output_path)
print(f"ONNX model saved to {output_path}")
