model advModel is "src/results/adversarial.onnx"

property verifySemanticAdv is {
    include "datasets/depression/properties/marabou/sbert22M/semantic/adversarial/semantic_*.vcl"
}
