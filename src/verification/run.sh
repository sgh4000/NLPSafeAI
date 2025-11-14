#!/bin/bash

vehicle verify \
  --specification safeAI_Verify.vcl \
  --verifier Marabou \
  --network safeNLP:../results/adversarial.onnx \
  --property property1

vehicle verify \
  --specification safeAI_Verify.vcl \
  --verifier Marabou \
  --network safeNLP:../results/adversarial.onnx \
  --property property3

vehicle verify \
  --specification safeAI_Verify.vcl \
  --verifier Marabou \
  --network safeNLP:../results/adversarial.onnx \
  --property property4

vehicle verify \
  --specification safeAI_Verify.vcl \
  --verifier Marabou \
  --network safeNLP:../results/adversarial.onnx \
  --property property17

vehicle verify \
  --specification safeAI_Verify.vcl \
  --verifier Marabou \
  --network safeNLP:../results/adversarial.onnx \
  --property property18

vehicle verify \
  --specification safeAI_Verify.vcl \
  --verifier Marabou \
  --network safeNLP:../results/adversarial.onnx \
  --property property19