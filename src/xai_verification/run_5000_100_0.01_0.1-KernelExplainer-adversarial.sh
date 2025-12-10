#!/bin/bash

SPEC="safeAI_Verify_5000_100_0.01_0.1-KernelExplainer-adversarial.vcl"
NETWORK="safeNLP:../results/adversarial.onnx"
RESULT_FILE="safeAI_Verify_5000_100_0.01_0.1-KernelExplainer-adversarial_results.txt"

> "$RESULT_FILE"

for i in {0..99}
do
    echo "=== Verifying property$i ===" | tee -a "$RESULT_FILE"

    # Capture output of vehicle verify
    OUTPUT=$(timeout 180s vehicle verify \
      --specification "$SPEC" \
      --verifier Marabou \
      --network "$NETWORK" \
      --property "property$i" 2>&1)

    # Print everything to terminal and log file
    echo "$OUTPUT" | tee -a "$RESULT_FILE"

    # Look for counterexample
    if echo "$OUTPUT" | grep -q "found a counterexample"; then
        echo "property$i: COUNTEREXAMPLE FOUND" | tee -a "$RESULT_FILE"
    elif echo "$OUTPUT" | grep -q "proved no counterexample exists"; then
        echo "property$i: NO COUNTEREXAMPLE" | tee -a "$RESULT_FILE"
    else
        echo "property$i: UNKNOWN RESULT - TIMEOUT" | tee -a "$RESULT_FILE"
    fi

    echo | tee -a "$RESULT_FILE"
done
