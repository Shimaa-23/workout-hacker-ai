"""
smoke_test.py
=============

Verifies that the trained fatigue model artefacts can be loaded and used
for inference without access to the original dataset.

Run after:
    python train_and_save.py

Purpose:
- Validate model loading
- Validate prediction pipeline (dict + array + batch)
- Ensure feature alignment is correct
- Provide a quick sanity check for integration teams

Usage:
    python smoke_test.py
"""

import numpy as np
from fatigue_predictor import FatiguePredictor

print("=" * 55)
print("WorkoutHacker – Fatigue Model Smoke Test")
print("=" * 55)


# Load predictor

predictor = FatiguePredictor()

print(f"\n✓ Model loaded successfully")
print(f"  Version       : {predictor.metadata.get('version', 'N/A')}")
print(f"  Features used : {predictor.metadata['n_features']}")
print(f"  Accuracy      : {predictor.metadata['overall_accuracy']:.4f}")
print(f"  F1 (macro)    : {predictor.metadata['f1_macro']:.4f}")
print(f"  G-Mean        : {predictor.metadata['g_mean']:.4f}")


# Test 1: dict input (missing features scenario)

print("\n── Test 1: dict input with partial features ──")

# Simulate missing sensor values (realistic scenario)
partial_dict = {
    predictor.feature_names[0]: 0.1,
    predictor.feature_names[1]: 0.2,
}

result = predictor.predict(partial_dict)

print(f"  Prediction : {result['fatigue_level']} (label={result['label_int']})")
print(f"  Confidence : {result['confidence']:.4f}")

# Ensure valid output
assert result["fatigue_level"] in ("low", "moderate", "high", "very_high")

print("  ✓ passed (mean imputation working)")


# Test 2: numpy array input

print("\n── Test 2: numpy array input ──")

random_arr = np.random.rand(predictor.metadata["n_features"])

result2 = predictor.predict(random_arr)

print(f"  Prediction : {result2['fatigue_level']} (label={result2['label_int']})")
print(f"  Confidence : {result2['confidence']:.4f}")

# Probabilities should sum to ~1
prob_sum = sum(result2["probabilities"].values())
assert 0.99 <= prob_sum <= 1.01

print("  ✓ passed")


# Test 3: batch prediction

print("\n── Test 3: batch prediction (5 samples) ──")

batch_arr = np.random.rand(5, predictor.metadata["n_features"])
results = predictor.predict_batch(batch_arr)

for i, r in enumerate(results):
    print(f"  Sample {i}: {r['fatigue_level']:10s}  confidence={r['confidence']:.3f}")

assert len(results) == 5

print("  ✓ passed")


# Test 4: feature integrity

print("\n── Test 4: feature list integrity ──")

assert len(predictor.feature_names) == predictor.metadata["n_features"]

print(f"  Feature names ({len(predictor.feature_names)}):")
for fn in predictor.feature_names:
    print(f"    {fn}")

print("  ✓ passed")

#
# Test 5: edge case – wrong input size

print("\n── Test 5: invalid input handling ──")

try:
    predictor.predict([0.1, 0.2])  # wrong size
    raise AssertionError("Expected failure did not occur")
except ValueError:
    print("  ✓ passed (invalid input correctly rejected)")


# Done

print("\n" + "=" * 55)
print("All tests passed ✓  The model is ready to integrate.")
print("=" * 55)