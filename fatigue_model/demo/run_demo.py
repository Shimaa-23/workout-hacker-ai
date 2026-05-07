
"""
run_demo.py
===========

Simulates a workout session and streams predictions from the fatigue model.

This demo:
- Generates synthetic EMG + IMU feature values
- Gradually increases fatigue over time
- Shows real-time model predictions

Run:
    python demo/run_demo.py
"""

import time
import numpy as np
from fatigue_predictor import FatiguePredictor


def simulate_sensor_data(step, feature_names):
    """
    Generate synthetic feature values.

    Fatigue increases over time:
    - EMG RMS increases
    - Movement becomes more unstable
    """

    data = {}

    for f in feature_names:
        if "emg" in f:
            # EMG increases with fatigue
            data[f] = np.random.uniform(0.02, 0.05) + step * 0.002
        elif "acce" in f or "gyro" in f:
            # IMU becomes noisier / unstable
            data[f] = np.random.uniform(0.5, 1.5) + step * 0.05
        else:
            # fallback
            data[f] = np.random.uniform(0.1, 1.0)

    return data


def main():
    predictor = FatiguePredictor()

    print("=" * 60)
    print("WorkoutHacker – Fatigue Detection Demo")
    print("=" * 60)
    print("\nSimulating workout session...\n")

    steps = 20  # number of time steps

    for step in range(steps):
        sample = simulate_sensor_data(step, predictor.feature_names)

        result = predictor.predict(sample)

        print(
            f"[Step {step:02d}] "
            f"Fatigue: {result['fatigue_level']:10s} | "
            f"Confidence: {result['confidence']:.2f}"
        )

        time.sleep(0.5)

    print("\nSession complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
