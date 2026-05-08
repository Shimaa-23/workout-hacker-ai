# WorkoutHacker – Rep Counter & Tempo Module

**Team 25 | Alexandria University | Faculty of Computers and Data Science**

This module provides **real-time exercise classification**, **rep counting**, and **tempo analysis** using pose landmarks from MediaPipe.


---

##  Quick Start (Integration Team)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 2. Run the test script

```bash
python test.py
```

Opens your webcam and runs live rep counting + tempo analysis. Change the `EXERCISE` variable at the top of `test.py` to test a specific exercise:

```python
EXERCISE = "bicep_curl"  # or any supported exercise
```

---

## 🧠 Integration Guide

### 🔹 Rep Counting (Auto-detect exercise)

```python
from rep_counter_interface import RepCounterInterface

rc = RepCounterInterface("Combined_model.pth")

# Call every frame with MediaPipe landmarks
state = rc.update(results.pose_landmarks.landmark)

print(state)
# {
#   'exercise': 'bicep_curl',
#   'reps': 5,
#   'confidence': 0.87,
#   'phase': 'UP'
# }
```

---

### 🔹 Rep Counting 

Useful for testing or when the exercise is already known from user input:

```python
rc = RepCounterInterface("Combined_model.pth", force_exercise="shoulder_press")
```

---

### 🔹 Tempo Analysis

```python
from tempo import TempoAnalyzer

tempo_ai = TempoAnalyzer("tempo_classifier.pkl", "tempo_config.json")

# Set exercise once per session (or update every frame)
tempo_ai.set_exercise(state['exercise'])

# Call every frame with current phase and FPS
tempo_ai.update(state['phase'], fps=30)

feedback = tempo_ai.get_state()
print(feedback)
# {'tempo': 'normal', 'quality': 85}
```

---

### 🔹 Full Pipeline (Rep Counter + Tempo together)

```python
from rep_counter_interface import RepCounterInterface
from tempo import TempoAnalyzer

rc       = RepCounterInterface("Combined_model.pth")
tempo_ai = TempoAnalyzer("tempo_classifier.pkl", "tempo_config.json")

# Inside your frame loop:
state    = rc.update(results.pose_landmarks.landmark)
tempo_ai.set_exercise(state['exercise'])
tempo_ai.update(state['phase'], fps=30)
feedback = tempo_ai.get_state()
```

---

## 🏋️ Supported Exercises

| Exercise           | Rep Logic                        |
| ------------------ | -------------------------------- |
| `bicep_curl`       | Elbow angle + swing guard        |
| `tricep_extension` | Elbow angle + wrist-near-head    |
| `front_raise`      | Wrist vs shoulder height         |
| `lateral_raise`    | Wrist vs shoulder height         |
| `shoulder_press`   | Wrist vs nose height             |
| `push_up`          | Elbow angle                      |
| `pull_up`          | Nose vs wrist height             |
| `bench_pressing`   | Elbow angle                      |

---

## ⏱️ Tempo Classes

| Tempo    | Meaning                        |
| -------- | ------------------------------ |
| `fast`   | Rep completed in < 1 second    |
| `normal` | Rep completed in 1–3.5 seconds |
| `slow`   | Rep completed in > 3.5 seconds |

The `quality` field (0–100) reflects model confidence in the tempo prediction.

---


---

## 🔗 System Context

* MediaPipe extracts 33 pose landmarks per frame
* `RepCounterInterface` classifies exercise + counts reps
* `TempoAnalyzer` tracks rep duration and classifies tempo
