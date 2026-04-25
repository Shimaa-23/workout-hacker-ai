# WorkoutHacker – Fatigue Detection Module

**Team 25 | Alexandria University | Faculty of Computers and Data Science**

This module contains the trained **Random Forest fatigue classifier** used in the WorkoutHacker IoT gym assistant.

It predicts user fatigue level using **EMG (muscle activity)** and **IMU (movement dynamics)** features, enabling real-time feedback for safer and more effective training.

---

## 🚀 Quick Start (Integration Team)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 2. Train & save the model *(one-time, ML team only)*

Download dataset:
https://figshare.com/articles/dataset/Dataset/15104079?file=29037888

Place it in:

```
data/database.xlsx
```

Run:

```bash
python train_and_save.py
```

This generates the following artefacts inside `model/`:

| File                      | Purpose                         |
| ------------------------- | ------------------------------- |
| `fatigue_rf_model.joblib` | Trained model                   |
| `feature_list.json`       | Ordered feature list            |
| `feature_means.json`      | Mean values for safe imputation |
| `label_map.json`          | Label → fatigue level           |
| `model_metadata.json`     | Performance + version info      |

> ✅ **Commit the `model/` folder** so other teams don’t need to retrain.

---

### 3. Run smoke test

```bash
python smoke_test.py
```

Ensures:

* Model loads correctly
* Predictions work
* Feature alignment is valid

---

## 🧠 Integration Guide

```python
from fatigue_predictor import FatiguePredictor

predictor = FatiguePredictor()
```

---

### 🔹 Single Prediction (dict input)

```python
sensor_reading = {
    "emg_rms_rectusFemoris": 0.042,
    "emg_rms_bicepsFemoris": 0.038,
    "gait_median_acce": 1.23,
}

result = predictor.predict(sensor_reading)
print(result)
```

---

### 🔹 Batch Prediction

```python
results = predictor.predict_batch([sample1, sample2])
```

---

### 🔹 Required Feature Order

```python
print(predictor.feature_names)
```

---

## ⚙️ Design Decisions

### ✔ No Feature Scaling

Random Forest is a tree-based model and **does not require feature scaling**.
The model is trained and used without scaling to ensure consistent behavior between training and inference.

---

### ✔ Missing Feature Handling

Missing features are filled using **training-set mean values**, not zeros.

This improves robustness when:

* Sensors temporarily fail
* Partial data is received

---

### ✔ Feature Extraction

Feature extraction (EMG RMS, IMU statistics) is performed on the **ESP32 firmware** in real-time.

For reproducibility and testing, a Python implementation can be added.

---

## 🧪 Model Performance

| Metric        | Value         |
| ------------- | ------------- |
| Accuracy      | **0.9469**    |
| F1 (macro)    | **0.9471**    |
| G-Mean        | **0.9434**    |
| Features used | **21 / 43**   |
| Algorithm     | Random Forest |
| Dataset       | 2919 samples  |

---

## 📊 Fatigue Levels

| Label | Level     | Meaning          |
| ----- | --------- | ---------------- |
| 1     | Low       | Normal training  |
| 2     | Moderate  | Monitor fatigue  |
| 3     | High      | Reduce intensity |
| 4     | Very High | Stop immediately |

---

## 📁 Project Structure

```
fatigue_model/
├── train_and_save.py
├── fatigue_predictor.py
├── smoke_test.py
├── requirements.txt
├── data/
│   └── database.xlsx
└── model/
    ├── fatigue_rf_model.joblib
    ├── feature_list.json
    ├── feature_means.json
    ├── label_map.json
    └── model_metadata.json
```

---

## 🔗 System Context

* ESP32 computes features (EMG + IMU)
* Data sent via MQTT
* App calls `FatiguePredictor`
* Output used for real-time coaching

---

## 🧠 Key Insight

* EMG → **muscle fatigue (primary signal)**
* IMU → **movement degradation (support signal)**

Combining both improves prediction reliability.

---

## 📌 Notes

* Model versioning is included in `model_metadata.json`
* No internet required for inference
* Runs in real-time (<10 ms per prediction)

---

## 📖 Reference

See project documentation:
`Graduation_project_documentation_team25.pdf` §4.6.8
