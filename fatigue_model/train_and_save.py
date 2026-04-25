"""
WorkoutHacker – Fatigue Detection Model
========================================
Trains the Random Forest fatigue classifier on the IMU–Gait–EMG dataset
and saves a self-contained artefact (model + feature list + label map + feature means)
that the mobile / backend team can load directly for inference.

Dataset:
  https://figshare.com/articles/dataset/Dataset/15104079?file=29037888
  Place the downloaded file as  data/database.xlsx  next to this script.

Output (saved to  model/):
  fatigue_rf_model.joblib   – trained RandomForestClassifier
  feature_list.json         – ordered list of the selected features
  feature_means.json        – mean value for each feature (for safe imputation)
  label_map.json            – int → fatigue-level string mapping
  model_metadata.json       – accuracy / F1 / G-mean + version info

Usage:
  pip install -r requirements.txt
  python train_and_save.py
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import gmean as geometric_mean
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, recall_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


# Paths

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "database.xlsx"
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)

# Label map

LABEL_MAP = {1: "low", 2: "moderate", 3: "high", 4: "very_high"}


# Helpers

def gmean_score(y_true, y_pred):
    """Geometric mean of per-class recall (balanced multi-class metric)."""
    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
    return float(geometric_mean(recalls))



# 1. Load data

print("Loading dataset …")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at {DATA_PATH}.\n"
        "Download it from https://figshare.com/articles/dataset/Dataset/15104079 "
        "and place it at data/database.xlsx"
    )

df = pd.read_excel(DATA_PATH)

# Drop index column if exists
if "Unnamed: 0" in df.columns:
    df.drop(columns="Unnamed: 0", inplace=True)

# Explicit feature list (prevents dependency on column order)
# This still dynamically loads all features except label
FEATURE_COLUMNS = [col for col in df.columns if col != "labels"]

X_all = df[FEATURE_COLUMNS].values
y_all = df["labels"].values

print(f"  Samples: {len(y_all)}, Raw features: {len(FEATURE_COLUMNS)}")

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42
)

# Keep DataFrame copies for feature selection
dtf = df.copy()
dtf_train, dtf_test = train_test_split(dtf, test_size=0.2, random_state=42)


# 2. Feature importance ranking

print("\nRanking features …")

rf_ranker = RandomForestClassifier(n_estimators=80, random_state=42)
rf_ranker.fit(X_train, y_train)

importances = rf_ranker.feature_importances_
sorted_indices = np.argsort(importances)[::-1]
sorted_features = np.array(FEATURE_COLUMNS)[sorted_indices].tolist()


# 3. Sequential feature selection

print("Running sequential feature selection …")

n_estimators_grid = [5, 10, 20, 40, 60, 80, 100]
best_overall_score = 0.0
best_n_features = 0
optimization_results = []

for N in range(1, len(sorted_features) + 1):
    current_feats = sorted_features[:N]

    X_tr = dtf_train[current_feats].values
    X_te = dtf_test[current_feats].values

    best_score_n, best_est_n = 0.0, 80

    for n_est in n_estimators_grid:
        clf = RandomForestClassifier(n_estimators=n_est, random_state=42)
        clf.fit(X_tr, y_train)
        sc = clf.score(X_te, y_test)

        if sc > best_score_n:
            best_score_n, best_est_n = sc, n_est

    optimization_results.append(
        {"N": N, "acc": best_score_n, "n_est": best_est_n}
    )

    if best_score_n > best_overall_score:
        best_overall_score = best_score_n
        best_n_features = N

print(f"  Optimal feature count: {best_n_features} (accuracy {best_overall_score:.4f})")


# 4. Train final model

optimal_features = sorted_features[:best_n_features]
opt_result = optimization_results[best_n_features - 1]
optimal_n_estimators = opt_result["n_est"]

X_train_opt = dtf_train[optimal_features].values
X_test_opt = dtf_test[optimal_features].values

print(f"\nTraining final RF (n_estimators={optimal_n_estimators}, features={best_n_features}) …")

final_model = RandomForestClassifier(
    n_estimators=optimal_n_estimators,
    random_state=42
)

final_model.fit(X_train_opt, y_train)

y_pred = final_model.predict(X_test_opt)

oa = final_model.score(X_test_opt, y_test)
macro = precision_recall_fscore_support(
    y_test, y_pred, average="macro", zero_division=0
)
gm = gmean_score(y_test, y_pred)

print(f"  Overall Accuracy : {oa:.4f}")
print(f"  F1 (macro)       : {macro[2]:.4f}")
print(f"  G-Mean           : {gm:.4f}")


# 5. Compute feature means (for safe imputation)

feature_means = dict(zip(optimal_features, X_train_opt.mean(axis=0)))


# 6. Save artefacts

print("\nSaving artefacts …")

joblib.dump(final_model, MODEL_DIR / "fatigue_rf_model.joblib")

with open(MODEL_DIR / "feature_list.json", "w") as f:
    json.dump(optimal_features, f, indent=2)

with open(MODEL_DIR / "feature_means.json", "w") as f:
    json.dump(feature_means, f, indent=2)

with open(MODEL_DIR / "label_map.json", "w") as f:
    json.dump({str(k): v for k, v in LABEL_MAP.items()}, f, indent=2)

# Model metadata (with versioning)
metadata = {
    "version": "v1.0",
    "model": "RandomForestClassifier",
    "n_estimators": int(optimal_n_estimators),
    "n_features": best_n_features,
    "overall_accuracy": round(oa, 4),
    "f1_macro": round(float(macro[2]), 4),
    "g_mean": round(gm, 4),
    "train_samples": int(len(y_train)),
    "test_samples": int(len(y_test)),
    "label_map": LABEL_MAP,
    "random_state": 42,
    "dataset": "IMU–Gait–EMG Fatigue Dataset (figshare 15104079)",
}

with open(MODEL_DIR / "model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\nAll artefacts saved to ./model/ ✓")