"""
WorkoutHacker – Fatigue Inference Module
==========================================

Drop-in inference wrapper used by the mobile / backend team.

After training (run train_and_save.py once), this module is the only
file needed at runtime — no pandas, no sklearn training code required
beyond loading the saved model.

Key Design Notes:
----------------
- Missing features are handled using training-set mean values
  (not zeros) to reduce prediction bias and improve robustness.
- Feature scaling is NOT applied intentionally:
  Random Forest models do not require scaling, and training/inference
  are kept consistent.
- Feature extraction (RMS, IMU stats) is performed on the ESP32 firmware.
  A Python equivalent can be added for offline testing if needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Union

import joblib
import numpy as np

_DEFAULT_MODEL_DIR = Path(__file__).parent / "model"


class FatiguePredictor:
    """
    Wraps the trained Random Forest fatigue classifier.

    Parameters
    ----------
    model_dir : str or Path, optional
        Directory containing artefact files produced by train_and_save.py.
        Defaults to ./model/
    """

    def __init__(self, model_dir: Union[str, Path, None] = None):
        model_dir = Path(model_dir) if model_dir else _DEFAULT_MODEL_DIR

        # Load trained model
        self._model = joblib.load(model_dir / "fatigue_rf_model.joblib")

        # Load ordered feature list
        with open(model_dir / "feature_list.json") as f:
            self._features: List[str] = json.load(f)

        # Load feature means for safe imputation
        with open(model_dir / "feature_means.json") as f:
            self._feature_means: Dict[str, float] = json.load(f)

        # Load label mapping
        with open(model_dir / "label_map.json") as f:
            raw = json.load(f)
            self._label_map: Dict[int, str] = {int(k): v for k, v in raw.items()}

        # Load metadata (includes version info)
        with open(model_dir / "model_metadata.json") as f:
            self.metadata: dict = json.load(f)

        self._n_features = len(self._features)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @property
    def feature_names(self) -> List[str]:
        """Ordered list of features expected by the model."""
        return list(self._features)

    def predict(self, sample: Union[Dict[str, float], np.ndarray, list]) -> dict:
        """
        Predict fatigue level for a single sample.

        Parameters
        ----------
        sample : dict | numpy array | list

            dict:
                Keys are feature names. Missing features are automatically
                filled using training-set mean values.

            array/list:
                Must match the exact feature order and length.

        Returns
        -------
        dict:
            {
                "fatigue_level": str,
                "label_int": int,
                "probabilities": dict,
                "confidence": float
            }
        """
        x = self._to_array(sample).reshape(1, -1)
        return self._format_single(x)

    def predict_batch(
        self,
        samples: Union[List[Dict[str, float]], np.ndarray],
    ) -> List[dict]:
        """
        Predict fatigue levels for multiple samples.

        Parameters
        ----------
        samples : list of dicts OR numpy array (N, n_features)

        Returns
        -------
        List of prediction dictionaries.
        """
        if isinstance(samples, np.ndarray):
            X = samples
        else:
            X = np.vstack([self._to_array(s) for s in samples])

        label_ints = self._model.predict(X)
        proba_matrix = self._model.predict_proba(X)
        classes = self._model.classes_

        return [
            self._build_result(int(label_ints[i]), proba_matrix[i], classes)
            for i in range(len(label_ints))
        ]

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _to_array(self, sample) -> np.ndarray:
        """
        Converts input sample into a feature vector.

        Missing features are filled using mean values computed during training.

        This ensures:
        - Robustness when some sensors are unavailable
        - Reduced bias compared to zero-filling
        """
        if isinstance(sample, (np.ndarray, list)):
            arr = np.asarray(sample, dtype=float)
            if arr.shape[0] != self._n_features:
                raise ValueError(
                    f"Expected {self._n_features} features, got {arr.shape[0]}"
                )
            return arr

        # dict input → ordered vector
        return np.array(
            [
                float(sample.get(f, self._feature_means[f]))
                for f in self._features
            ],
            dtype=float,
        )

    def _format_single(self, x: np.ndarray) -> dict:
        label_int = int(self._model.predict(x)[0])
        proba = self._model.predict_proba(x)[0]
        classes = self._model.classes_

        return self._build_result(label_int, proba, classes)

    def _build_result(self, label_int: int, proba: np.ndarray, classes) -> dict:
        proba_dict = {
            self._label_map[int(c)]: round(float(p), 4)
            for c, p in zip(classes, proba)
        }

        confidence = max(proba_dict.values())

        return {
            "fatigue_level": self._label_map[label_int],
            "label_int": label_int,
            "probabilities": proba_dict,
            "confidence": confidence,
        }