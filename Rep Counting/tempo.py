import time
import json
import numpy as np
import joblib


class TempoAnalyzer:
    """
    ML-based tempo classifier.
    Usage:
        analyzer = TempoAnalyzer()
        analyzer.set_exercise('bicep_curl')
        analyzer.update(state['phase'], fps)
        feedback = analyzer.get_state()  # {'tempo': 'normal', 'quality': 85}
    """

    EXERCISE_FALLBACK = 0

    def __init__(self,
                 model_path='tempo_classifier.pkl',
                 config_path='tempo_config.json'):

        self.model = joblib.load(model_path)

        with open(config_path) as f:
            cfg = json.load(f)

        self.exercise_classes = cfg['exercise_classes']
        self.feature_cols     = cfg['feature_cols']
        self.fps              = cfg['fps']

        # Internal state
        self._phase       = 'UNKNOWN'
        self._phase_start = None
        self._rep_start   = None
        self._mid_time    = None
        self._exercise    = None

        # Output
        self.tempo   = 'unknown'
        self.quality = 0

    def set_exercise(self, exercise_name: str):
        """Call this every frame with the current exercise."""
        self._exercise = exercise_name

    def update(self, phase: str, fps: float):
        """Called every frame with the current phase (UP / DOWN / UNKNOWN)."""
        now      = time.time()
        self.fps = fps
        prev     = self._phase

        if phase != prev:
            if prev == 'UNKNOWN' and phase in ('UP', 'DOWN'):
                # First phase detected — rep started
                self._rep_start   = now
                self._phase_start = now

            elif prev in ('UP', 'DOWN') and phase in ('UP', 'DOWN'):
                # Phase flipped — this is the midpoint of the rep
                self._mid_time    = now
                self._phase_start = now

                # We have a complete rep
                if self._rep_start is not None and self._mid_time is not None:
                    total_s = now - self._rep_start
                    if 0.3 < total_s < 15:
                        self._classify_rep(total_s, now)
                    self._rep_start = now  # reset for next rep

        self._phase = phase

    def _classify_rep(self, total_s: float, now: float):
        """Run the ML model on the completed rep."""
        half1_s    = self._mid_time - (now - total_s)
        half2_s    = total_s - half1_s
        half1_s    = max(half1_s, 0.05)
        half2_s    = max(half2_s, 0.05)
        half_ratio = half1_s / half2_s

        if self._exercise and self._exercise in self.exercise_classes:
            ex_enc = self.exercise_classes.index(self._exercise)
        else:
            ex_enc = self.EXERCISE_FALLBACK

        features = np.array([[total_s, half1_s, half2_s, half_ratio, ex_enc]])
        proba    = self.model.predict_proba(features)[0]
        classes  = self.model.classes_

        pred_idx     = np.argmax(proba)
        self.tempo   = classes[pred_idx]
        self.quality = int(round(proba[pred_idx] * 100))

    def get_state(self) -> dict:
        return {'tempo': self.tempo, 'quality': self.quality}