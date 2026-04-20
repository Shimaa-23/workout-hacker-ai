"""
rep_counter_interface.py

Production-ready real-time exercise classification + rep counting module.

Uses the exact same feature engineering and model structure used during training,
so it remains fully compatible with Combined_model.pth.

Input:
    MediaPipe Pose landmarks (33 landmarks per frame)

Output:
    {
        "exercise": str | None,
        "reps": int,
        "confidence": float,
        "phase": "UP" | "DOWN" | "UNKNOWN"
    }
"""

import numpy as np
import torch
import torch.nn as nn
from collections import deque


# ==========================================================
# MediaPipe Landmark Indices
# ==========================================================
NOSE = 0

L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16

L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28


# ==========================================================
# BiLSTM + Attention Model
# ==========================================================
class RepCountLSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size=128,
        num_layers=2,
        num_classes=8,
        dropout=0.3
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        d = hidden_size * 2

        self.attention = nn.Linear(d, 1)

        self.classifier = nn.Sequential(
            nn.Linear(d, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, real_mask):
        lstm_out, _ = self.lstm(x)

        attn = self.attention(lstm_out).squeeze(-1)

        empty = ~real_mask.any(dim=1, keepdim=True)
        safe_mask = real_mask | empty.expand_as(real_mask)

        attn = attn.masked_fill(~safe_mask, float("-inf"))

        weights = torch.nan_to_num(
            torch.softmax(attn, dim=1),
            nan=1.0 / real_mask.shape[1]
        )

        context = (weights.unsqueeze(-1) * lstm_out).sum(dim=1)

        return self.classifier(context)


# ==========================================================
# Geometry Helpers
# ==========================================================
def _get_lm(kps, idx):
    """Return x,y,z for a landmark index from flat keypoints."""
    return kps[idx * 4: idx * 4 + 3]


def _angle_between(a, b, c):
    """Angle at joint b using points a-b-c."""
    ba = a - b
    bc = c - b

    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-8
    cos_val = np.dot(ba, bc) / denom
    cos_val = np.clip(cos_val, -1, 1)

    return np.degrees(np.arccos(cos_val))


def _torso_size(kps):
    mid_sh = (_get_lm(kps, L_SHOULDER) + _get_lm(kps, R_SHOULDER)) / 2
    mid_hp = (_get_lm(kps, L_HIP) + _get_lm(kps, R_HIP)) / 2

    return np.linalg.norm(mid_sh - mid_hp) + 1e-8


# ==========================================================
# Exact Training-Time Feature Engineering (46 Features)
# ==========================================================
def engineer_features(kps):
    """
    Convert 132 raw keypoint values into 46 engineered features.
    Exact logic used during training.
    """
    if np.all(kps == 0):
        return np.zeros(46, dtype=np.float32)

    ts = _torso_size(kps)
    feats = []

    # ------------------------------------------------------
    # 12 Joint Angles
    # ------------------------------------------------------
    angle_triplets = [
        (L_SHOULDER, L_ELBOW, L_WRIST),
        (R_SHOULDER, R_ELBOW, R_WRIST),

        (L_HIP, L_SHOULDER, L_ELBOW),
        (R_HIP, R_SHOULDER, R_ELBOW),

        (L_SHOULDER, L_HIP, L_KNEE),
        (R_SHOULDER, R_HIP, R_KNEE),

        (L_SHOULDER, R_SHOULDER, R_HIP),

        (L_HIP, L_KNEE, L_ANKLE),
        (R_HIP, R_KNEE, R_ANKLE),

        (L_KNEE, L_ANKLE, R_ANKLE),
        (R_KNEE, R_ANKLE, L_ANKLE),

        (L_HIP, R_HIP, R_KNEE),
    ]

    for a, b, c in angle_triplets:
        ang = _angle_between(
            _get_lm(kps, a),
            _get_lm(kps, b),
            _get_lm(kps, c)
        ) / 180.0
        feats.append(ang)

    # ------------------------------------------------------
    # 12 Relative Distances
    # ------------------------------------------------------
    dist_pairs = [
        (L_WRIST, R_WRIST),
        (L_WRIST, L_SHOULDER),
        (R_WRIST, R_SHOULDER),

        (L_WRIST, L_HIP),
        (R_WRIST, R_HIP),

        (L_ANKLE, R_ANKLE),
        (L_KNEE, R_KNEE),

        (L_KNEE, L_HIP),
        (R_KNEE, R_HIP),

        (NOSE, L_HIP),
        (L_ELBOW, R_ELBOW),
        (L_WRIST, NOSE),
    ]

    for a, b in dist_pairs:
        d = np.linalg.norm(_get_lm(kps, a) - _get_lm(kps, b)) / ts
        feats.append(d)

    # ------------------------------------------------------
    # 11 Vertical Positions
    # ------------------------------------------------------
    mid_hip_y = (_get_lm(kps, L_HIP)[1] + _get_lm(kps, R_HIP)[1]) / 2

    key_joints = [
        NOSE,
        L_SHOULDER, R_SHOULDER,
        L_ELBOW, R_ELBOW,
        L_WRIST, R_WRIST,
        L_KNEE, R_KNEE,
        L_ANKLE, R_ANKLE
    ]

    for j in key_joints:
        val = (_get_lm(kps, j)[1] - mid_hip_y) / ts
        feats.append(val)

    # ------------------------------------------------------
    # 11 Visibility Scores
    # ------------------------------------------------------
    for j in key_joints:
        feats.append(kps[j * 4 + 3])

    return np.array(feats, dtype=np.float32)


# ==========================================================
# MediaPipe Landmarks -> Flat Keypoints
# ==========================================================
def landmarks_to_keypoints(landmarks):
    kps = np.zeros(33 * 4, dtype=np.float32)

    for i, lm in enumerate(landmarks):
        kps[i * 4] = lm.x
        kps[i * 4 + 1] = lm.y
        kps[i * 4 + 2] = lm.z
        kps[i * 4 + 3] = lm.visibility

    return kps


# ==========================================================
# Rep Counting Signal
# ==========================================================
def _get_elbow_angle(kps):
    return _angle_between(
        _get_lm(kps, R_SHOULDER),
        _get_lm(kps, R_ELBOW),
        _get_lm(kps, R_WRIST)
    )


# ==========================================================
# Main Interface
# ==========================================================
class RepCounterInterface:
    DEFAULT_CLASSES = [
        "front_raise",
        "push_up",
        "pull_up",
        "bench_pressing",
        "bicep_curl",
        "tricep_extension",
        "lateral_raise",
        "shoulder_press"
    ]

    def __init__(
        self,
        model_path="Combined_model.pth",
        inference_every=5,
        min_confidence=0.6,
        up_threshold=150.0,
        down_threshold=90.0
    ):
        self.device = torch.device("cpu")

        self.inference_every = inference_every
        self.min_confidence = min_confidence
        self.up_threshold = up_threshold
        self.down_threshold = down_threshold

        ckpt = torch.load(model_path, map_location=self.device)
        cfg = ckpt["config"]

        self.classes = ckpt.get("classes", self.DEFAULT_CLASSES)
        self.max_seq_len = ckpt.get("max_seq_len", 30)

        self.norm_mean = ckpt.get("norm_mean", None)
        self.norm_std = ckpt.get("norm_std", None)

        self.model = RepCountLSTM(
            input_size=cfg["input_size"],
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            num_classes=cfg["num_classes"],
            dropout=cfg.get("dropout", 0.3)
        ).to(self.device)

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.buffer = deque(maxlen=self.max_seq_len)
        self.frame_count = 0

        self.reps = 0
        self.phase = "UNKNOWN"

        self.exercise = None
        self.confidence = 0.0

    # ------------------------------------------------------
    # Public API
    # ------------------------------------------------------
    def update(self, landmarks):
        """
        Call once per frame.
        """
        if landmarks is None:
            return self._state()

        kps = landmarks_to_keypoints(landmarks)

        feat = engineer_features(kps)

        if self.norm_mean is not None:
            if not np.all(feat == 0):
                feat = np.clip(
                    (feat - self.norm_mean) / self.norm_std,
                    -5,
                    5
                )

        self.buffer.append(feat)
        self.frame_count += 1

        self._update_phase(kps)

        if (
            self.frame_count % self.inference_every == 0
            and len(self.buffer) > 0
        ):
            self._run_inference()

        return self._state()

    def reset_reps(self):
        self.reps = 0
        self.phase = "UNKNOWN"

    def reset_all(self):
        self.reset_reps()
        self.buffer.clear()
        self.frame_count = 0
        self.exercise = None
        self.confidence = 0.0

    # ------------------------------------------------------
    # Internal
    # ------------------------------------------------------
    def _run_inference(self):
        frames = list(self.buffer)
        n = len(frames)

        seq = np.zeros((self.max_seq_len, 46), dtype=np.float32)
        seq[:n] = frames

        mask = np.array(
            [True] * n + [False] * (self.max_seq_len - n)
        )

        x = torch.tensor(seq).unsqueeze(0)
        m = torch.tensor(mask).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(x, m)
            probs = torch.softmax(logits, dim=1)[0].numpy()

        idx = int(np.argmax(probs))
        conf = float(probs[idx])

        if conf >= self.min_confidence:
            self.exercise = self.classes[idx]
            self.confidence = conf

    def _update_phase(self, kps):
        angle = _get_elbow_angle(kps)

        prev = self.phase

        if angle > self.up_threshold:
            self.phase = "UP"
        elif angle < self.down_threshold:
            self.phase = "DOWN"

        if prev == "DOWN" and self.phase == "UP":
            self.reps += 1

    def _state(self):
        return {
            "exercise": self.exercise,
            "reps": self.reps,
            "confidence": round(self.confidence, 3),
            "phase": self.phase
        }