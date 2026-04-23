import numpy as np

NOSE = 0
L_HIP, R_HIP = 23, 24
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16


def _get_lm(kps, idx):
    return kps[idx*4:idx*4+3]


def _angle_between(a, b, c):
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-8
    cos = np.clip(np.dot(ba, bc) / denom, -1, 1)
    return np.degrees(np.arccos(cos))


def landmarks_to_keypoints(landmarks):
    kps = np.zeros(33*4, dtype=np.float32)
    for i, lm in enumerate(landmarks):
        kps[i*4]   = lm.x
        kps[i*4+1] = lm.y
        kps[i*4+2] = lm.z
        kps[i*4+3] = lm.visibility
    return kps


class RepCounterInterface:

    CALIB_REPS_NEEDED = 3
    CALIB_LOOSE_THRESHOLD = 80
    CALIB_TOLERANCE_SIGMA = 1.5

    def __init__(self, exercise="bicep_curl"):
        self.reps = 0
        self.phase = "DOWN"
        self.last_elbow = None

        self.exercise = exercise
        self.confidence = 1.0

        # ---- bicep calibration ----
        self.calibrated = False
        self.calib_swings = []
        self.current_rep_max_swing = 0
        self.swing_threshold = self.CALIB_LOOSE_THRESHOLD
        self.swing_ok = True

        # ---- tricep ROM tracking ----
        self.min_elbow = 999
        self.max_elbow = 0

    def update(self, landmarks):
        if landmarks is None:
            return self._state()

        kps = landmarks_to_keypoints(landmarks)
        self._update_phase(kps)
        return self._state()

    def _update_phase(self, kps):
        prev = self.phase

        r_elbow = _angle_between(_get_lm(kps, R_SHOULDER), _get_lm(kps, R_ELBOW), _get_lm(kps, R_WRIST))
        l_elbow = _angle_between(_get_lm(kps, L_SHOULDER), _get_lm(kps, L_ELBOW), _get_lm(kps, L_WRIST))

        # ============================
        # BICEP CURL 
        # ============================
        if self.exercise == "bicep_curl":

            r_swing = _angle_between(
                _get_lm(kps, R_ELBOW),
                _get_lm(kps, R_SHOULDER),
                _get_lm(kps, R_HIP)
            )
            l_swing = _angle_between(
                _get_lm(kps, L_ELBOW),
                _get_lm(kps, L_SHOULDER),
                _get_lm(kps, L_HIP)
            )

            #  visibility weights (DO NOT drop arms)
            r_vis = min(
                kps[R_SHOULDER*4+3],
                kps[R_ELBOW*4+3],
                kps[R_WRIST*4+3]
            )

            l_vis = min(
                kps[L_SHOULDER*4+3],
                kps[L_ELBOW*4+3],
                kps[L_WRIST*4+3]
            )

            # avoid zero influence
            r_w = max(r_vis, 0.1)
            l_w = max(l_vis, 0.1)

            #  weighted fusion
            elbow = (r_elbow * r_w + l_elbow * l_w) / (r_w + l_w)
            swing = (r_swing * r_w + l_swing * l_w) / (r_w + l_w)

            # smoothing
            if self.last_elbow is None:
                self.last_elbow = elbow
            elbow = 0.7 * self.last_elbow + 0.3 * elbow
            self.last_elbow = elbow

            # phase
            if elbow < 100:
                self.phase = "UP"
            elif elbow > 130:
                self.phase = "DOWN"

            # swing tracking
            if self.phase == "DOWN":
                self.current_rep_max_swing = max(self.current_rep_max_swing, swing)
                if swing > self.swing_threshold:
                    self.swing_ok = False

            if prev == "UP" and self.phase == "DOWN":
                self.swing_ok = True
                self.current_rep_max_swing = 0

            if prev == "DOWN" and self.phase == "UP" and self.swing_ok:
                self.reps += 1

                if not self.calibrated:
                    self.calib_swings.append(self.current_rep_max_swing)

                    if len(self.calib_swings) >= self.CALIB_REPS_NEEDED:
                        mean = np.mean(self.calib_swings)
                        std = np.std(self.calib_swings)
                        self.swing_threshold = mean + self.CALIB_TOLERANCE_SIGMA * std
                        self.calibrated = True

                self.current_rep_max_swing = 0

        # ============================
        # TRICEP EXTENSION 
        # ============================
        elif self.exercise == "tricep_extension":

            r_sh_y = _get_lm(kps, R_SHOULDER)[1]
            l_sh_y = _get_lm(kps, L_SHOULDER)[1]

            r_wr_y = _get_lm(kps, R_WRIST)[1]
            l_wr_y = _get_lm(kps, L_WRIST)[1]

            if r_elbow > l_elbow:
                elbow = r_elbow
                wrist = _get_lm(kps, R_WRIST)
                wrist_y = r_wr_y
                shoulder_y = r_sh_y
            else:
                elbow = l_elbow
                wrist = _get_lm(kps, L_WRIST)
                wrist_y = l_wr_y
                shoulder_y = l_sh_y

            arm_up = wrist_y < shoulder_y

            nose = _get_lm(kps, NOSE)
            dist_to_head = np.linalg.norm(wrist - nose)

            shoulder_mid = (_get_lm(kps, L_SHOULDER) + _get_lm(kps, R_SHOULDER)) / 2
            hip_mid = (_get_lm(kps, L_HIP) + _get_lm(kps, R_HIP)) / 2
            torso = np.linalg.norm(shoulder_mid - hip_mid) + 1e-8

            near_head = (dist_to_head / torso) < 1.5
            good_position = arm_up and near_head

            if self.last_elbow is None:
                self.last_elbow = elbow
            elbow = 0.5 * self.last_elbow + 0.5 * elbow
            self.last_elbow = elbow

            self.min_elbow = min(self.min_elbow, elbow)
            self.max_elbow = max(self.max_elbow, elbow)
            rom = self.max_elbow - self.min_elbow

            if elbow > 140:
                self.phase = "UP"
            elif elbow < 115:
                self.phase = "DOWN"

            if prev == "DOWN" and self.phase == "UP" and good_position and rom > 35:
                self.reps += 1
                self.min_elbow = elbow
                self.max_elbow = elbow

    def _state(self):
        return {
            "exercise": self.exercise,
            "reps": self.reps,
            "confidence": self.confidence,
            "phase": self.phase,
        }
