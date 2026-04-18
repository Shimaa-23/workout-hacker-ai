import copy
import json
import os
import subprocess

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATHS = {
    "lite": os.path.join(SCRIPT_DIR, "models", "pose_landmarker_lite.task"),
    "full": os.path.join(SCRIPT_DIR, "models", "pose_landmarker_full.task"),
    "heavy": os.path.join(SCRIPT_DIR, "models", "pose_landmarker.task"),
}

POSE_CONNECTIONS = frozenset(
    [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 7),
        (0, 4),
        (4, 5),
        (5, 6),
        (6, 8),
        (9, 10),
        (11, 12),
        (11, 13),
        (13, 15),
        (15, 17),
        (15, 19),
        (15, 21),
        (17, 19),
        (12, 14),
        (14, 16),
        (16, 18),
        (16, 20),
        (16, 22),
        (18, 20),
        (11, 23),
        (12, 24),
        (23, 24),
        (23, 25),
        (25, 27),
        (27, 29),
        (27, 31),
        (29, 31),
        (24, 26),
        (26, 28),
        (28, 30),
        (28, 32),
        (30, 32),
    ]
)


def _resolve_model_path(model_complexity: str) -> str:
    model_path = MODEL_PATHS.get(model_complexity, MODEL_PATHS["heavy"])
    if os.path.exists(model_path):
        return model_path

    fallback = MODEL_PATHS["heavy"]
    if os.path.exists(fallback):
        return fallback

    raise FileNotFoundError(
        f"No Pose Landmarker model found. Expected one of: {list(MODEL_PATHS.values())}"
    )


def _rolling_smooth_xyz(arr: np.ndarray, window_size: int) -> np.ndarray:
    if arr.shape[0] < window_size:
        return arr
    for joint_idx in range(33):
        for axis in range(3):
            arr[:, joint_idx, axis] = (
                pd.Series(arr[:, joint_idx, axis])
                .rolling(window=window_size, center=True, min_periods=1)
                .mean()
                .to_numpy()
            )
    return arr


class LivePoseTracker:
    def __init__(self, use_gpu: bool = False, model_complexity: str = "full"):
        model_path = _resolve_model_path(model_complexity)

        base_options = mp.tasks.BaseOptions
        pose_landmarker = mp.tasks.vision.PoseLandmarker
        pose_options = mp.tasks.vision.PoseLandmarkerOptions
        running_mode = mp.tasks.vision.RunningMode

        delegate = (
            base_options.Delegate.GPU if use_gpu else base_options.Delegate.CPU
        )

        options = pose_options(
            base_options=base_options(model_asset_path=model_path, delegate=delegate),
            running_mode=running_mode.VIDEO,
        )
        self.landmarker = pose_landmarker.create_from_options(options)

    def process_frame(self, frame, timestamp_ms):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        frame_landmarks = []
        if detection_result.pose_landmarks:
            for landmark in detection_result.pose_landmarks[0]:
                frame_landmarks.append(
                    {
                        "x": landmark.x,
                        "y": landmark.y,
                        "z": landmark.z,
                        "visibility": landmark.visibility,
                    }
                )
        return frame_landmarks

    def close(self):
        self.landmarker.close()


def process_video(
    video_path,
    start_seconds=0.0,
    end_seconds=None,
    progress_callback=None,
    use_gpu=False,
    model_complexity="heavy",
):
    model_path = _resolve_model_path(model_complexity)

    base_options = mp.tasks.BaseOptions
    pose_landmarker = mp.tasks.vision.PoseLandmarker
    pose_options = mp.tasks.vision.PoseLandmarkerOptions
    running_mode = mp.tasks.vision.RunningMode

    if use_gpu:
        delegate = base_options.Delegate.GPU
        mode = running_mode.IMAGE
    else:
        delegate = base_options.Delegate.CPU
        mode = running_mode.VIDEO

    options = pose_options(
        base_options=base_options(model_asset_path=model_path, delegate=delegate),
        running_mode=mode,
    )

    pose_data = []
    try:
        with pose_landmarker.create_from_options(options) as landmarker:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if fps <= 0:
                fps = 30.0

            start_frame = max(0, int(start_seconds * fps))
            end_frame = (
                min(total_frames, int(end_seconds * fps))
                if end_seconds is not None
                else total_frames
            )

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frame_count = start_frame
            total_process_frames = max(end_frame - start_frame, 0)
            processed_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame_count >= end_frame:
                    break

                timestamp_ms = int((frame_count / fps) * 1000)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                if use_gpu:
                    detection_result = landmarker.detect(mp_image)
                else:
                    detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

                frame_landmarks = []
                if detection_result.pose_landmarks:
                    for landmark in detection_result.pose_landmarks[0]:
                        frame_landmarks.append(
                            {
                                "x": landmark.x,
                                "y": landmark.y,
                                "z": landmark.z,
                                "visibility": landmark.visibility,
                            }
                        )

                pose_data.append(
                    {
                        "frame": frame_count,
                        "timestamp_ms": timestamp_ms,
                        "landmarks": frame_landmarks,
                    }
                )
                frame_count += 1
                processed_count += 1

                if progress_callback and total_process_frames > 0:
                    progress_callback(processed_count / total_process_frames)

            cap.release()
    except Exception as exc:
        if use_gpu and "ValidatedGraphConfig Initialization failed" in str(exc):
            raise RuntimeError(
                "GPU initialization failed. This MediaPipe build may not support GPU on this system. "
                f"Error: {exc}"
            )
        raise

    return pose_data


def render_skeleton_on_frame(img, landmarks, width, height):
    color_point = (0, 255, 0)
    color_line = (0, 255, 255)

    points = {}
    for idx, lm in enumerate(landmarks):
        if lm.get("visibility", 1.0) < 0.5:
            continue
        px, py = int(lm["x"] * width), int(lm["y"] * height)
        points[idx] = (px, py)
        cv2.circle(img, (px, py), 4, color_point, -1)

    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx in points and end_idx in points:
            cv2.line(img, points[start_idx], points[end_idx], color_line, 2)
    return img


def render_skeleton_video(
    skeleton_data, output_video_path, width=1280, height=720, fps=30
):
    skeleton_data.sort(key=lambda x: x["frame"])
    temp_output_path = output_video_path.replace(".mp4", "_temp.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

    for frame_data in skeleton_data:
        img = np.zeros((height, width, 3), dtype=np.uint8)
        landmarks = frame_data.get("landmarks", [])
        img = render_skeleton_on_frame(img, landmarks, width, height)

        timestamp = frame_data.get("timestamp_ms", 0)
        cv2.putText(
            img,
            f"Time: {timestamp / 1000:.2f}s",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        out.write(img)

    out.release()

    try:
        command = [
            "ffmpeg",
            "-y",
            "-i",
            temp_output_path,
            "-vcodec",
            "libx264",
            "-f",
            "mp4",
            output_video_path,
        ]
        subprocess.run(
            command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except Exception:
        if os.path.exists(temp_output_path):
            if os.path.exists(output_video_path):
                os.remove(output_video_path)
            os.rename(temp_output_path, output_video_path)
    finally:
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)

    return output_video_path


def normalize_skeleton_for_classifier(skeleton_arr):
    normalized = skeleton_arr.copy()
    for i in range(len(normalized)):
        left_hip = normalized[i, 23, :3]
        right_hip = normalized[i, 24, :3]
        hip_center = (left_hip + right_hip) / 2.0
        normalized[i, :, :3] -= hip_center

        left_sh = normalized[i, 11, :3]
        right_sh = normalized[i, 12, :3]
        shoulder_dist = np.linalg.norm(left_sh - right_sh)
        if shoulder_dist > 0:
            normalized[i, :, :3] /= shoulder_dist
    return normalized


def calculate_angle(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))


def extract_features_from_skeleton(normalized_arr, window_size=15):
    num_frames = len(normalized_arr)
    l_shoulder, r_shoulder = 11, 12
    l_elbow, r_elbow = 13, 14
    l_wrist, r_wrist = 15, 16
    l_hip, r_hip = 23, 24

    features_list = []
    for i in range(num_frames):
        frame_lm = normalized_arr[i]
        l_elbow_angle = calculate_angle(
            frame_lm[l_shoulder, :3], frame_lm[l_elbow, :3], frame_lm[l_wrist, :3]
        )
        r_elbow_angle = calculate_angle(
            frame_lm[r_shoulder, :3], frame_lm[r_elbow, :3], frame_lm[r_wrist, :3]
        )
        l_shoulder_angle = calculate_angle(
            frame_lm[l_hip, :3], frame_lm[l_shoulder, :3], frame_lm[l_elbow, :3]
        )
        r_shoulder_angle = calculate_angle(
            frame_lm[r_hip, :3], frame_lm[r_shoulder, :3], frame_lm[r_elbow, :3]
        )
        l_wrist_y_rel = frame_lm[l_wrist, 1] - frame_lm[l_shoulder, 1]
        r_wrist_y_rel = frame_lm[r_wrist, 1] - frame_lm[r_shoulder, 1]
        features_list.append(
            [
                l_elbow_angle,
                r_elbow_angle,
                l_shoulder_angle,
                r_shoulder_angle,
                l_wrist_y_rel,
                r_wrist_y_rel,
            ]
        )

    df_feat = pd.DataFrame(
        features_list,
        columns=[
            "l_elbow_angle",
            "r_elbow_angle",
            "l_shoulder_angle",
            "r_shoulder_angle",
            "l_wrist_y_rel",
            "r_wrist_y_rel",
        ],
    )

    arm_landmarks = {"l_sh": 11, "r_sh": 12, "l_el": 13, "r_el": 14, "l_wr": 15, "r_wr": 16}

    for name, idx in arm_landmarks.items():
        coords = normalized_arr[:, idx, :3]
        df_coords = pd.DataFrame(coords, columns=[f"{name}_x", f"{name}_y", f"{name}_z"])

        df_vel = df_coords.diff().fillna(0)
        df_vel.columns = [f"{name}_vx", f"{name}_vy", f"{name}_vz"]

        df_acc = df_vel.diff().fillna(0)
        df_acc.columns = [f"{name}_ax", f"{name}_ay", f"{name}_az"]

        df_std = df_coords.rolling(window=window_size, min_periods=1).std().fillna(0)
        df_std.columns = [f"{name}_stdx", f"{name}_stdy", f"{name}_stdz"]

        df_feat = pd.concat([df_feat, df_vel, df_acc, df_std], axis=1)

    return df_feat


def classify_exercise(skeleton_data, model, smoothing_window=5):
    num_frames = len(skeleton_data)
    if num_frames == 0:
        return "Unknown", []

    arr = np.zeros((num_frames, 33, 4))
    for i, frame in enumerate(skeleton_data):
        landmarks = frame.get("landmarks", []) if isinstance(frame, dict) else frame
        for j, lm in enumerate(landmarks):
            if j < 33:
                arr[i, j, 0] = lm["x"]
                arr[i, j, 1] = lm["y"]
                arr[i, j, 2] = lm["z"]
                arr[i, j, 3] = lm.get("visibility", 1.0)

    if smoothing_window > 0:
        arr = _rolling_smooth_xyz(arr, smoothing_window)

    normalized_arr = normalize_skeleton_for_classifier(arr)
    X = extract_features_from_skeleton(normalized_arr)

    label_map = {
        0: "Bicep Curl",
        1: "Lateral Raise",
        2: "Null/Unknown",
        3: "Shoulder Press",
        4: "Triceps Extension",
        5: "Front Raises",
    }

    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X)
        avg_proba = np.mean(probas, axis=0)
        final_prediction_idx = int(np.argmax(avg_proba))
        predictions = np.argmax(probas, axis=1)
        actual_label = model.classes_[final_prediction_idx]
        return label_map.get(actual_label, "Unknown"), predictions

    predictions = model.predict(X)
    final_prediction = np.bincount(predictions).argmax()
    return label_map.get(final_prediction, "Unknown"), predictions


def calculate_midpoint(p1, p2):
    return {
        "x": (p1["x"] + p2["x"]) / 2,
        "y": (p1["y"] + p2["y"]) / 2,
        "z": (p1["z"] + p2["z"]) / 2,
        "visibility": min(p1["visibility"], p2["visibility"]),
    }


def get_distance(p1, p2):
    return np.sqrt((p1["x"] - p2["x"]) ** 2 + (p1["y"] - p2["y"]) ** 2)


def normalize_frame(landmarks, target_height=0.3):
    if not landmarks:
        return []

    left_shoulder, right_shoulder, left_hip, right_hip = 11, 12, 23, 24
    if len(landmarks) <= max(left_shoulder, right_shoulder, left_hip, right_hip):
        return landmarks

    shoulder_center = calculate_midpoint(
        landmarks[left_shoulder], landmarks[right_shoulder]
    )
    hip_center = calculate_midpoint(landmarks[left_hip], landmarks[right_hip])

    offset_x = 0.5 - hip_center["x"]
    offset_y = 0.5 - hip_center["y"]
    offset_z = -hip_center["z"]

    translated = [
        {
            "x": lm["x"] + offset_x,
            "y": lm["y"] + offset_y,
            "z": lm["z"] + offset_z,
            "visibility": lm["visibility"],
        }
        for lm in landmarks
    ]

    current_torso_height = get_distance(shoulder_center, hip_center)
    if current_torso_height == 0:
        return translated

    scale_factor = target_height / current_torso_height
    return [
        {
            "x": 0.5 + (lm["x"] - 0.5) * scale_factor,
            "y": 0.5 + (lm["y"] - 0.5) * scale_factor,
            "z": lm["z"] * scale_factor,
            "visibility": lm["visibility"],
        }
        for lm in translated
    ]


def normalize_skeleton_data(skeleton_data):
    normalized_data = []
    for frame in skeleton_data:
        new_frame = copy.deepcopy(frame)
        if "landmarks" in new_frame:
            new_frame["landmarks"] = normalize_frame(new_frame["landmarks"])
        normalized_data.append(new_frame)
    return normalized_data
