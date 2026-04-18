import json
import os
import tempfile
import time

import cv2
import joblib
import streamlit as st
import yt_dlp

from utils import (
    LivePoseTracker,
    classify_exercise,
    process_video,
    render_skeleton_on_frame,
    render_skeleton_video,
)


st.set_page_config(page_title="Workout Hacker AI", layout="wide")
st.title("Workout Hacker AI: Exercise Recognition")


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output_data")
MODEL_PATH = os.path.join(
    os.path.dirname(SCRIPT_DIR), "models", "exercise_classifier_rf.joblib"
)


def update_progress(progress):
    progress_bar = st.session_state.get("progress_bar")
    status_text = st.session_state.get("status_text")
    if progress_bar is not None and status_text is not None:
        value = float(progress) if progress is not None else 0.0
        progress_bar.progress(value)
        status_text.text(f"Processing: {int(value * 100)}%")


@st.cache_resource
def load_classifier(path):
    if os.path.exists(path):
        return joblib.load(path)
    return None


def ensure_output_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_video_props(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if fps <= 0:
        fps = 30.0
    duration = total_frames / fps if fps > 0 else 0.0
    return fps, total_frames, duration


def render_video_editor(source_path, fps, total_frames, key_prefix):
    st.subheader("Video Editor")
    col1, col2 = st.columns(2)
    with col1:
        start_frame = st.number_input(
            "Start Frame", 0, total_frames, 0, key=f"{key_prefix}_start_frame"
        )
    with col2:
        end_frame = st.number_input(
            "End Frame",
            0,
            total_frames,
            total_frames,
            key=f"{key_prefix}_end_frame",
        )

    start_time = start_frame / fps if fps > 0 else 0.0
    end_time = end_frame / fps if fps > 0 else 0.0
    st.info(
        f"Time Range: {start_time:.2f}s - {end_time:.2f}s | "
        f"Total Frames: {total_frames} (FPS: {fps:.2f})"
    )

    preview_frame = st.number_input(
        "Preview at Frame",
        int(start_frame),
        int(end_frame),
        int(start_frame),
        key=f"{key_prefix}_preview_frame",
    )

    cap = cv2.VideoCapture(source_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, preview_frame)
    ret, frame = cap.read()
    cap.release()
    if ret:
        st.image(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            use_container_width=True,
            caption=f"Frame {preview_frame} ({preview_frame / fps:.2f}s)",
        )

    return start_time, end_time, preview_frame


def get_uploaded_source(uploaded_file):
    if "upload_cache" not in st.session_state:
        st.session_state.upload_cache = {}

    file_key = f"{uploaded_file.name}_{uploaded_file.size}"
    cached = st.session_state.upload_cache.get(file_key)
    if cached and os.path.exists(cached):
        return cached

    suffix = f'.{uploaded_file.name.split(".")[-1]}' if "." in uploaded_file.name else ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tfile:
        tfile.write(uploaded_file.read())
        source_path = tfile.name
    st.session_state.upload_cache[file_key] = source_path
    uploaded_file.seek(0)
    return source_path


def get_youtube_info(youtube_url):
    if "yt_info_cache" not in st.session_state:
        st.session_state.yt_info_cache = {}

    cached = st.session_state.yt_info_cache.get(youtube_url)
    if isinstance(cached, dict):
        return cached

    try:
        with yt_dlp.YoutubeDL({"quiet": True}) as ydl:  # type: ignore
            info = ydl.extract_info(youtube_url, download=False)
            data = {
                "duration": info.get("duration", 0.0) or 0.0,
                "fps": info.get("fps", 30.0) or 30.0,
            }
            st.session_state.yt_info_cache[youtube_url] = data
            return data
    except Exception:
        return {"duration": 0.0, "fps": 30.0}


def run_live_mode(classifier_model, use_gpu, model_complexity):
    st.info("Click the button below to start live exercise recognition.")
    run_live = st.toggle("Start Live Camera")
    if not run_live:
        return

    st.subheader("Live Inference")
    frame_window = st.image([])
    status_metric = st.empty()

    cap = cv2.VideoCapture(0)
    tracker = LivePoseTracker(use_gpu=use_gpu, model_complexity=model_complexity)
    history = []
    window_size = 30

    last_prediction = "Null/Unknown"
    last_change_time = 0.0

    try:
        live_start = time.time()
        while run_live:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access camera.")
                break

            h, w, _ = frame.shape
            timestamp_ms = int((time.time() - live_start) * 1000)
            landmarks = tracker.process_frame(frame, timestamp_ms)

            if landmarks:
                history.append(landmarks)
                if len(history) > window_size:
                    history.pop(0)

                if classifier_model and len(history) >= 10:
                    new_exercise, _ = classify_exercise(history, classifier_model)
                    now = time.time()
                    if (
                        new_exercise != last_prediction
                        and now - last_change_time >= 3.0
                    ):
                        last_prediction = new_exercise
                        last_change_time = now
                    status_metric.metric("Detected Exercise", last_prediction)

                frame = render_skeleton_on_frame(frame, landmarks, w, h)

            frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            time.sleep(0.01)
    except Exception as exc:
        st.error(f"Live Error: {exc}")
    finally:
        cap.release()
        tracker.close()


def analyze_video(
    input_option,
    source_path,
    input_filename,
    start_time,
    end_time,
    youtube_url,
    classifier_model,
    use_gpu,
    model_complexity,
):
    try:
        if input_option == "YouTube URL" and youtube_url:
            with st.spinner("Downloading YouTube video..."):
                ydl_opts = {
                    "format": "best[ext=mp4]/best",
                    "outtmpl": os.path.join(tempfile.gettempdir(), "%(title)s.%(ext)s"),
                    "noplaylist": True,
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:  # type: ignore
                    info = ydl.extract_info(youtube_url, download=True)
                    source_path = ydl.prepare_filename(info)
            input_filename = os.path.basename(source_path)
        elif input_option == "Upload Video" and not source_path:
            st.error("Please provide a valid input.")
            st.stop()

        if not source_path:
            return

        st.session_state.progress_bar = st.progress(0)
        st.session_state.status_text = st.empty()

        with st.spinner("Step 1: Extracting Pose Landmarks..."):
            skeleton_data = process_video(
                source_path,
                start_seconds=start_time,
                end_seconds=end_time,
                progress_callback=update_progress,
                use_gpu=use_gpu,
                model_complexity=model_complexity,
            )

        if not skeleton_data:
            st.warning("No pose data extracted from selected range.")
            return

        input_name = os.path.splitext(str(input_filename))[0]

        if classifier_model:
            with st.spinner("Step 2: Classifying Exercise..."):
                exercise_name, _ = classify_exercise(skeleton_data, classifier_model)
                st.metric(label="Predicted Exercise", value=exercise_name)
        else:
            st.warning(
                "Classifier model not found at 'models/exercise_classifier_rf.joblib'. "
                "Skipping classification."
            )

        output_filename = f"{input_name}_skeleton_raw.json"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        with open(output_path, "w") as f:
            json.dump(skeleton_data, f, indent=4)

        with st.spinner("Step 3: Rendering visualization..."):
            render_filename = f"{input_name}_skeleton_raw_rendered.mp4"
            render_path = os.path.join(OUTPUT_DIR, render_filename)
            render_skeleton_video(skeleton_data, render_path)

        st.success("Analysis complete!")
        col_vid, col_info = st.columns([2, 1])

        with col_vid:
            st.video(render_path)

        with col_info:
            st.subheader("Data Downloads")
            with open(output_path, "r") as f:
                st.download_button(
                    label="Download Skeleton JSON",
                    data=f.read(),
                    file_name=output_filename,
                    mime="application/json",
                )
            with open(render_path, "rb") as f:
                st.download_button(
                    label="Download Rendered MP4",
                    data=f.read(),
                    file_name=render_filename,
                    mime="video/mp4",
                )
    except Exception as exc:
        st.error(f"Error processing video: {exc}")
    finally:
        if (
            input_option == "YouTube URL"
            and source_path
            and os.path.exists(source_path)
        ):
            os.remove(source_path)


ensure_output_dir(OUTPUT_DIR)
classifier_model = load_classifier(MODEL_PATH)

st.sidebar.title("Configuration")
device_option = st.sidebar.radio(
    "Processing Device",
    ("CPU", "GPU"),
    index=1,
    help=(
        "Select GPU for faster processing. If MediaPipe GPU graph init fails, switch to CPU."
    ),
)
use_gpu = device_option == "GPU"

model_complexity = st.sidebar.selectbox(
    "Pose Model Complexity",
    ("lite", "full", "heavy"),
    index=1,
    help="Lite: fastest. Full: balanced. Heavy: most accurate and slowest.",
)

input_option = st.radio("Select Input Source", ("Upload Video", "YouTube URL", "Device Camera"))

source_path = None
input_filename = None
start_time = 0.0
end_time = 0.0
youtube_url = ""

if input_option == "Upload Video":
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])
    if uploaded_file is not None:
        source_path = get_uploaded_source(uploaded_file)
        input_filename = uploaded_file.name
        fps, total_frames, _ = get_video_props(source_path)
        start_time, end_time, preview_frame = render_video_editor(
            source_path, fps, total_frames, "upload"
        )
        st.video(uploaded_file, start_time=int(preview_frame / fps))

elif input_option == "YouTube URL":
    youtube_url = st.text_input("Enter YouTube URL")
    if youtube_url:
        yt_info = get_youtube_info(youtube_url)
        duration = yt_info.get("duration", 0.0)
        fps = yt_info.get("fps", 30.0)
        total_frames = int(duration * fps)

        if duration > 0:
            st.subheader("Video Editor")
            col1, col2 = st.columns(2)
            with col1:
                start_frame = st.number_input(
                    "Start Frame", 0, total_frames, 0, key="yt_start_frame"
                )
            with col2:
                end_frame = st.number_input(
                    "End Frame",
                    0,
                    total_frames,
                    total_frames,
                    key="yt_end_frame",
                )
            start_time = start_frame / fps
            end_time = end_frame / fps
            st.info(
                f"Time Range: {start_time:.2f}s - {end_time:.2f}s | "
                f"Total Frames: {total_frames} (FPS: ~{fps:.2f})"
            )
            preview_frame = st.number_input(
                "Preview at Frame",
                int(start_frame),
                int(end_frame),
                int(start_frame),
                key="yt_preview_frame",
            )
            st.video(youtube_url, start_time=int(preview_frame / fps))
        else:
            st.video(youtube_url)

else:
    run_live_mode(classifier_model, use_gpu, model_complexity)


if st.button("Analyze Exercise", disabled=input_option == "Device Camera"):
    analyze_video(
        input_option=input_option,
        source_path=source_path,
        input_filename=input_filename,
        start_time=start_time,
        end_time=end_time,
        youtube_url=youtube_url,
        classifier_model=classifier_model,
        use_gpu=use_gpu,
        model_complexity=model_complexity,
    )
