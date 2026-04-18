use crate::FrameInput;

const NUM_LANDMARKS: usize = 33;
const NUM_DIMS: usize = 4;
const SMOOTHING_WINDOW: usize = 5;
const FEATURE_STD_WINDOW: usize = 15;
const ARM_JOINTS: [usize; 6] = [11, 12, 13, 14, 15, 16];
const FEATURE_COUNT: usize = 60;

fn calculate_angle(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
    let ba = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let bc = [c[0] - b[0], c[1] - b[1], c[2] - b[2]];

    let dot = ba[0] * bc[0] + ba[1] * bc[1] + ba[2] * bc[2];
    let ba_norm = (ba[0] * ba[0] + ba[1] * ba[1] + ba[2] * ba[2]).sqrt();
    let bc_norm = (bc[0] * bc[0] + bc[1] * bc[1] + bc[2] * bc[2]).sqrt();

    let cosine = dot / (ba_norm * bc_norm + 1e-6);
    cosine.clamp(-1.0, 1.0).acos().to_degrees()
}

fn build_frame_array(frames: &[FrameInput]) -> Vec<[[f64; NUM_DIMS]; NUM_LANDMARKS]> {
    let mut arr = vec![[[0.0; NUM_DIMS]; NUM_LANDMARKS]; frames.len()];
    for (i, frame) in frames.iter().enumerate() {
        for (j, lm) in frame.landmarks().iter().take(NUM_LANDMARKS).enumerate() {
            arr[i][j] = [lm.x, lm.y, lm.z, lm.visibility.unwrap_or(1.0)];
        }
    }
    arr
}

fn smooth_landmarks(arr: &mut Vec<[[f64; NUM_DIMS]; NUM_LANDMARKS]>, smoothing_window: usize) {
    let n = arr.len();
    if n < smoothing_window || smoothing_window == 0 {
        return;
    }

    let half_window = smoothing_window / 2;
    let mut smoothed = arr.clone();

    for j in 0..NUM_LANDMARKS {
        for k in 0..3 {
            for i in 0..n {
                let start = i.saturating_sub(half_window);
                let end = (i + half_window).min(n - 1);
                let mut sum = 0.0;
                let mut count = 0usize;
                for t in start..=end {
                    sum += arr[t][j][k];
                    count += 1;
                }
                smoothed[i][j][k] = sum / count as f64;
            }
        }
    }

    *arr = smoothed;
}

fn normalize_skeleton_for_classifier(arr: &mut Vec<[[f64; NUM_DIMS]; NUM_LANDMARKS]>) {
    for frame in arr {
        let left_hip = [frame[23][0], frame[23][1], frame[23][2]];
        let right_hip = [frame[24][0], frame[24][1], frame[24][2]];
        let hip_center = [
            (left_hip[0] + right_hip[0]) / 2.0,
            (left_hip[1] + right_hip[1]) / 2.0,
            (left_hip[2] + right_hip[2]) / 2.0,
        ];

        for lm in frame.iter_mut() {
            lm[0] -= hip_center[0];
            lm[1] -= hip_center[1];
            lm[2] -= hip_center[2];
        }

        let left_shoulder = [frame[11][0], frame[11][1], frame[11][2]];
        let right_shoulder = [frame[12][0], frame[12][1], frame[12][2]];
        let shoulder_dist = ((left_shoulder[0] - right_shoulder[0]).powi(2)
            + (left_shoulder[1] - right_shoulder[1]).powi(2)
            + (left_shoulder[2] - right_shoulder[2]).powi(2))
        .sqrt();

        if shoulder_dist > 0.0 {
            for lm in frame.iter_mut() {
                lm[0] /= shoulder_dist;
                lm[1] /= shoulder_dist;
                lm[2] /= shoulder_dist;
            }
        }
    }
}

fn rolling_std(values: &[f64], end_idx: usize, window: usize) -> f64 {
    let start = (end_idx + 1).saturating_sub(window);
    let slice = &values[start..=end_idx];
    let n = slice.len();

    if n <= 1 {
        return 0.0;
    }

    let mean = slice.iter().sum::<f64>() / n as f64;
    let var = slice
        .iter()
        .map(|v| {
            let d = *v - mean;
            d * d
        })
        .sum::<f64>()
        / (n as f64 - 1.0);

    var.sqrt()
}

fn extract_temporal_features(arr: &[[[f64; NUM_DIMS]; NUM_LANDMARKS]]) -> Vec<Vec<f64>> {
    let num_frames = arr.len();
    let mut out = Vec::with_capacity(num_frames);

    let mut coords_hist = vec![vec![Vec::with_capacity(num_frames); 3]; ARM_JOINTS.len()];
    let mut prev_vel = vec![[0.0; 3]; ARM_JOINTS.len()];

    for i in 0..num_frames {
        let mut row = Vec::with_capacity(FEATURE_COUNT);

        let frame = &arr[i];
        let l_shoulder = [frame[11][0], frame[11][1], frame[11][2]];
        let r_shoulder = [frame[12][0], frame[12][1], frame[12][2]];
        let l_elbow = [frame[13][0], frame[13][1], frame[13][2]];
        let r_elbow = [frame[14][0], frame[14][1], frame[14][2]];
        let l_wrist = [frame[15][0], frame[15][1], frame[15][2]];
        let r_wrist = [frame[16][0], frame[16][1], frame[16][2]];
        let l_hip = [frame[23][0], frame[23][1], frame[23][2]];
        let r_hip = [frame[24][0], frame[24][1], frame[24][2]];

        row.push(calculate_angle(l_shoulder, l_elbow, l_wrist));
        row.push(calculate_angle(r_shoulder, r_elbow, r_wrist));
        row.push(calculate_angle(l_hip, l_shoulder, l_elbow));
        row.push(calculate_angle(r_hip, r_shoulder, r_elbow));
        row.push(l_wrist[1] - l_shoulder[1]);
        row.push(r_wrist[1] - r_shoulder[1]);

        for (joint_i, joint_idx) in ARM_JOINTS.iter().enumerate() {
            let coord = [
                frame[*joint_idx][0],
                frame[*joint_idx][1],
                frame[*joint_idx][2],
            ];

            for axis in 0..3 {
                coords_hist[joint_i][axis].push(coord[axis]);
            }

            let vel = if i == 0 {
                [0.0, 0.0, 0.0]
            } else {
                let prev = &arr[i - 1][*joint_idx];
                [coord[0] - prev[0], coord[1] - prev[1], coord[2] - prev[2]]
            };

            let acc = if i == 0 {
                [0.0, 0.0, 0.0]
            } else {
                [
                    vel[0] - prev_vel[joint_i][0],
                    vel[1] - prev_vel[joint_i][1],
                    vel[2] - prev_vel[joint_i][2],
                ]
            };
            prev_vel[joint_i] = vel;

            row.extend_from_slice(&vel);
            row.extend_from_slice(&acc);

            let stdx = rolling_std(&coords_hist[joint_i][0], i, FEATURE_STD_WINDOW);
            let stdy = rolling_std(&coords_hist[joint_i][1], i, FEATURE_STD_WINDOW);
            let stdz = rolling_std(&coords_hist[joint_i][2], i, FEATURE_STD_WINDOW);
            row.push(stdx);
            row.push(stdy);
            row.push(stdz);
        }

        out.push(row);
    }

    out
}

pub fn extract_feature_rows(frames: &[FrameInput]) -> Vec<Vec<f32>> {
    if frames.is_empty() {
        return Vec::new();
    }

    let mut arr = build_frame_array(frames);
    smooth_landmarks(&mut arr, SMOOTHING_WINDOW);
    normalize_skeleton_for_classifier(&mut arr);

    extract_temporal_features(&arr)
        .into_iter()
        .map(|row| row.into_iter().map(|v| v as f32).collect())
        .collect()
}
