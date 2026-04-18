mod features;
mod rf_model;

use anyhow::{Context, Result};
use clap::Parser;
use serde::Deserialize;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "exercise-recognition")]
#[command(about = "Run exercise recognition with Rust RandomForest model")]
struct Cli {
    #[arg(long)]
    input: PathBuf,

    #[arg(long, default_value = "model_rf.json")]
    model: PathBuf,

    #[arg(long, default_value_t = false)]
    debug_timing: bool,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
pub enum FrameInput {
    FrameRecord { landmarks: Vec<Landmark> },
    LandmarkList(Vec<Landmark>),
}

impl FrameInput {
    pub fn landmarks(&self) -> &[Landmark] {
        match self {
            FrameInput::FrameRecord { landmarks } => landmarks,
            FrameInput::LandmarkList(landmarks) => landmarks,
        }
    }
}

#[derive(Debug, Deserialize, Clone, Copy)]
pub struct Landmark {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub visibility: Option<f64>,
}

fn label_for_class_id(class_id: i64) -> &'static str {
    match class_id {
        0 => "Bicep Curl",
        1 => "Lateral Raise",
        2 => "Null/Unknown",
        3 => "Shoulder Press",
        4 => "Triceps Extension",
        5 => "Front Raises",
        _ => "Unknown",
    }
}

fn argmax(values: &[f32]) -> usize {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in values.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    best_i
}

fn average_probabilities(frame_probs: &[Vec<f32>]) -> Vec<f32> {
    if frame_probs.is_empty() {
        return Vec::new();
    }
    let mut avg = vec![0.0_f32; frame_probs[0].len()];
    for probs in frame_probs {
        for (i, &p) in probs.iter().enumerate().take(avg.len()) {
            avg[i] += p;
        }
    }

    let n = frame_probs.len() as f32;
    if n > 0.0 {
        for v in &mut avg {
            *v /= n;
        }
    }
    avg
}

fn print_prediction(avg: &[f32], class_ids: &[i64]) {
    let winner_idx = argmax(avg);
    let winner_class_id = class_ids
        .get(winner_idx)
        .copied()
        .unwrap_or(winner_idx as i64);

    println!(
        "Predicted Exercise: {}",
        label_for_class_id(winner_class_id)
    );
    println!("Class Probabilities:");

    for (i, &class_id) in class_ids.iter().enumerate() {
        let class_label = label_for_class_id(class_id);
        let score = avg.get(i).copied().unwrap_or(0.0);
        println!("  {}: {:.6}", class_label, score);
    }
}

fn main() -> Result<()> {
    let total_start = Instant::now();
    let cli = Cli::parse();

    if cli.debug_timing {
        eprintln!("[debug] input: {}", cli.input.display());
        eprintln!("[debug] model: {}", cli.model.display());
    }

    let io_start = Instant::now();
    let raw = fs::read_to_string(&cli.input)
        .with_context(|| format!("failed to read input file: {}", cli.input.display()))?;
    if cli.debug_timing {
        eprintln!(
            "[debug] read input JSON: {:.3}s",
            io_start.elapsed().as_secs_f64()
        );
    }

    let parse_start = Instant::now();
    let frames: Vec<FrameInput> = serde_json::from_str(&raw)
        .with_context(|| format!("failed to parse json: {}", cli.input.display()))?;
    if cli.debug_timing {
        eprintln!(
            "[debug] parse input JSON: {:.3}s (frames={})",
            parse_start.elapsed().as_secs_f64(),
            frames.len()
        );
    }

    if frames.is_empty() {
        println!("Predicted Exercise: Unknown (empty input)");
        return Ok(());
    }

    let valid_landmark_frames = frames
        .iter()
        .filter(|frame| !frame.landmarks().is_empty())
        .count();
    if valid_landmark_frames == 0 {
        println!("Predicted Exercise: Unknown (no valid landmarks)");
        return Ok(());
    }

    let prep_start = Instant::now();
    let feature_rows = features::extract_feature_rows(&frames);
    if cli.debug_timing {
        eprintln!(
            "[debug] feature extraction: {:.3}s (rows={})",
            prep_start.elapsed().as_secs_f64(),
            feature_rows.len()
        );
    }
    if feature_rows.is_empty() {
        println!("Predicted Exercise: Unknown (no extracted features)");
        return Ok(());
    }

    let runner = rf_model::RandomForestRunner::load(&cli.model, cli.debug_timing)
        .with_context(|| format!("failed RF setup with {}", cli.model.display()))?;
    let frame_probs = runner
        .predict_probabilities(&feature_rows, cli.debug_timing)
        .with_context(|| format!("failed RF inference with {}", cli.model.display()))?;
    if frame_probs.is_empty() {
        println!("Predicted Exercise: Unknown (empty model output)");
        return Ok(());
    }

    let avg = average_probabilities(&frame_probs);
    print_prediction(&avg, runner.class_ids());

    if cli.debug_timing {
        eprintln!(
            "[debug] total runtime: {:.3}s",
            total_start.elapsed().as_secs_f64()
        );
    }

    Ok(())
}
