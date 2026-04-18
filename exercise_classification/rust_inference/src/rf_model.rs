use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::fs;
use std::path::Path;
use std::time::Instant;

const FEATURE_COUNT: usize = 60;

#[derive(Debug, Deserialize)]
struct ForestFile {
    n_features: usize,
    n_classes: usize,
    classes: Vec<i64>,
    trees: Vec<TreeFile>,
}

#[derive(Debug, Deserialize)]
struct TreeFile {
    children_left: Vec<i32>,
    children_right: Vec<i32>,
    feature: Vec<i32>,
    threshold: Vec<f64>,
    values: Vec<Vec<f64>>,
}

#[derive(Debug)]
struct TreeModel {
    children_left: Vec<i32>,
    children_right: Vec<i32>,
    feature: Vec<i32>,
    threshold: Vec<f64>,
    values: Vec<Vec<f64>>,
}

#[derive(Debug)]
pub struct RandomForestRunner {
    n_features: usize,
    n_classes: usize,
    classes: Vec<i64>,
    trees: Vec<TreeModel>,
}

impl TreeModel {
    fn predict_proba(&self, row: &[f32], n_classes: usize) -> Result<Vec<f32>> {
        let mut node: usize = 0;
        loop {
            let left = *self
                .children_left
                .get(node)
                .with_context(|| format!("tree node {} out of bounds", node))?;
            let right = *self
                .children_right
                .get(node)
                .with_context(|| format!("tree node {} out of bounds", node))?;

            if left == -1 && right == -1 {
                let counts = self
                    .values
                    .get(node)
                    .with_context(|| format!("leaf values missing for node {}", node))?;
                if counts.len() != n_classes {
                    bail!(
                        "leaf class count size {} does not match n_classes {}",
                        counts.len(),
                        n_classes
                    );
                }

                let total: f64 = counts.iter().sum();
                if total <= 0.0 {
                    return Ok(vec![0.0; n_classes]);
                }

                let mut proba = vec![0.0_f32; n_classes];
                for (i, &count) in counts.iter().enumerate() {
                    proba[i] = (count / total) as f32;
                }
                return Ok(proba);
            }

            let feature_idx = *self
                .feature
                .get(node)
                .with_context(|| format!("feature missing for node {}", node))?;
            if feature_idx < 0 {
                bail!("invalid feature index {} at node {}", feature_idx, node);
            }
            let feature_idx = feature_idx as usize;
            let threshold = *self
                .threshold
                .get(node)
                .with_context(|| format!("threshold missing for node {}", node))?;

            let feature_value = *row
                .get(feature_idx)
                .with_context(|| format!("row missing feature {}", feature_idx))?
                as f64;

            node = if feature_value <= threshold {
                left as usize
            } else {
                right as usize
            };
        }
    }
}

impl RandomForestRunner {
    pub fn load(model_path: &Path, debug_timing: bool) -> Result<Self> {
        let load_start = Instant::now();
        let raw = fs::read_to_string(model_path)
            .with_context(|| format!("failed to read RF model file: {}", model_path.display()))?;
        let file: ForestFile = serde_json::from_str(&raw)
            .with_context(|| format!("failed to parse RF model json: {}", model_path.display()))?;

        if file.n_features != FEATURE_COUNT {
            bail!(
                "RF model expects {} features, code expects {}",
                file.n_features,
                FEATURE_COUNT
            );
        }

        let mut trees = Vec::with_capacity(file.trees.len());
        for t in file.trees {
            let node_count = t.children_left.len();
            if t.children_right.len() != node_count
                || t.feature.len() != node_count
                || t.threshold.len() != node_count
                || t.values.len() != node_count
            {
                bail!("inconsistent node array lengths in one tree");
            }
            trees.push(TreeModel {
                children_left: t.children_left,
                children_right: t.children_right,
                feature: t.feature,
                threshold: t.threshold,
                values: t.values,
            });
        }

        if debug_timing {
            eprintln!(
                "[debug] RF load+prepare: {:.3}s (trees={})",
                load_start.elapsed().as_secs_f64(),
                trees.len()
            );
        }

        Ok(Self {
            n_features: file.n_features,
            n_classes: file.n_classes,
            classes: file.classes,
            trees,
        })
    }

    pub fn predict_probabilities(
        &self,
        feature_rows: &[Vec<f32>],
        debug_timing: bool,
    ) -> Result<Vec<Vec<f32>>> {
        if feature_rows.is_empty() {
            return Ok(Vec::new());
        }

        let infer_start = Instant::now();
        let mut all_probs = Vec::with_capacity(feature_rows.len());

        for row in feature_rows {
            if row.len() != self.n_features {
                bail!(
                    "feature row has {} values, expected {}",
                    row.len(),
                    self.n_features
                );
            }

            let mut sum_probs = vec![0.0_f32; self.n_classes];
            for tree in &self.trees {
                let tree_probs = tree.predict_proba(row, self.n_classes)?;
                for (i, p) in tree_probs.iter().enumerate() {
                    sum_probs[i] += *p;
                }
            }

            let inv = 1.0_f32 / self.trees.len() as f32;
            for p in &mut sum_probs {
                *p *= inv;
            }
            all_probs.push(sum_probs);
        }

        if debug_timing {
            eprintln!(
                "[debug] RF inference end-to-end: {:.3}s (rows={})",
                infer_start.elapsed().as_secs_f64(),
                all_probs.len()
            );
        }

        Ok(all_probs)
    }

    pub fn class_ids(&self) -> &[i64] {
        &self.classes
    }
}
