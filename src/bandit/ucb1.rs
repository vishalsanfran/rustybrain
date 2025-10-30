//! # UCB1 Multi-Armed Bandit
//!
//! Deterministic bandit algorithm balancing exploration and exploitation
//! using the Upper Confidence Bound principle.
//!
//! For each arm i, select the one maximizing:
//! ```text
//! score_i = value_i + c * sqrt(2 * ln t / n_i)
//! ```
//! where t is the total number of pulls.
//!
//! ## Example
//! ```
//! use rustybrain::bandit::ucb1::Ucb1;
//!
//! let mut agent = Ucb1::new(3, 2.0);
//! let arm = agent.select_arm();
//! agent.update(arm, 1.0);
//! ```

use std::f64;

/// UCB1 Bandit implementation.
///
/// Deterministic exploration-exploitation balance using confidence intervals.
#[derive(Debug, Clone)]
pub struct Ucb1 {
    /// Exploration parameter (controls aggressiveness of exploration).
    c: f64,
    /// Number of pulls for each arm.
    counts: Vec<u64>,
    /// Average reward for each arm.
    values: Vec<f64>,
}

impl Ucb1 {
    /// Create a new UCB1 agent with `num_arms` and exploration factor `c`.
    pub fn new(num_arms: usize, c: f64) -> Self {
        assert!(num_arms > 0, "must have at least one arm");
        assert!(c >= 0.0, "c must be non-negative");
        Self {
            c,
            counts: vec![0; num_arms],
            values: vec![0.0; num_arms],
        }
    }

    /// Selects the next arm index based on UCB1 formula.
    pub fn select_arm(&self) -> usize {
        // total pulls so far
        let total: u64 = self.counts.iter().sum();

        // If any arm hasn't been tried yet, pick it first.
        if let Some((idx, _)) = self.counts.iter().enumerate().find(|(_, &n)| n == 0) {
            return idx;
        }

        // Compute UCB1 score for each arm
        let t = total as f64;
        let mut best_arm = 0;
        let mut best_score = f64::NEG_INFINITY;

        for i in 0..self.values.len() {
            let mean = self.values[i];
            let n = self.counts[i] as f64;
            let bonus = self.c * (2.0 * t.ln() / n).sqrt();
            let score = mean + bonus;
            if score > best_score {
                best_score = score;
                best_arm = i;
            }
        }
        best_arm
    }

    /// Updates the reward statistics for the selected arm.
    pub fn update(&mut self, chosen_arm: usize, reward: f64) {
        let n = self.counts[chosen_arm] + 1;
        let old_value = self.values[chosen_arm];
        let new_value = old_value + (reward - old_value) / n as f64;
        self.counts[chosen_arm] = n;
        self.values[chosen_arm] = new_value;
    }

    /// Returns total number of selections per arm.
    pub fn counts(&self) -> &[u64] {
        &self.counts
    }

    /// Returns average rewards per arm.
    pub fn values(&self) -> &[f64] {
        &self.values
    }
}