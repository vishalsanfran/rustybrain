//! # ε-Greedy Multi-Armed Bandit
//!
//! A simple, well-known algorithm for balancing **exploration** and **exploitation**
//! when choosing between multiple uncertain reward sources (or "arms").
//!
//! ## Overview
//!
//! In the ε-Greedy strategy, the agent maintains an estimated mean reward
//! for each arm. On each selection:
//!
//! * With probability **ε**, it **explores** by selecting a random arm.  
//! * With probability **1 − ε**, it **exploits** by choosing the arm with the
//!   highest estimated mean reward.
//!
//! Over time, these estimates converge toward the true expected rewards,
//! balancing discovery of new options with refinement of known good ones.
//!
//! Mathematically, for each arm *i*, after receiving reward *r*,  
//! the running average is updated as:
//!
//! ```text
//! value[i] ← value[i] + (r − value[i]) / count[i]
//! ```
//!
//! ## Example
//!
//! ```
//! use rustybrain::bandit::epsilon_greedy::EpsilonGreedy;
//!
//! // Create an agent with 3 arms and 10% exploration rate.
//! let mut bandit = EpsilonGreedy::new(3, 0.1);
//!
//! // Choose an arm to pull.
//! let arm = bandit.select_arm();
//!
//! // Report the observed reward for that arm.
//! bandit.update(arm, 1.0);
//!
//! // Inspect current estimates.
//! println!("Arm values: {:?}", bandit.values());
//! ```
//!
//! ## Determinism
//!
//! The internal RNG (`StdRng`) is seeded with a fixed value for reproducible tests.
//!
//! ## Complexity
//!
//! * Selection: **O(k)** to find max over k arms.  
//! * Update: **O(1)** per reward.

use rand::{rngs::StdRng, Rng, SeedableRng};

/// ε-Greedy multi-armed bandit agent.
///
/// Maintains average reward estimates for each arm and selects arms
/// according to the ε-greedy exploration policy.
#[derive(Debug, Clone)]
pub struct EpsilonGreedy {
    /// Exploration probability (0.0 = always exploit, 1.0 = always explore).
    epsilon: f64,
    /// Number of times each arm has been selected.
    counts: Vec<u64>,
    /// Current estimated mean reward for each arm.
    values: Vec<f64>,
    /// Deterministic random number generator for reproducibility.
    rng: StdRng,
}

impl EpsilonGreedy {
    /// Creates a new ε-Greedy agent with `num_arms` choices and exploration rate `epsilon`.
    ///
    /// # Panics
    /// - If `num_arms == 0`
    /// - If `epsilon` is outside `[0.0, 1.0]`
    pub fn new(num_arms: usize, epsilon: f64) -> Self {
        assert!(num_arms > 0, "must have at least one arm");
        assert!(
            (0.0..=1.0).contains(&epsilon),
            "epsilon must be between 0.0 and 1.0"
        );

        Self {
            epsilon,
            counts: vec![0; num_arms],
            values: vec![0.0; num_arms],
            rng: StdRng::seed_from_u64(42), // deterministic seed for reproducibility
        }
    }

    /// Selects an arm index according to the ε-greedy policy.
    ///
    /// * With probability `epsilon`, a random arm is chosen (exploration).  
    /// * Otherwise, the arm with the highest estimated value is selected (exploitation).
    pub fn select_arm(&mut self) -> usize {
        let p: f64 = self.rng.gen();
        if p < self.epsilon {
            // Explore
            self.rng.gen_range(0..self.values.len())
        } else {
            // Exploit
            self.argmax()
        }
    }

    /// Updates the reward statistics for the chosen arm.
    ///
    /// Uses an incremental mean update rule that does not require storing
    /// the full history of rewards.
    ///
    /// # Example
    /// ```
    /// let mut agent = rustybrain::bandit::epsilon_greedy::EpsilonGreedy::new(1, 0.0);
    /// agent.update(0, 1.0);
    /// agent.update(0, 3.0);
    /// assert_eq!(agent.values()[0], 2.0);
    /// ```
    pub fn update(&mut self, chosen_arm: usize, reward: f64) {
        let n = self.counts[chosen_arm] + 1;
        let value = self.values[chosen_arm];
        let new_value = value + (reward - value) / n as f64;

        self.counts[chosen_arm] = n;
        self.values[chosen_arm] = new_value;
    }

    /// Internal helper: returns the index of the arm with the highest estimated reward.
    fn argmax(&self) -> usize {
        let mut max_index = 0;
        let mut max_value = f64::NEG_INFINITY;
        for (i, &v) in self.values.iter().enumerate() {
            if v > max_value {
                max_value = v;
                max_index = i;
            }
        }
        max_index
    }

    /// Returns the number of times each arm has been selected.
    pub fn counts(&self) -> &[u64] {
        &self.counts
    }

    /// Returns the current estimated mean reward values for all arms.
    pub fn values(&self) -> &[f64] {
        &self.values
    }
}