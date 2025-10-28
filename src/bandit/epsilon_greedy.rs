//! ε-Greedy Multi-Armed Bandit
//!
//! Selects between multiple reward sources using a probabilistic exploration-exploitation strategy.
//!
//! Example:
//! ```
//! use rustybrain::bandit::epsilon_greedy::EpsilonGreedy;
//!
//! let mut bandit = EpsilonGreedy::new(3, 0.1);
//! let arm = bandit.select_arm();
//! bandit.update(arm, 1.0);
//! ```

use rand::{rngs::StdRng, Rng, SeedableRng};

#[derive(Debug, Clone)]
pub struct EpsilonGreedy {
    epsilon: f64,
    counts: Vec<u64>,
    values: Vec<f64>,
    rng: StdRng,
}

impl EpsilonGreedy {
    /// Create a new ε-Greedy agent with `num_arms` choices and exploration rate `epsilon`.
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
            rng: StdRng::seed_from_u64(42), // deterministic seed for tests
        }
    }

    /// Select an arm index using ε-greedy policy.
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

    /// Update reward statistics for the chosen arm.
    pub fn update(&mut self, chosen_arm: usize, reward: f64) {
        let n = self.counts[chosen_arm] + 1;
        let value = self.values[chosen_arm];
        let new_value = value + (reward - value) / n as f64;

        self.counts[chosen_arm] = n;
        self.values[chosen_arm] = new_value;
    }

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

    /// Accessors for tests / metrics
    pub fn counts(&self) -> &[u64] {
        &self.counts
    }
    pub fn values(&self) -> &[f64] {
        &self.values
    }
}