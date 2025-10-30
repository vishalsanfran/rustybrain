//! Rolling Reward Tracker
//!
//! Maintains a sliding window of the most recent N rewards,
//! exposing mean, min, max, and count. This is useful for
//! tracking recent performance trends or stabilizing feedback
//! in adaptive systems.

#[derive(Debug, Clone)]
pub struct RewardTracker {
    window: usize,
    values: Vec<f64>,
}

impl RewardTracker {
    /// Creates a new tracker with a given window size.
    pub fn new(window: usize) -> Self {
        assert!(window > 0, "window size must be > 0");
        Self {
            window,
            values: Vec::with_capacity(window),
        }
    }

    /// Adds a new reward to the tracker, evicting the oldest if full.
    pub fn update(&mut self, reward: f64) {
        if self.values.len() == self.window {
            self.values.remove(0);
        }
        self.values.push(reward);
    }

    /// Returns the mean of stored rewards.
    pub fn mean(&self) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }
        self.values.iter().sum::<f64>() / self.values.len() as f64
    }

    /// Returns the minimum reward seen in the current window.
    pub fn min(&self) -> f64 {
        self.values
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min)
            .is_finite()
            .then(|| self.values.iter().cloned().fold(f64::INFINITY, f64::min))
            .unwrap_or(0.0)
    }

    /// Returns the maximum reward seen in the current window.
    pub fn max(&self) -> f64 {
        self.values
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
            .is_finite()
            .then(|| self.values.iter().cloned().fold(f64::NEG_INFINITY, f64::max))
            .unwrap_or(0.0)
    }

    /// Returns the number of rewards currently stored.
    pub fn count(&self) -> usize {
        self.values.len()
    }

    /// Returns all stored rewards (for debugging/inspection).
    pub fn values(&self) -> &[f64] {
        &self.values
    }
}