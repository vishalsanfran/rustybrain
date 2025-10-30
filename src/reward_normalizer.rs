//! # RewardNormalizer
//!
//! A utility for dynamically rescaling **streaming rewards** into a stable
//! normalized range of `[0.0, 1.0]` using a rolling mean and standard deviation.
//!
//! ## Motivation
//!
//! In reinforcement learning or online optimization systems, reward values can
//! vary widely in scale or units (e.g., latency in milliseconds vs. accuracy
//! in percentages).  The `RewardNormalizer` smooths these variations, ensuring
//! that downstream algorithms such as bandits or tuners operate on consistent
//! value ranges.
//!
//! ## How It Works
//!
//! The normalizer maintains a fixed-size **rolling window** of the most recent
//! `N` reward values.  Each time you call [`update`], the oldest value is
//! discarded (once the buffer is full), and the new reward is appended.
//!
//! When you call [`normalized`]:
//!
//! 1. Compute the mean μ and standard deviation σ of the stored rewards.  
//! 2. Convert the given reward *r* to a z-score:  
//!    `z = (r − μ) / σ`  
//! 3. Map `z` through a sigmoid to produce a smooth normalized score in (0, 1):  
//!    `norm = 1 / (1 + e^(−z))`  
//!
//! The sigmoid mapping avoids hard clipping and provides graceful saturation
//! for outliers.
//!
//! ## Edge Cases
//! - **Empty buffer:** returns `0.5` as a neutral midpoint.  
//! - **Zero variance:** returns `0.5` to avoid division-by-zero.  
//! - **Window overflow:** oldest element is removed (FIFO behavior).  
//!
//! ## Example
//! ```
//! use rustybrain::reward_normalizer::RewardNormalizer;
//!
//! let mut rn = RewardNormalizer::new(3);
//! rn.update(1.0);
//! rn.update(2.0);
//! rn.update(3.0);
//!
//! let norm = rn.normalized(2.5);
//! assert!(norm > 0.5 && norm < 1.0);
//! ```
//!
//! ## Complexity
//! - **Time:** O(N) per normalization (due to mean/std computation).  
//! - **Space:** O(N) for the rolling buffer.  
//!
//! For larger windows, consider a streaming mean/std algorithm (Welford’s).

/// Dynamically rescales streaming reward values into a stable [0, 1] range.
///
/// See [module-level documentation](index.html) for usage and examples.
#[derive(Debug, Clone)]
pub struct RewardNormalizer {
    /// Number of recent values to retain in the rolling window.
    window: usize,
    /// Stored reward values (oldest first).
    values: Vec<f64>,
}

impl RewardNormalizer {
    /// Creates a new [`RewardNormalizer`] with a rolling window of size `window`.
    ///
    /// # Panics
    /// Panics if `window == 0`.
    pub fn new(window: usize) -> Self {
        assert!(window > 0, "window size must be > 0");
        Self {
            window,
            values: Vec::with_capacity(window),
        }
    }

    /// Inserts a new raw reward into the rolling window.
    ///
    /// If the window is already full, the oldest value is removed (FIFO).
    pub fn update(&mut self, reward: f64) {
        if self.values.len() == self.window {
            self.values.remove(0);
        }
        self.values.push(reward);
    }

    /// Normalizes the provided reward based on the current mean and standard deviation.
    ///
    /// Returns a value in `[0.0, 1.0]` using a sigmoid transformation of the z-score.
    ///
    /// # Behavior
    /// - If no rewards have been recorded → returns `0.5`.
    /// - If all stored rewards are identical → returns `0.5`.
    ///
    /// # Example
    /// ```
    /// let mut rn = rustybrain::reward_normalizer::RewardNormalizer::new(3);
    /// rn.update(10.0);
    /// rn.update(20.0);
    /// let val = rn.normalized(15.0);
    /// assert!((0.0..=1.0).contains(&val));
    /// ```
    pub fn normalized(&self, reward: f64) -> f64 {
        if self.values.is_empty() {
            return 0.5;
        }

        let mean = self.mean();
        let std = self.std(mean);

        // Avoid division by zero when all values are equal.
        if std == 0.0 {
            return 0.5;
        }

        // Convert to z-score and map through a sigmoid into (0, 1)
        let z = (reward - mean) / std;
        1.0 / (1.0 + (-z).exp())
    }

    /// Computes the arithmetic mean of all stored rewards.
    fn mean(&self) -> f64 {
        self.values.iter().copied().sum::<f64>() / self.values.len() as f64
    }

    /// Computes the population standard deviation of stored rewards.
    fn std(&self, mean: f64) -> f64 {
        let var = self
            .values
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>()
            / self.values.len() as f64;
        var.sqrt()
    }
}