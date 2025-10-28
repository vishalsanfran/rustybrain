//! RewardNormalizer — dynamically rescales streaming rewards
//! into a stable [0,1] range using a rolling mean and std deviation.

#[derive(Debug, Clone)]
pub struct RewardNormalizer {
    window: usize,
    values: Vec<f64>,
}

impl RewardNormalizer {
    /// Create a new RewardNormalizer with a rolling window of N values.
    pub fn new(window: usize) -> Self {
        assert!(window > 0, "window size must be > 0");
        Self {
            window,
            values: Vec::with_capacity(window),
        }
    }

    /// Update the normalizer with a new raw reward.
    pub fn update(&mut self, reward: f64) {
        if self.values.len() == self.window {
            self.values.remove(0);
        }
        self.values.push(reward);
    }

    /// Normalize an input reward using the current mean and std deviation.
    /// Returns 0.5 when no values are recorded (neutral baseline).
    pub fn normalized(&self, reward: f64) -> f64 {
        if self.values.is_empty() {
            return 0.5;
        }

        let mean = self.mean();
        let std = self.std(mean);
        if std == 0.0 {
            return 0.5;
        }

        // standard score → sigmoid mapping into (0,1)
        let z = (reward - mean) / std;
        1.0 / (1.0 + (-z).exp())
    }

    fn mean(&self) -> f64 {
        self.values.iter().copied().sum::<f64>() / self.values.len() as f64
    }

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