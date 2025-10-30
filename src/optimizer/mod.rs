//! Optimizer primitives for simple ML-style parameter tuning.
//!
//! This module provides a deterministic, 1-D hill-climbing optimizer that
//! improves a single parameter `x` based on observed rewards. It’s designed
//! for tight unit tests and future expansion (e.g., multi-D, annealing).

/// A minimal interface for iterative optimization of a single parameter.
pub trait Optimizer {
    /// Propose the next parameter value to evaluate.
    fn suggest(&mut self) -> f64;

    /// Report the reward obtained from evaluating the most recent suggestion.
    fn observe(&mut self, reward: f64);

    /// Current best-known parameter.
    fn param(&self) -> f64;

    /// Reset the optimizer to a fresh starting value.
    fn reset(&mut self, x0: f64);
}

/// Deterministic 1-D hill climber with step adaptation.
///
/// Strategy:
/// - Start at x0, keep a direction (+1/-1) and step size.
/// - If reward improves, move further in same direction and grow step slightly.
/// - If reward worsens, reverse direction and shrink step.
/// - Stops shrinking below `min_step` but remains deterministic.
#[derive(Debug, Clone)]
pub struct HillClimber1D {
    x: f64,
    dir: f64,
    step: f64,
    min_step: f64,
    grow: f64,
    shrink: f64,
    last_reward: Option<f64>,
    last_suggested: Option<f64>,
}

impl HillClimber1D {
    pub fn new(x0: f64) -> Self {
        Self::with_params(x0, 0.5, 0.1, 1.1, 0.5)
    }

    /// Create with full control over parameters.
    /// - `step`: initial step size
    /// - `min_step`: minimum step size threshold
    /// - `grow`: factor to grow step on improvement (>1.0)
    /// - `shrink`: factor to shrink step on failure (0..1)
    pub fn with_params(x0: f64, step: f64, min_step: f64, grow: f64, shrink: f64) -> Self {
        assert!(step > 0.0 && min_step > 0.0);
        assert!(grow > 1.0 && (0.0..1.0).contains(&shrink));
        Self {
            x: x0,
            dir: 1.0,
            step,
            min_step,
            grow,
            shrink,
            last_reward: None,
            last_suggested: None,
        }
    }
}

impl Optimizer for HillClimber1D {
    fn suggest(&mut self) -> f64 {
        // First suggestion is the current x.
        let s = match self.last_suggested {
            None => self.x,
            Some(_) => self.x + self.dir * self.step,
        };
        self.last_suggested = Some(s);
        s
    }

    fn observe(&mut self, reward: f64) {
        match (self.last_reward, self.last_suggested) {
            (None, Some(s)) => {
                // First observation initializes state; adopt s as current.
                self.x = s;
                self.last_reward = Some(reward);
            }
            (Some(prev_r), Some(s)) => {
                if reward >= prev_r {
                    // Improvement: keep moving in same direction; grow step
                    self.x = s;
                    self.step *= self.grow;
                    self.last_reward = Some(reward);
                } else {
                    // No improvement: reverse direction; shrink step
                    self.dir = -self.dir;
                    self.step = (self.step * self.shrink).max(self.min_step);
                    // keep x at previous best; last_reward unchanged
                }
            }
            _ => {} // Should not happen; safe no-op.
        }
        // Ready for next suggest()
    }

    fn param(&self) -> f64 {
        self.x
    }

    fn reset(&mut self, x0: f64) {
        self.x = x0;
        self.dir = 1.0;
        self.step = self.step.max(self.min_step);
        self.last_reward = None;
        self.last_suggested = None;
    }
}