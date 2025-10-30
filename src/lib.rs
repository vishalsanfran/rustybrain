//! rustybrain — a test-driven AI/ML infrastructure toolkit in Rust.
//!
//! The initial module implements a RewardNormalizer utility that
//! stabilizes reward values in online-learning scenarios (e.g. bandits).

pub mod reward_normalizer;
pub mod service {
    pub mod rest_bandit;
}

pub mod metrics {
    pub mod reward_tracker;
}

pub mod bandit {
    pub mod epsilon_greedy;
    pub mod ucb1;
}