use rustybrain::reward_normalizer::RewardNormalizer;
use approx::assert_relative_eq;

#[test]
fn test_empty_buffer_returns_neutral() {
    let rn = RewardNormalizer::new(5);
    assert_relative_eq!(rn.normalized(1.0), 0.5, epsilon = 1e-9);
}

#[test]
fn test_constant_rewards_returns_neutral() {
    let mut rn = RewardNormalizer::new(3);
    for _ in 0..3 {
        rn.update(10.0);
    }
    assert_relative_eq!(rn.normalized(10.0), 0.5, epsilon = 1e-9);
}

#[test]
fn test_increasing_rewards() {
    let mut rn = RewardNormalizer::new(5);
    for i in 1..=5 {
        rn.update(i as f64);
    }

    let low = rn.normalized(1.0);
    let mid = rn.normalized(3.0);
    let high = rn.normalized(5.0);

    assert!(low < mid && mid < high);
    assert!(low > 0.0 && high < 1.0);
}

#[test]
fn test_window_sliding_behavior() {
    let mut rn = RewardNormalizer::new(3);
    rn.update(1.0);
    rn.update(2.0);
    rn.update(3.0);
    rn.update(4.0); // pushes out 1.0

    // Behavior check: normalization should increase as reward increases
    let low = rn.normalized(2.0);
    let high = rn.normalized(4.0);
    assert!(high > low);
}