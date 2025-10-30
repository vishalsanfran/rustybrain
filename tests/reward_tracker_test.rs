use rustybrain::metrics::reward_tracker::RewardTracker;
use approx::assert_relative_eq;

#[test]
fn test_empty_tracker() {
    let rt = RewardTracker::new(3);
    assert_eq!(rt.count(), 0);
    assert_eq!(rt.mean(), 0.0);
    assert_eq!(rt.min(), 0.0);
    assert_eq!(rt.max(), 0.0);
}

#[test]
fn test_basic_stats() {
    let mut rt = RewardTracker::new(3);
    rt.update(1.0);
    rt.update(2.0);
    rt.update(3.0);

    assert_relative_eq!(rt.mean(), 2.0, epsilon = 1e-12);
    assert_relative_eq!(rt.min(), 1.0, epsilon = 1e-12);
    assert_relative_eq!(rt.max(), 3.0, epsilon = 1e-12);
}

#[test]
fn test_window_rolls_over() {
    let mut rt = RewardTracker::new(3);
    for i in 1..=5 {
        rt.update(i as f64);
    }
    // now holds [3.0, 4.0, 5.0]
    assert_relative_eq!(rt.mean(), 4.0, epsilon = 1e-12);
    assert_relative_eq!(rt.min(), 3.0, epsilon = 1e-12);
    assert_relative_eq!(rt.max(), 5.0, epsilon = 1e-12);
    assert_eq!(rt.count(), 3);
}