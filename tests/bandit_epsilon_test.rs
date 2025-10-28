use rustybrain::bandit::epsilon_greedy::EpsilonGreedy;
use approx::assert_relative_eq;

#[test]
fn test_initialization() {
    let agent = EpsilonGreedy::new(3, 0.1);
    assert_eq!(agent.counts().len(), 3);
    assert_eq!(agent.values().len(), 3);
}

#[test]
fn test_exploitation_when_epsilon_zero() {
    let mut agent = EpsilonGreedy::new(3, 0.0);
    // Manually bias arm 2
    agent.update(2, 10.0);
    agent.update(1, 1.0);
    agent.update(0, 1.0);
    // Should always pick the best arm (index 2)
    for _ in 0..20 {
        let arm = agent.select_arm();
        assert_eq!(arm, 2);
    }
}

#[test]
fn test_exploration_when_epsilon_high() {
    let mut agent = EpsilonGreedy::new(3, 1.0);
    let mut seen = vec![false; 3];
    for _ in 0..100 {
        let arm = agent.select_arm();
        seen[arm] = true;
    }
    assert!(seen.iter().all(|&s| s), "All arms should be explored");
}

#[test]
fn test_reward_update_increments_average() {
    let mut agent = EpsilonGreedy::new(1, 0.1);
    agent.update(0, 1.0);
    agent.update(0, 3.0);
    // avg should be (1 + 3) / 2 = 2.0
    assert_relative_eq!(agent.values()[0], 2.0, epsilon = 1e-12);
    assert_eq!(agent.counts()[0], 2);
}

#[test]
fn test_deterministic_behavior_with_seed() {
    let mut agent1 = EpsilonGreedy::new(3, 0.5);
    let mut agent2 = EpsilonGreedy::new(3, 0.5);
    let sequence1: Vec<_> = (0..10).map(|_| agent1.select_arm()).collect();
    let sequence2: Vec<_> = (0..10).map(|_| agent2.select_arm()).collect();
    assert_eq!(sequence1, sequence2, "Deterministic RNG ensures reproducibility");
}