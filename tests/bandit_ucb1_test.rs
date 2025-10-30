use rustybrain::bandit::ucb1::Ucb1;
use approx::assert_relative_eq;

#[test]
fn test_initial_selection_cycles_through_arms() {
    let mut agent = Ucb1::new(3, 2.0);
    // Each time we select an arm, mark it as pulled.
    let mut seen = vec![false; 3];
    for _ in 0..3 {
        let arm = agent.select_arm();
        seen[arm] = true;
        agent.update(arm, 1.0);
    }
    // After 3 updates, all arms should have been explored once.
    assert!(seen.iter().all(|&v| v), "all arms should be tried once");
}

#[test]
fn test_update_and_exploitation() {
    let mut agent = Ucb1::new(2, 2.0);
    agent.update(0, 1.0);
    agent.update(0, 1.0);
    agent.update(1, 0.1);
    // Arm 0 has higher mean; should be selected next.
    let arm = agent.select_arm();
    assert_eq!(arm, 0);
}

#[test]
fn test_deterministic_behavior() {
    let mut a1 = Ucb1::new(3, 2.0);
    let mut a2 = Ucb1::new(3, 2.0);

    for _ in 0..10 {
        let arm1 = a1.select_arm();
        let arm2 = a2.select_arm();
        assert_eq!(arm1, arm2);
        a1.update(arm1, 1.0);
        a2.update(arm2, 1.0);
    }
}

#[test]
fn test_average_updates_correctly() {
    let mut agent = Ucb1::new(1, 2.0);
    agent.update(0, 1.0);
    agent.update(0, 3.0);
    assert_relative_eq!(agent.values()[0], 2.0, epsilon = 1e-12);
    assert_eq!(agent.counts()[0], 2);
}