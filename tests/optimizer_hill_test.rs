use rustybrain::optimizer::Optimizer;
use rustybrain::optimizer::HillClimber1D;

// Simple convex reward surface: maximum at x = 3.0
fn reward_fn(x: f64) -> f64 {
    // -(x - 3)^2 + 10
    -(x - 3.0).powi(2) + 10.0
}

#[test]
fn hill_climber_converges_near_optimum() {
    let mut opt = HillClimber1D::with_params(
        0.0,   // x0
        0.5,   // step
        0.01,  // min_step
        1.1,   // grow
        0.5,   // shrink
    );

    // Iterate a fixed number of steps deterministically.
    for _ in 0..100 {
        let x = opt.suggest();
        let r = reward_fn(x);
        opt.observe(r);
    }

    let x_star = opt.param();
    assert!((x_star - 3.0).abs() < 0.2, "expected ~3.0, got {}", x_star);
}

#[test]
fn hill_climber_is_deterministic() {
    let mut a = HillClimber1D::new(1.0);
    let mut b = HillClimber1D::new(1.0);

    for _ in 0..50 {
        let xa = a.suggest();
        let xb = b.suggest();
        assert!((xa - xb).abs() < 1e-12);

        let ra = reward_fn(xa);
        let rb = reward_fn(xb);
        assert!((ra - rb).abs() < 1e-12);

        a.observe(ra);
        b.observe(rb);
    }

    assert!((a.param() - b.param()).abs() < 1e-12);
}