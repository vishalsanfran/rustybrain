//! REST API for Optimizer control (HillClimber1D).
//!
//! Exposes endpoints for creating optimizers, suggesting new parameters,
//! reporting observed rewards, and inspecting current optimizer state.
//!
//! ## Endpoints
//! - POST /optimizer                -> create optimizer
//! - GET  /optimizer/:id/suggest    -> returns { "x": f64 }
//! - POST /optimizer/:id/observe    -> body: { "reward": f64 }
//! - GET  /optimizer/:id/state      -> returns { "x": f64 }

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use axum::{
    extract::{Path, State},
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{
    bandit::{epsilon_greedy::EpsilonGreedy, ucb1::Ucb1},
    metrics::reward_tracker::RewardTracker,
};

#[derive(Clone)]
struct EpsilonGreedyTracked {
    bandit: EpsilonGreedy,
    tracker: RewardTracker,
}

#[derive(Clone)]
enum Strategy {
    EpsilonGreedy(EpsilonGreedyTracked),
    Ucb1(Ucb1),
}

#[derive(Clone, Default)]
struct Registry {
    map: Arc<Mutex<HashMap<String, Strategy>>>,
}

// ===== Request / Response DTOs =====

#[derive(Deserialize)]
struct CreateReq {
    strategy: String, // "epsilon_greedy" or "ucb1"
    param: f64,       // epsilon or c
    num_arms: usize,
}

#[derive(Serialize)]
struct CreateResp {
    id: String,
}

#[derive(Serialize)]
struct SelectResp {
    arm: u32,
}

#[derive(Deserialize)]
struct UpdateReq {
    arm: u32,
    reward: f64,
}

#[derive(Serialize)]
struct StatsResp {
    mean: f64,
    min: f64,
    max: f64,
    count: usize,
}

// ===== Handlers =====

async fn create_bandit(
    State(reg): State<Registry>,
    Json(req): Json<CreateReq>,
) -> Result<Json<CreateResp>, (StatusCode, String)> {
    if !(req.strategy == "epsilon_greedy" || req.strategy == "ucb1") {
        return Err((StatusCode::BAD_REQUEST, "unsupported strategy".into()));
    }
    if req.num_arms == 0 {
        return Err((StatusCode::BAD_REQUEST, "invalid number of arms".into()));
    }

    let id = Uuid::new_v4().to_string();

    let strategy = match req.strategy.as_str() {
        "epsilon_greedy" => {
            if !(0.0..=1.0).contains(&req.param) {
                return Err((StatusCode::BAD_REQUEST, "invalid epsilon".into()));
            }
            let tracked = EpsilonGreedyTracked {
                bandit: EpsilonGreedy::new(req.num_arms, req.param),
                tracker: RewardTracker::new(50),
            };
            Strategy::EpsilonGreedy(tracked)
        }
        "ucb1" => {
            if req.param < 0.0 {
                return Err((StatusCode::BAD_REQUEST, "invalid exploration factor".into()));
            }
            Strategy::Ucb1(Ucb1::new(req.num_arms, req.param))
        }
        _ => unreachable!(),
    };

    reg.map.lock().unwrap().insert(id.clone(), strategy);
    Ok(Json(CreateResp { id }))
}

async fn select_arm(
    State(reg): State<Registry>,
    Path(id): Path<String>,
) -> Result<Json<SelectResp>, (StatusCode, String)> {
    let mut map = reg.map.lock().unwrap();
    let entry = map
        .get_mut(&id)
        .ok_or((StatusCode::NOT_FOUND, "unknown id".into()))?;

    let arm = match entry {
        Strategy::EpsilonGreedy(t) => t.bandit.select_arm() as u32,
        Strategy::Ucb1(b) => b.select_arm() as u32,
    };
    Ok(Json(SelectResp { arm }))
}

async fn update_reward(
    State(reg): State<Registry>,
    Path(id): Path<String>,
    Json(req): Json<UpdateReq>,
) -> Result<(), (StatusCode, String)> {
    let mut map = reg.map.lock().unwrap();
    let entry = map
        .get_mut(&id)
        .ok_or((StatusCode::NOT_FOUND, "unknown id".into()))?;

    match entry {
        Strategy::EpsilonGreedy(t) => {
            t.bandit.update(req.arm as usize, req.reward);
            t.tracker.update(req.reward);
        }
        Strategy::Ucb1(b) => b.update(req.arm as usize, req.reward),
    }
    Ok(())
}

async fn get_stats(
    State(reg): State<Registry>,
    Path(id): Path<String>,
) -> Result<Json<StatsResp>, (StatusCode, String)> {
    let map = reg.map.lock().unwrap();
    let entry = map
        .get(&id)
        .ok_or((StatusCode::NOT_FOUND, "unknown id".into()))?;

    match entry {
        Strategy::EpsilonGreedy(t) => Ok(Json(StatsResp {
            mean: t.tracker.mean(),
            min: t.tracker.min(),
            max: t.tracker.max(),
            count: t.tracker.count(),
        })),
        Strategy::Ucb1(b) => {
            let avg = b.values().iter().sum::<f64>() / b.values().len() as f64;
            Ok(Json(StatsResp {
                mean: avg,
                min: b.values().iter().fold(f64::INFINITY, |a, &x| a.min(x)),
                max: b.values().iter().fold(f64::NEG_INFINITY, |a, &x| a.max(x)),
                count: b.counts().iter().sum::<u64>() as usize,
            }))
        }
    }
}

// ===== Router =====

pub fn routes() -> Router {
    let reg = Registry::default();
    Router::new()
        .route("/", post(create_bandit))
        .route("/:id/select", get(select_arm))
        .route("/:id/update", post(update_reward))
        .route("/:id/stats", get(get_stats))
        .with_state(reg)
}

/// Convenience function to run the API directly.
pub async fn start_server(addr: &str) -> Result<(), Box<dyn std::error::Error>> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    println!("🚀 Bandit API running at http://{addr}/bandit");
    axum::serve(listener, routes()).await?;
    Ok(())
}