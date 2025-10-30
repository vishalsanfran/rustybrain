//! Training orchestration API (mini Axolotl controller).
//!
//! Endpoints:
//! - POST /train/start   -> launch training job
//! - POST /train/metrics -> record metrics (loss, reward)
//! - POST /train/stop    -> terminate job

use axum::{extract::State, routing::post, Json, Router};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    process::Stdio,
    sync::{Arc, Mutex},
};
use tokio::{process::Command, task::JoinHandle};
use uuid::Uuid;
use crate::metrics::reward_tracker::RewardTracker;

struct TrainingJob {
    id: String,
    handle: JoinHandle<()>,
    tracker: RewardTracker,
}

#[derive(Clone, Default)]
struct TrainingRegistry {
    jobs: Arc<Mutex<HashMap<String, TrainingJob>>>,
}

#[derive(Deserialize)]
struct StartReq {
    cmd: String, // e.g., "python train.py --epochs 2"
}

#[derive(Serialize)]
struct StartResp {
    id: String,
}

async fn start_job(
    State(reg): State<TrainingRegistry>,
    Json(req): Json<StartReq>,
) -> Json<StartResp> {
    let id = Uuid::new_v4().to_string();
    let tracker = RewardTracker::new(50);

    // Launch subprocess in background
    let handle = tokio::spawn(async move {
        let _ = Command::new("sh")
            .arg("-c")
            .arg(req.cmd)
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .status()
            .await;
    });

    reg.jobs.lock().unwrap().insert(
        id.clone(),
        TrainingJob {
            id: id.clone(),
            handle,
            tracker,
        },
    );

    Json(StartResp { id })
}

#[derive(Deserialize)]
struct MetricsReq {
    reward: f64,
}

async fn update_metrics(
    State(reg): State<TrainingRegistry>,
    Json(req): Json<MetricsReq>,
) {
    for job in reg.jobs.lock().unwrap().values_mut() {
        job.tracker.update(req.reward);
    }
}

#[derive(Deserialize)]
struct StopReq {
    id: String,
}

async fn stop_job(State(reg): State<TrainingRegistry>, Json(req): Json<StopReq>) {
    if let Some(job) = reg.jobs.lock().unwrap().remove(&req.id) {
        job.handle.abort();
        println!("🛑 Training job {} stopped", req.id);
    }
}

pub fn routes() -> Router {
    let reg = TrainingRegistry::default();
    Router::new()
        .route("/start", post(start_job))
        .route("/metrics", post(update_metrics))
        .route("/stop", post(stop_job))
        .with_state(reg)
}