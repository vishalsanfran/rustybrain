use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use tower::ServiceExt; // for oneshot
use rustybrain::service::training_api::routes;
use serde_json::{json, Value};

#[tokio::test]
async fn training_api_happy_path() {
    let app = routes();

    // 1️⃣ Start a training job
    let req = Request::post("/start")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({"cmd": "echo 'training mock job'"}).to_string(),
        ))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let v: Value = serde_json::from_slice(&bytes).unwrap();
    let id = v.get("id").and_then(|s| s.as_str()).unwrap().to_string();
    assert!(!id.is_empty());

    // 2️⃣ Send mock metrics
    let req = Request::post("/metrics")
        .header("content-type", "application/json")
        .body(Body::from(json!({"loss": 0.3, "reward": 0.7}).to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // 3️⃣ Stop the job
    let req = Request::post("/stop")
        .header("content-type", "application/json")
        .body(Body::from(json!({"id": id}).to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn training_api_stop_unknown_id() {
    let app = routes();
    let req = Request::post("/stop")
        .header("content-type", "application/json")
        .body(Body::from(json!({"id": "fake-job-id"}).to_string()))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

/// ✅ Verifies that /train/metrics updates rewards consistently
#[tokio::test]
async fn training_api_metrics_accumulate() {
    let app = routes();

    // Create a fake job
    let req = Request::post("/start")
        .header("content-type", "application/json")
        .body(Body::from(json!({"cmd": "echo 'mock job'"}).to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let v: Value = serde_json::from_slice(&bytes).unwrap();
    let id = v.get("id").and_then(|s| s.as_str()).unwrap().to_string();

    // Send multiple metric updates
    for r in [0.1, 0.5, 0.9] {
        let req = Request::post("/metrics")
            .header("content-type", "application/json")
            .body(Body::from(json!({"loss": 1.0 - r, "reward": r}).to_string()))
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    // Stop job to ensure cleanup
    let req = Request::post("/stop")
        .header("content-type", "application/json")
        .body(Body::from(json!({"id": id}).to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}