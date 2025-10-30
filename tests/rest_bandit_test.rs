use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use tower::ServiceExt; // for `oneshot`
use rustybrain::service::rest_bandit::app;
use serde_json::{json, Value};

#[tokio::test]
async fn rest_bandit_happy_path() {
    let app = app();

    // Create bandit
    let req = Request::post("/bandit")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({"strategy":"epsilon_greedy","param":0.1,"num_arms":3}).to_string(),
        ))
        .unwrap();

    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let v: Value = serde_json::from_slice(&bytes).unwrap();
    let id = v.get("id").and_then(|s| s.as_str()).unwrap().to_string();
    assert!(!id.is_empty());

    // Select arm
    let req = Request::get(format!("/bandit/{}/select", id))
        .body(Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let v: Value = serde_json::from_slice(&bytes).unwrap();
    let arm = v.get("arm").and_then(|n| n.as_u64()).unwrap();
    assert!(arm < 3);

    // Update reward
    let req = Request::post(format!("/bandit/{}/update", id))
        .header("content-type", "application/json")
        .body(Body::from(json!({"arm": arm, "reward": 1.0}).to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn rest_bandit_not_found() {
    let app = app();

    // Unknown id
    let req = Request::get("/bandit/does-not-exist/select")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}