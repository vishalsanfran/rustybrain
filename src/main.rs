use axum::Router;
use rustybrain::service::{bandit_api, optimizer_api, training_api};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let app = Router::new()
        .nest("/bandit", bandit_api::routes())
        .nest("/optimizer", optimizer_api::routes())
        .nest("/train", training_api::routes());

    let listener = tokio::net::TcpListener::bind("127.0.0.1:8080").await?;
    println!("🚀 rustybrain orchestrator running at http://127.0.0.1:8080");
    axum::serve(listener, app).await?;
    Ok(())
}