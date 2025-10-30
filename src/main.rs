use rustybrain::service::{bandit_api, optimizer_api};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let app = axum::Router::new()
        .nest("/bandit", bandit_api::routes())
        .nest("/optimizer", optimizer_api::routes());

    let listener = tokio::net::TcpListener::bind("127.0.0.1:8080").await?;
    println!("🚀 rustybrain API running at http://127.0.0.1:8080");
    axum::serve(listener, app).await?;
    Ok(())
}