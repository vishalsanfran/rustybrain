use rustybrain::service::rest_bandit::start_server;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = "127.0.0.1:8080";
    println!("🚀 rustybrain REST server running at http://{addr}");
    start_server(addr).await
}