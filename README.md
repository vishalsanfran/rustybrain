# rustybrain
Lightweight, test-driven AI systems toolkit in Rust

rustybrain provides modular, well-tested primitives for adaptive systems — from reward normalization to multi-armed bandit algorithms — and exposes them through a simple REST API built on Axum.



## Run the REST API
`cargo run`

## REST APIs
### 1️⃣ Create a new ε-greedy bandit
curl -X POST http://127.0.0.1:8080/bandit \
  -H "Content-Type: application/json" \
  -d '{"strategy":"epsilon_greedy","param":0.1,"num_arms":3}'

### 2️⃣ Select an arm
curl http://127.0.0.1:8080/bandit/<id>/select

### 3️⃣ Update reward
curl -X POST http://127.0.0.1:8080/bandit/<id>/update \
  -H "Content-Type: application/json" \
  -d '{"arm":1,"reward":0.9}'

### 4️⃣ Get rolling reward stats
curl http://127.0.0.1:8080/bandit/<id>/stats

## Testing
cargo test
