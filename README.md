# rustybrain
Lightweight, test-driven AI systems toolkit in Rust

rustybrain provides modular, well-tested primitives for adaptive systems — including reward normalization, multi-armed bandits, and parameter optimizers — all exposed through a clean REST API built on Axum.

## Run the REST API
`cargo run`

# REST APIs
## 🎯 Bandit API
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

## ⚙️ Optimizer API
### 1️⃣ Create optimizer
```
curl -X POST http://127.0.0.1:8080/optimizer \
  -H "Content-Type: application/json" \
  -d '{"x0":0.0}'
  ```

### 2️⃣ Get next suggestion
```
curl http://127.0.0.1:8080/optimizer/<id>/suggest
```

### 3️⃣ Submit observed reward
```
curl -X POST http://127.0.0.1:8080/optimizer/<id>/observe \
  -H "Content-Type: application/json" \
  -d '{"reward":8.2}'
```

### 4️⃣ Check optimizer state
```
curl http://127.0.0.1:8080/optimizer/<id>/state
```

## Testing
cargo test
