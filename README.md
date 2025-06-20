# 🐙 octopus

## 🎯 **Mission Statement**

Multi-threaded, and ergonomic Rust crate for Multi-Armed Bandit algorithms — with a clear path to contextual bandits and custom reward modeling. Built for engineers, not for academia.

## ✅ **Goals**

1. **Support key MAB algorithms**:

   * MVP: `Epsilon-Greedy`, `UCB1`
   * Future: `Thompson Sampling`, `LinUCB`, etc.

2. **Built-in multi-threading** to offload computational bottlenecks typical in Python/GIL-bound environments.

3. **Clean, extensible abstraction** for stateless and contextual bandits.

4. **User-defined reward types** and optionally, custom arm identifiers (flexible typing via generics or trait bounds).

5. **Ergonomic configuration** via builder pattern.

6. **Target audience**: ML engineers & backend engineers running real-time services in Rust.

## 🚫 **Non-Goals**

* No general-purpose RL framework (e.g., no DQN, A3C, etc.)
* Not a research toolkit — no simulation DSLs or experiment runners.

## 🧱 **Core Abstractions**

```rust
// Trait for stateless bandit algorithms
pub trait Bandit<Arm, Reward> {
    fn select_arm(&self) -> Arm;
    fn update(&mut self, arm: &Arm, reward: Reward);
}

// Trait for contextual bandits
pub trait ContextualBandit<Arm, Context, Reward> {
    fn select_arm(&self, context: &Context) -> Arm;
    fn update(&mut self, context: &Context, arm: &Arm, reward: Reward);
}
```

* `Arm`: generic, must be `Clone + Eq + Hash + Send + Sync`
* `Reward`: customizable (e.g. struct with `ctr`, `revenue`, etc.), must be `Clone + Send + Sync`
* `Context`: flexible type, must be `Clone + Send + Sync`

## 🧪 **Initial Algorithms**

### `epsilon_greedy::EpsilonGreedy`

* Parameters: `epsilon: f64`
* Uses average reward tracking per arm

### `ucb::UCB1`

* Parameters: none (classical version)
* Tracks counts and empirical means

## 🔁 **Concurrency Design**

* Parallelism across arms(not across rounds)
* Internals will use multi-threaded computation with `rayon` or scoped threads.
* Safe internal mutability using `RwLock`/`Mutex`/`Atomic` depending on performance testing.
* Designed so user **does not need to manage concurrency** explicitly.
  

Example:

```rust
let bandit = UCB1::new(num_arms);
let chosen = bandit.select_arm(); // thread-safe
bandit.update(&chosen, reward);   // thread-safe
```

## ⚙️ **Configuration Example (Builder Pattern)**

```rust
let bandit = EpsilonGreedy::builder()
    .epsilon(0.1)
    .arms(vec!["red", "blue", "green"])
    .build();
```

## 📦 **Integration & Ecosystem**

* `serde` optional feature for serialization
* `rayon` for multi-threaded internal ops
* Optional `log` or `tracing` support

## 🧩 **Future Roadmap**

* [ ] Add `ThompsonSampling` with support for conjugate priors
* [ ] Add `LinUCB` for contextual bandits
* [ ] Benchmark suite comparing with Python implementations (on real-world reward logs)
* [ ] Async reward update support (if requested)

## 🧪 **Test Strategy**

* Unit tests for all algorithms (stateless and contextual)
* Integration tests with simulated reward distributions
* Stress tests for multi-threaded update scenarios
