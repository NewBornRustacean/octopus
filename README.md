# 🐙 octopus

## 🎯 **Mission Statement**

A generic, multi-threaded, and ergonomic Rust crate for Multi-Armed Bandit (MAB) algorithms—designed for engineers who need extensibility, custom reward modeling, and a clear path to contextual bandits.

## ✅ **Goals**

1. **Support key MAB algorithms**:
   * MVP: `Epsilon-Greedy`, `UCB1`, `Thompson Sampling`
   * Future: `LinUCB`, etc.
2. **Built-in multi-threading** for computational bottlenecks (e.g., regret calculation, reward evaluation).
3. **Clean, extensible abstraction** for stateless and contextual bandits, using Rust traits and generics.
4. **User-defined reward types** and flexible action/context typing via trait bounds.
5. **Target audience**: ML engineers & backend engineers running real-time services in Rust.

## 🚫 **Non-Goals**

* Not a general-purpose RL framework (e.g., no DQN, A3C, etc.)
* Not a rich visualization tool

## 🧱 **Core Abstractions**

```rust
// Action: Represents an arm/action in the bandit problem
pub trait Action: Clone + Eq + Hash + Send + Sync + 'static {
    type ValueType;
    fn id(&self) -> usize;
    fn name(&self) -> String { ... }
    fn value(&self) -> Self::ValueType;
}

// Reward: Represents the reward signal
pub trait Reward: Clone + Send + Sync + 'static {
    fn value(&self) -> f64;
}

// Context: Represents contextual information (for contextual bandits)
pub trait Context: Clone + Send + Sync + 'static {
    type DimType: ndarray::Dimension;
    fn to_ndarray(&self) -> ndarray::Array<f64, Self::DimType>;
}

// BanditPolicy: Core trait for all bandit algorithms
pub trait BanditPolicy<A, R, C>: Send + Sync + 'static
where
    A: Action,
    R: Reward,
    C: Context,
{
    fn choose_action(&self, context: &C) -> A;
    fn update(&mut self, context: &C, action: &A, reward: &R);
    fn reset(&mut self);
}

// Environment: Simulated environment for running experiments
pub trait Environment<A, R, C>: Send + Sync + 'static { ... }
```

* **Extensibility**: Implement these traits for your own types to plug into the framework.
* **Action/Reward/Context**: Highly generic, must satisfy trait bounds above.

## 🧪 **Implemented Algorithms**

### `epsilon_greedy::EpsilonGreedyPolicy`

* Parameters: `epsilon: f64`, initial actions
* Tracks average reward and count per action
* Generic over action, reward, and context types

## 🏗️ **Simulation Engine**

* The `Simulator` struct orchestrates the interaction between a bandit policy and an environment.
* Collects cumulative rewards, regret, and per-step metrics via `SimulationResults`.

**Example:**

```rust
use octopus::algorithms::epsilon_greedy::EpsilonGreedyPolicy;
use octopus::simulation::simulator::Simulator;
use octopus::traits::entities::{Action, Reward, Context, DummyContext};

// Define your own Action, Reward, and Environment types implementing the required traits
// ...

let actions = vec![/* your actions here */];
let mut policy = EpsilonGreedyPolicy::new(0.1, &actions).unwrap();
let environment = /* your environment here */;
let mut simulator = Simulator::new(policy, environment);
let results = simulator.run(1000, &actions);
println!("Cumulative reward: {}", results.cumulative_reward);
```

## 🔁 **Concurrency Design**

* Internal parallelism (e.g., for regret calculation) uses `rayon` and thread-safe primitives (`Mutex`).
* User-facing API is single-threaded for simplicity; internal operations are parallelized where beneficial.
* No explicit builder pattern or thread-safe wrappers in the current API.

## 📦 **Integration & Ecosystem**

* Uses `ndarray` for context features
* Uses `rayon` for parallelism
* Error handling via `thiserror`
* (Planned) Optional `serde` for serialization

## 🧩 **Future Roadmap**

* [ ] Add more algorithms: `UCB1`, `LinUCB` etc.
* [ ] Benchmark suite comparing with Python implementations
* [ ] Async/streaming reward update support
* [ ] Optional logging/tracing integration

## 🧪 **Test Strategy**

* Unit tests for all algorithms and simulation logic
* Integration tests with simulated reward distributions
* Stress tests for multi-threaded scenarios
