use ndarray::{Array, Array1, Dimension, Ix1};
use std::hash::Hash; // For 1-dimensional feature vectors
// use ndarray::ArrayD; // Keeping this here to remind about future ArrayD expansion

/// Defines the contract for any type that can represent an action (or arm) in a
/// Multi-Armed Bandit problem.
///
/// Implementations must satisfy the following bounds:
/// - `Clone`: Actions need to be copiable, as they are passed by value or cloned
///   when recorded in logs, updated in policies, etc.
/// - `Eq + Hash`: Allows actions to be used as keys in hash maps, which is common
///   for tracking arm-specific statistics within bandit algorithms (e.g., counts,
///   sum of rewards for each arm).
/// - `Send + Sync + 'static`: Essential for multi-threaded environments, ensuring
///   actions can be safely sent between threads or shared across them, and live
///   for the duration of the application.
pub trait Action: Clone + Eq + Hash + Send + Sync + 'static {
    /// Returns a unique identifier for this action.
    ///
    /// This ID is typically used internally by bandit algorithms to map actions
    /// to array indices or to uniquely identify an arm in a collection.
    /// It should be a stable and unique mapping for each distinct action instance.
    fn id(&self) -> usize;

    /// Returns a human-readable string representation of the action.
    /// Useful for logging and debugging.
    fn name(&self) -> String {
        format!("Action-{}", self.id())
    }
}

/// Defines the contract for any type that can represent a reward received
/// after taking an action in a specific context.
///
/// Implementations must satisfy the following bounds:
/// - `Clone`: Rewards need to be copiable, as they are passed by value or cloned
///   when recorded in logs, used in policy updates, etc.
/// - `Send + Sync + 'static`: Essential for multi-threaded environments, ensuring
///   rewards can be safely sent between threads or shared across them, and live
///   for the duration of the application.
pub trait Reward: Clone + Send + Sync + 'static {
    /// Returns the scalar numerical value of the reward.
    ///
    /// For problems where the goal is to minimize a cost (like in the food delivery
    /// scenario), the reward value should typically be the negative of the cost.
    /// This way, maximizing the reward corresponds to minimizing the cost.
    fn value(&self) -> f64;
}

/// Defines the contract for any type that represents the contextual information
/// available to the bandit algorithm when making a decision.
///
/// Implementations must satisfy the following bounds:
/// - `Clone`: Contexts need to be copiable, as they are passed by value or cloned
///   when recorded in logs, used in policy updates, or sent to a policy server.
/// - `Send + Sync + 'static`: Essential for multi-threaded environments, ensuring
///   contexts can be safely sent between threads (e.g., in a `PolicyServer` handling
///   concurrent requests) or shared across them, and live for the duration of the application.
pub trait Context<D: Dimension>: Clone + Send + Sync + 'static {
    /// Converts the context into a 1-dimensional `ndarray::Array1<f64>` of numerical features.
    ///
    /// This is the primary way contextual information is consumed by most
    /// bandit policies (e.g., for linear models, neural networks, or for
    /// simply providing features to algorithms like LinUCB).
    ///
    /// While we initially focus on `Array1`, the underlying context type can conceptually
    /// represent higher-dimensional data which could be flattened here, or a future
    /// extension could introduce a `as_higher_dim_features() -> ArrayD<f64>` method.
    fn to_ndarray(&self) -> Array<f64, D>;
}

#[derive(Debug, Clone, PartialEq)]
pub struct DummyContext;
impl Context<Ix1> for DummyContext {
    fn to_ndarray(&self) -> Array<f64, Ix1> {
        Array1::from_vec(vec![0.0])
    }
}
