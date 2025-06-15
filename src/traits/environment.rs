use crate::traits::entities::{Context, Action, Reward};
use ndarray::Dimension;

/// Defines the interface for a simulated environment that interacts with a bandit policy.
/// An environment is responsible for providing context and generating rewards.
pub trait Environment<A, R, C, D>: Send + Sync + 'static
where
    A: Action,
    R: Reward,
    C: Context<D>,
    D: Dimension,
{
    /// Provides the current context (features) to the policy.
    /// For non-contextual bandits, this might return a dummy context.
    fn get_context(&self) -> C;

    /// Generates a reward for a given action taken by the policy in the current context.
    fn get_reward(&self, action: &A, context: &C) -> R;

    /// Returns the optimal reward that could have been obtained in the given context.
    /// This is crucial for calculating regret in simulations.
    fn get_optimal_reward(&self, context: &C) -> R;

}