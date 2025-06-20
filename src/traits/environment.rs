use crate::traits::entities::{Action, Context, Reward};
use rayon::prelude::*;
/// Defines the interface for a simulated environment that interacts with a bandit policy.
/// An environment is responsible for providing context and generating rewards.
pub trait Environment<A, R, C>: Send + Sync + 'static
where
    A: Action,
    R: Reward,
    C: Context,
{
    /// Provides the current context (features) to the policy.
    /// For non-contextual bandits, this might return a dummy context.
    fn get_context(&self) -> C;

    /// Generates a reward for a given action taken by the policy in the current context.
    fn get_reward(&self, action: &A, context: &C) -> R;

    /// Returns the optimal reward that could have been obtained in the given context.
    /// This is crucial for calculating regret in simulations.
    fn get_optimal_reward(&self, context: &C, actions: &[A]) -> R {
        actions
            .par_iter()
            .map(|a| self.get_reward(a, context))
            .max_by(|r1, r2| r1.value().partial_cmp(&r2.value()).unwrap())
            .expect("No actions provided")
    }
}
