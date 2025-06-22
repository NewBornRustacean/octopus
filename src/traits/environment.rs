use crate::traits::entities::{Action, Context, Reward};

/// Defines the interface for an environment that interacts with a bandit policy.
///
/// An environment provides context and generates rewards, either for simulation or real-world feedback.
pub trait Environment<A, R, C>: Clone + Send + Sync + 'static
where
    A: Action,
    R: Reward,
    C: Context,
{
    /// Returns the current context (features) for the policy.
    /// For non-contextual bandits, this may return a dummy context.
    fn get_context(&self) -> C;

    /// Generates a reward for a given action taken in the provided context.
    fn get_reward(&self, action: &A, context: &C) -> R;

    /// Returns the optimal reward that could be obtained in the given context from the provided actions.
    /// Used for regret calculation in simulation.
    fn get_optimal_reward(&self, context: &C, actions: &[A]) -> R {
        actions
            .iter()
            .map(|a| self.get_reward(a, context))
            .max_by(|r1, r2| r1.value().partial_cmp(&r2.value()).unwrap())
            .expect("No actions provided")
    }
}
