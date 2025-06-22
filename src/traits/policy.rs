use crate::traits::entities::{Action, Context, Reward};

/// Core trait for all Multi-Armed Bandit (MAB) algorithms and policies.
///
/// Implementors define how to select actions, update internal state, and reset for new experiments.
/// Generic over action, reward, and context types.
pub trait BanditPolicy<A, R, C>: Clone + Send + Sync + 'static
where
    A: Action,
    R: Reward,
    C: Context,
{
    /// Selects an action based on the current context.
    ///
    /// For non-contextual policies, the context may be ignored.
    fn choose_action(&self, context: &C) -> A;

    /// Updates the policy's internal state based on the observed outcome.
    fn update(&mut self, context: &C, action: &A, reward: &R);

    /// Resets the policy to its initial state (for repeated experiments).
    fn reset(&mut self);

    // Optionally, implementors may add persistence methods.
}
