use crate::traits::entities::{Action, Context, Reward};
use ndarray::Dimension;

/// Defines the contract for any Multi-Armed Bandit (MAB) algorithm or policy.
///
/// This trait outlines the core functionalities: choosing an action based on context,
/// and updating the policy's internal state based on observed rewards.
///
/// Type Parameters:
/// - `C`: The type representing the contextual information. Must implement `Context<D>`.
/// - `A`: The type representing an action (arm). Must implement `Action`.
/// - `R`: The type representing the reward received. Must implement `Reward`.
/// - `D`: The `ndarray` Dimension type of the features produced by the `Context`.
///
/// Implementations must satisfy the following bounds for thread safety and longevity:
/// - `Send + Sync + 'static`: Essential for multi-threaded environments, allowing policy
///   instances to be safely sent between threads (e.g., in a `PolicyServer` handling
///   concurrent requests) or shared across them.
pub trait BanditPolicy<C, A, R, D>: Send + Sync + 'static
where
    C: Context<D>,
    A: Action,
    R: Reward,
    D: Dimension,
{
    /// Chooses an action based on the current context.
    ///
    /// This method embodies the "exploration-exploitation" dilemma. The policy uses
    /// its internal knowledge and the provided `context` to decide which action (arm)
    /// to pull.
    ///
    /// For non-contextual policies, the `context` might be ignored or expected
    /// to be a trivial type (e.g., `()`).
    ///
    /// # Arguments
    /// * `context` - A reference to the current contextual information (`C`).
    ///
    /// # Returns
    /// The chosen action (`A`).
    fn choose_action(&self, context: &C) -> A;

    /// Updates the policy's internal state based on the observed outcome of an action.
    ///
    /// This method is where the learning happens. The policy uses the feedback
    /// (`context`, `action`, `reward`) to refine its understanding of the environment
    /// and improve future decisions.
    ///
    /// # Arguments
    /// * `context` - A reference to the context in which the action was chosen (`C`).
    /// * `action` - A reference to the action that was taken (`A`).
    /// * `reward` - A reference to the reward received for taking that action (`R`).
    fn update(&mut self, context: &C, action: &A, reward: &R);

    /// Resets the policy's internal state to its initial configuration.
    ///
    /// This is useful for running multiple independent simulations or experiments
    /// with the same policy instance without carrying over state from previous runs.
    fn reset(&mut self);

    // Optional: Add methods for saving/loading policy state if persistence is a requirement.
    // fn save_state(&self) -> Vec<u8>;
    // fn load_state(&mut self, state_data: &[u8]) -> Result<(), PolicyError>;
}
