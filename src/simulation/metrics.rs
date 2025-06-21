/// Stores the results of a single bandit simulation episode.
#[derive(Debug, Clone, PartialEq)] // Derive common traits for convenience
pub struct SimulationResults {
    /// Total reward accumulated by the policy.
    pub cumulative_reward: f64,
    /// Total reward that would have been obtained by always choosing the optimal action.
    pub cumulative_optimal_reward: f64,
    /// Reward received at each step.
    pub steps_rewards: Vec<f64>,
    /// Cumulative regret at each step.
    pub steps_regret: Vec<f64>,
}

impl SimulationResults {
    /// Creates a new SimulationResults instance.
    pub fn new(
        cumulative_reward: f64,
        cumulative_optimal_reward: f64,
        steps_rewards: Vec<f64>,
        steps_regret: Vec<f64>,
    ) -> Self {
        SimulationResults {
            cumulative_reward,
            cumulative_optimal_reward,
            steps_rewards,
            steps_regret,
        }
    }

    /// Returns the final simple regret (difference from optimal at the last step).
    pub fn final_simple_regret(&self) -> f64 {
        self.cumulative_optimal_reward - self.cumulative_reward
    }
}
