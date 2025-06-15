/// Stores the results of a single bandit simulation episode.
#[derive(Debug, Clone, PartialEq)] // Derive common traits for convenience
pub struct SimulationResults {
    /// The total sum of rewards accumulated by the policy over the simulation.
    pub cumulative_reward: f64,
    /// The total sum of rewards that would have been obtained if the optimal arm
    /// was chosen at every step (only available in simulations).
    pub cumulative_optimal_reward: f64,
    /// A vector of rewards obtained at each step of the simulation.
    pub steps_rewards: Vec<f64>,
    /// A vector of cumulative regret at each step of the simulation.
    pub steps_regret: Vec<f64>,
}

impl SimulationResults {
    /// Creates a new `SimulationResults` instance.
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

    /// Calculates the final simple regret (difference from optimal at the last step).
    pub fn final_simple_regret(&self) -> f64 {
        self.cumulative_optimal_reward - self.cumulative_reward
    }
}
