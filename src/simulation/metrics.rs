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

#[derive(Debug)]
pub struct SummaryStats {
    pub average_cumulative_reward: f64,
    pub average_cumulative_regret: f64,
    pub final_simple_regrets: Vec<f64>,
    pub mean_final_simple_regret: f64,
    pub std_final_simple_regret: f64,
    pub average_step_rewards: Vec<f64>,
    pub average_step_regrets: Vec<f64>,
}

pub fn analyze_results(results: &[SimulationResults]) -> SummaryStats {
    let num_episodes = results.len();
    assert!(num_episodes > 0, "Must have at least one simulation result");

    let num_steps = results[0].steps_rewards.len();
    let mut sum_cumulative_reward = 0.0;
    let mut sum_cumulative_regret = 0.0;
    let mut final_simple_regrets = Vec::with_capacity(num_episodes);

    let mut step_rewards = vec![0.0; num_steps];
    let mut step_regrets = vec![0.0; num_steps];

    for res in results {
        sum_cumulative_reward += res.cumulative_reward;
        sum_cumulative_regret += res.cumulative_optimal_reward - res.cumulative_reward;

        let final_regret = res.final_simple_regret();
        final_simple_regrets.push(final_regret);

        for t in 0..num_steps {
            step_rewards[t] += res.steps_rewards[t];
            step_regrets[t] += res.steps_regret[t];
        }
    }

    let average_cumulative_reward = sum_cumulative_reward / num_episodes as f64;
    let average_cumulative_regret = sum_cumulative_regret / num_episodes as f64;

    for t in 0..num_steps {
        step_rewards[t] /= num_episodes as f64;
        step_regrets[t] /= num_episodes as f64;
    }

    let mean_final_simple_regret = final_simple_regrets.iter().sum::<f64>() / num_episodes as f64;
    let std_final_simple_regret = (final_simple_regrets
        .iter()
        .map(|r| (r - mean_final_simple_regret).powi(2))
        .sum::<f64>()
        / num_episodes as f64)
        .sqrt();

    SummaryStats {
        average_cumulative_reward,
        average_cumulative_regret,
        final_simple_regrets,
        mean_final_simple_regret,
        std_final_simple_regret,
        average_step_rewards: step_rewards,
        average_step_regrets: step_regrets,
    }
}
