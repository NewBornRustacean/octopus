use crate::traits::policy::BanditPolicy;
use crate::traits::entities::{Context, Action, Reward};
use crate::simulation::metrics::SimulationResults;
use crate::traits::environment::Environment;

use ndarray::Dimension;
use std::marker::PhantomData;

/// The core simulator for running Multi-Armed Bandit experiments.
/// It orchestrates the interaction between a bandit policy and a simulated environment.
pub struct Simulator<P, C, A, R, D, E>
where
    P: BanditPolicy<C, A, R, D>,
    C: Context<D>,
    A: Action,
    R: Reward,
    D: Dimension,
    E: Environment<A, R, C, D>,
{
    policy: P,
    environment: E,
    _phantom: PhantomData<(C, A, R, D)>,
}

impl<P, C, A, R, D, E> Simulator<P, C, A, R, D, E>
where
    P: BanditPolicy<C, A, R, D>,
    C: Context<D>,
    A: Action,
    R: Reward,
    D: Dimension,
    E: Environment<A, R, C, D>,
{
    /// Creates a new Simulator instance.
    ///
    /// # Arguments
    /// * `policy` - The bandit policy to be evaluated.
    /// * `environment` - The simulated environment providing contexts and rewards.
    pub fn new(policy: P, environment: E) -> Self {
        Simulator {
            policy,
            environment,
            _phantom: PhantomData,
        }
    }

    /// Runs a single simulation episode for a specified number of steps.
    ///
    /// # Arguments
    /// * `num_steps` - The total number of interactions (time steps) to simulate.
    ///
    /// # Returns
    /// A `SimulationResults` object containing cumulative rewards, regret, and other metrics.
    pub fn run(&mut self, num_steps: usize) -> SimulationResults {
        let mut cumulative_reward: f64 = 0.0;
        let mut cumulative_optimal_reward: f64 = 0.0;
        let mut steps_rewards: Vec<f64> = Vec::with_capacity(num_steps);
        let mut steps_regret: Vec<f64> = Vec::with_capacity(num_steps);
        // We could also store chosen actions, contexts, etc., if needed for detailed analysis.

        for step in 0..num_steps {
            let current_context = self.environment.get_context();
            let chosen_action = self.policy.choose_action(&current_context);
            let reward = self.environment.get_reward(&chosen_action, &current_context);

            self.policy.update(&current_context, &chosen_action, &reward);
            cumulative_reward += reward.value();

            // To calculate regret, we need the optimal reward in this context.
            // This assumes the environment can provide it (critical for simulations).
            let optimal_reward_for_context = self.environment.get_optimal_reward(&current_context);
            cumulative_optimal_reward += optimal_reward_for_context.value();

            let current_regret = cumulative_optimal_reward - cumulative_reward;

            steps_rewards.push(reward.value());
            steps_regret.push(current_regret); // Store regret at each step
        }

        SimulationResults::new(
            cumulative_reward,
            cumulative_optimal_reward,
            steps_rewards,
            steps_regret,
        )
    }
}