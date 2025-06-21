use crate::simulation::metrics::SimulationResults;
use crate::traits::entities::{Action, Context, Reward};
use crate::traits::environment::Environment;
use crate::traits::policy::BanditPolicy;

use ndarray::Dimension;
use std::marker::PhantomData;

/// Simulator for running Multi-Armed Bandit experiments.
///
/// Orchestrates the interaction between a bandit policy and an environment, collecting metrics for analysis.
pub struct Simulator<P, A, R, C, E>
where
    P: BanditPolicy<A, R, C>,
    C: Context,
    A: Action,
    R: Reward,
    E: Environment<A, R, C>,
{
    policy: P,
    environment: E,
    _phantom: PhantomData<(C, A, R)>,
}

impl<P, A, R, C, E> Simulator<P, A, R, C, E>
where
    P: BanditPolicy<A, R, C>,
    C: Context,
    A: Action,
    R: Reward,
    E: Environment<A, R, C>,
{
    /// Creates a new Simulator.
    ///
    /// * `policy` - The bandit policy to evaluate.
    /// * `environment` - The environment providing contexts and rewards.
    pub fn new(policy: P, environment: E) -> Self {
        Simulator {
            policy,
            environment,
            _phantom: PhantomData,
        }
    }

    /// Runs a simulation episode for a given number of steps.
    ///
    /// * `num_steps` - Number of time steps to simulate.
    /// * `all_actions` - Slice of all possible actions (for regret calculation).
    ///
    /// Returns a SimulationResults object with cumulative rewards and regret.
    pub fn run(&mut self, num_steps: usize, all_actions: &[A]) -> SimulationResults {
        let mut cumulative_reward: f64 = 0.0;
        let mut cumulative_optimal_reward: f64 = 0.0;
        let mut steps_rewards: Vec<f64> = Vec::with_capacity(num_steps);
        let mut steps_regret: Vec<f64> = Vec::with_capacity(num_steps);

        for _step in 0..num_steps {
            let current_context = self.environment.get_context();
            let chosen_action = self.policy.choose_action(&current_context);
            let reward = self.environment.get_reward(&chosen_action, &current_context);

            self.policy.update(&current_context, &chosen_action, &reward);
            cumulative_reward += reward.value();

            // Regret calculation: difference between optimal and actual reward.
            let optimal_reward_for_context =
                self.environment.get_optimal_reward(&current_context, all_actions);
            cumulative_optimal_reward += optimal_reward_for_context.value();

            let current_regret = cumulative_optimal_reward - cumulative_reward;

            steps_rewards.push(reward.value());
            steps_regret.push(current_regret);
        }

        SimulationResults::new(
            cumulative_reward,
            cumulative_optimal_reward,
            steps_rewards,
            steps_regret,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::epsilon_greedy::EpsilonGreedyPolicy;
    use crate::traits::entities::DummyContext;
    use ndarray::Array1;

    #[derive(Debug, Clone, PartialEq, Eq, Hash)] // Needs Eq and Hash for HashMap keys
    struct DummyAction {
        id: usize,
        value: i32,
        name: &'static str,
    }

    impl Action for DummyAction {
        type ValueType = i32;
        fn id(&self) -> usize {
            self.id
        }

        fn value(&self) -> i32 {
            self.value.clone()
        }
    }

    #[derive(Debug, Clone, PartialEq)]
    struct DummyReward {
        value: f64,
    }

    impl DummyReward {
        fn new(reward: f64) -> Self {
            Self { value: reward }
        }
    }

    impl Reward for DummyReward {
        fn value(&self) -> f64 {
            self.value.clone()
        }
    }

    #[derive(Debug, Clone)]
    struct DummyEnvironment {
        name: String,
    }

    impl Environment<DummyAction, DummyReward, DummyContext> for DummyEnvironment {
        fn get_context(&self) -> DummyContext {
            DummyContext
        }

        fn get_reward(&self, action: &DummyAction, context: &DummyContext) -> DummyReward {
            let raw = action.value + 100;
            DummyReward::new(raw as f64)
        }
    }
    #[test]
    fn test_run_simulation() {
        let actions = vec![
            DummyAction {
                id: 0,
                value: 10,
                name: "a0",
            },
            DummyAction {
                id: 1,
                value: 20,
                name: "a1",
            },
            DummyAction {
                id: 2,
                value: 30,
                name: "a2",
            },
        ];
        let eps_greedy_policy =
            EpsilonGreedyPolicy::<DummyAction, DummyReward, DummyContext>::new(0.2, &actions)
                .unwrap();
        let dummy_env = DummyEnvironment {
            name: "dummy".to_string(),
        };

        let mut simulator = Simulator::new(eps_greedy_policy, dummy_env);

        let result = simulator.run(10, &actions);
        println!("{:?}", result);
    }
}
