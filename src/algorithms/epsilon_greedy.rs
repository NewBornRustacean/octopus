use rand::prelude::IndexedRandom;
use rand::rngs::StdRng;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Mutex;

use crate::traits::entities::{Action, ActionStorage, Context, Reward};
use crate::traits::policy::BanditPolicy;
use crate::utils::error::OctopusError;
use rand::{Rng, SeedableRng};

/// Epsilon-Greedy policy for Multi-Armed Bandit problems.
///
/// With probability `epsilon`, selects a random action (exploration).
/// With probability `1 - epsilon`, selects the action with the highest average reward (exploitation).
///
/// Generic over action, reward, and context types. Context is ignored (non-contextual), but required for trait bounds.
#[derive(Debug)]
pub struct EpsilonGreedyPolicy<A, R, C>
where
    C: Context,
    A: Action,
    R: Reward,
{
    epsilon: f64,
    counts: HashMap<u32, u64>,
    sum_rewards: HashMap<u32, f64>,
    action_map: ActionStorage<A>,
    total_pulls: u64,
    rng: Mutex<StdRng>,
    _phantom: PhantomData<(R, C)>,
}

impl<A, R, C> EpsilonGreedyPolicy<A, R, C>
where
    C: Context,
    A: Action,
    R: Reward,
{
    /// Creates a new EpsilonGreedyPolicy.
    ///
    /// * `epsilon` - Probability of exploration (0.0 to 1.0).
    /// * `initial_actions` - Slice of all possible actions.
    ///
    /// Returns an error if `epsilon` is out of bounds or if actions are empty.
    pub fn new(epsilon: f64, initial_actions: &[A]) -> Result<Self, OctopusError> {
        if !(0.0..=1.0).contains(&epsilon) {
            return Err(OctopusError::InvalidParameter {
                parameter_name: "epsilon".to_string(),
                value: epsilon.to_string(),
                expected_range: "0.0 to 1.0 inclusive".to_string(),
            });
        }
        let counts: HashMap<u32, u64> =
            initial_actions.iter().map(|action| (action.id(), 0)).collect();
        let sum_rewards: HashMap<u32, f64> =
            initial_actions.iter().map(|action| (action.id(), 0.0)).collect();
        Ok(EpsilonGreedyPolicy {
            epsilon,
            counts,
            sum_rewards,
            action_map: ActionStorage::new(initial_actions)?,
            total_pulls: 0,
            rng: Mutex::new(StdRng::seed_from_u64((epsilon * 10.0) as u64)),
            _phantom: PhantomData,
        })
    }

    /// Returns the average reward for the given action ID.
    /// Returns 0.0 if the action has not been selected yet.
    fn get_average_reward(&self, action_id: u32) -> f64 {
        let count = *self.counts.get(&action_id).unwrap_or(&0);
        let sum_reward = *self.sum_rewards.get(&action_id).unwrap_or(&0.0);
        if count == 0 {
            0.0
        } else {
            sum_reward / count as f64
        }
    }
}

impl<A, R, C> Clone for EpsilonGreedyPolicy<A, R, C>
where
    C: Context,
    A: Action,
    R: Reward,
    A: Clone,
{
    fn clone(&self) -> Self {
        EpsilonGreedyPolicy {
            epsilon: self.epsilon,
            counts: self.counts.clone(),
            sum_rewards: self.sum_rewards.clone(),
            action_map: self.action_map.clone(),
            total_pulls: self.total_pulls,
            rng: Mutex::new(StdRng::seed_from_u64((self.epsilon * 10.0) as u64)),
            _phantom: PhantomData,
        }
    }
}


impl<A, R, C> BanditPolicy<A, R, C> for EpsilonGreedyPolicy<A, R, C>
where
    C: Context,
    A: Action + 'static,
    R: Reward,
    EpsilonGreedyPolicy<A, R, C>: Clone,
{
    /// Selects an action using the epsilon-greedy strategy.
    /// Ignores context (non-contextual).
    fn choose_action(&self, _context: &C) -> A {
        let mut rng = self.rng.lock().unwrap();
        let random_float: f64 = rng.random_range(0.0..1.0);
        if random_float < self.epsilon {
            // Explore: random action
            let action_ids: Vec<&u32> = self.action_map.keys().collect();
            let rand_id = action_ids.choose(&mut rng).unwrap();
            self.action_map.get(rand_id).unwrap().clone()
        } else {
            // Exploit: action with highest average reward
            let mut best_action_id: u32 = *self.action_map.keys().next().unwrap();
            let mut max_avg_reward: f64 = self.get_average_reward(best_action_id);
            for &action_id in self.action_map.keys() {
                let current_avg = self.get_average_reward(action_id);
                if current_avg > max_avg_reward {
                    max_avg_reward = current_avg;
                    best_action_id = action_id;
                }
            }
            self.action_map.get(&best_action_id).unwrap().clone()
        }
    }

    /// Updates the statistics for the selected action and received reward.
    /// Ignores context (non-contextual).
    fn update(&mut self, _context: &C, action: &A, reward: &R) {
        let action_id = action.id();
        *self.counts.entry(action_id).or_insert(0) += 1;
        *self.sum_rewards.entry(action_id).or_insert(0.0) += reward.value();
        self.total_pulls += 1;
    }

    /// Resets all statistics to their initial state.
    fn reset(&mut self) {
        self.total_pulls = 0;
        for &action_id in self.action_map.keys() {
            *self.counts.get_mut(&action_id).unwrap() = 0;
            *self.sum_rewards.get_mut(&action_id).unwrap() = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::entities::{Action, DummyContext, NumericAction};

    
    #[derive(Debug, Clone, PartialEq)]
    struct DummyReward(f64);

    impl Reward for DummyReward {
        fn value(&self) -> f64 {
            self.0
        }
    }

    #[test]
    fn test_epsilon_greedy_init_success() {
        let actions = vec![
            NumericAction::new(0i32, "Action A"),
            NumericAction::new(10i32, "Action B"),
            NumericAction::new(20i32, "Action C"),
        ];
        let policy =
            EpsilonGreedyPolicy::<NumericAction<i32>, DummyReward, DummyContext>::new(0.1, &actions)
                .unwrap();

        assert_eq!(policy.epsilon, 0.1);
        assert_eq!(policy.action_map.len(), 3);
        assert_eq!(policy.total_pulls, 0);

        for action in actions {
            assert_eq!(*policy.counts.get(&action.id()).unwrap(), 0);
            assert_eq!(*policy.sum_rewards.get(&action.id()).unwrap(), 0.0);
        }
    }

    #[test]
    fn test_epsilon_greedy_init_invalid_epsilon() {
        let actions = vec![NumericAction::new(0i32, "Action A")];

        let error_high =
            EpsilonGreedyPolicy::<NumericAction<i32>, DummyReward, DummyContext>::new(1.5, &actions)
                .unwrap_err();
        assert_eq!(
            error_high,
            OctopusError::InvalidParameter {
                parameter_name: "epsilon".to_string(),
                value: "1.5".to_string(),
                expected_range: "0.0 to 1.0 inclusive".to_string(),
            }
        );

        let error_low =
            EpsilonGreedyPolicy::<NumericAction<i32>, DummyReward, DummyContext>::new(-0.1, &actions)
                .unwrap_err();
        assert_eq!(
            error_low,
            OctopusError::InvalidParameter {
                parameter_name: "epsilon".to_string(),
                value: "-0.1".to_string(),
                expected_range: "0.0 to 1.0 inclusive".to_string(),
            }
        );
    }

    #[test]
    fn test_epsilon_greedy_update_and_average() {
        let actions = vec![
            NumericAction::new(0i32, "Action A"),
            NumericAction::new(10i32, "Action B"),
        ];
        let mut policy =
            EpsilonGreedyPolicy::<NumericAction<i32>, DummyReward, DummyContext>::new(0.0, &actions)
                .unwrap();
        let dummy_context = DummyContext;

        let action_a = NumericAction::new(10i32, "Action A"); 
        let action_b = NumericAction::new(20i32, "Action B"); 

        // Update Action A
        policy.update(&dummy_context, &action_a, &DummyReward(10.0));
        policy.update(&dummy_context, &action_a, &DummyReward(20.0));

        // Update Action B
        policy.update(&dummy_context, &action_b, &DummyReward(5.0));

        assert_eq!(policy.total_pulls, 3);

        // Check Action A's stats
        assert_eq!(*policy.counts.get(&action_a.id()).unwrap(), 2);
        assert_eq!(*policy.sum_rewards.get(&action_a.id()).unwrap(), 30.0);
        assert_eq!(policy.get_average_reward(action_a.id()), 15.0);

        // Check Action B's stats
        assert_eq!(*policy.counts.get(&action_b.id()).unwrap(), 1);
        assert_eq!(*policy.sum_rewards.get(&action_b.id()).unwrap(), 5.0);
        assert_eq!(policy.get_average_reward(action_b.id()), 5.0);
    }

    #[test]
    fn test_epsilon_greedy_exploitation() {
        let actions = vec![
            NumericAction::new(10i32, "Bad Action"),
            NumericAction::new(20i32, "Mediocre Action"),
            NumericAction::new(30i32, "Good Action"),
        ];
        // Epsilon = 0.0 means always exploit
        let mut policy =
            EpsilonGreedyPolicy::<NumericAction<i32>, DummyReward, DummyContext>::new(0.0, &actions)
                .unwrap();
        let dummy_context = DummyContext;

        // Simulate some pulls to establish average rewards
        policy.update(
            &dummy_context,
            actions.get(0).unwrap(),
            &DummyReward(1.0),
        ); // Avg: 1.0
        policy.update(
            &dummy_context,
            actions.get(1).unwrap(),
            &DummyReward(10.0),
        ); // Avg: 10.0
        policy.update(
            &dummy_context,
            actions.get(2).unwrap(),
            &DummyReward(12.0),
        ); // Avg: 12.0
        policy.update(
            &dummy_context,
            actions.get(0).unwrap(),
            &DummyReward(5.0),
        ); // Avg: 3.0
        
        let reward0 = policy.get_average_reward(actions.get(0).unwrap().id());
        let reward1 = policy.get_average_reward(actions.get(1).unwrap().id());
        let reward2 = policy.get_average_reward(actions.get(2).unwrap().id());
        // The "Good Action" should have the highest average reward
        assert_eq!(reward0, 3.0);
        assert_eq!(reward1, 10.0);
        assert_eq!(reward2, 12.0);

        // Policy should consistently choose the "Good Action", since epsilon is zero.
        for _ in 0..100 {
            let chosen_action = policy.choose_action(&dummy_context);
            assert_eq!(
                chosen_action.name(),
                "Good Action"
            );
        }
    }

    #[test]
    fn test_epsilon_greedy_exploration() {
        let actions = vec![
            NumericAction::new(10i32, "Action A"),
            NumericAction::new(10i32, "Action B"),
        ];
        let id0 = actions.get(0).unwrap().id();
        let id1 = actions.get(1).unwrap().id();
        // Epsilon = 1.0 means always explore (random choice)
        let policy =
            EpsilonGreedyPolicy::<NumericAction<i32>, DummyReward, DummyContext>::new(1.0, &actions)
                .unwrap();
        let dummy_context = DummyContext;

        let mut counts_chosen: HashMap<u32, u64> = HashMap::new();
        counts_chosen.insert(id0.clone(), 0);
        counts_chosen.insert(id1.clone(), 0);

        // Perform many choices and check if both actions are chosen roughly equally
        let num_trials = 1000;
        for _ in 0..num_trials {
            let chosen = policy.choose_action(&dummy_context);
            *counts_chosen.get_mut(&chosen.id()).unwrap() += 1;
        }
        println!("Counts: {:?}", policy.counts);
        let chosen_a = *counts_chosen.get(&id0).unwrap();
        let chosen_b = *counts_chosen.get(&id1).unwrap();

        // Check if counts are reasonably close to 50/50 for random exploration
        let expected_per_action = num_trials as f64 / actions.len() as f64;
        let tolerance = expected_per_action * 0.2; // Allow 20% deviation for randomness

        assert!(
            (chosen_a as f64 - expected_per_action).abs() < tolerance,
            "Chosen A: {}",
            chosen_a
        );
        assert!(
            (chosen_b as f64 - expected_per_action).abs() < tolerance,
            "Chosen B: {}",
            chosen_b
        );
        assert_eq!(chosen_a + chosen_b, num_trials as u64);
    }

    #[test]
    fn test_epsilon_greedy_reset() {
        let actions = vec![
            NumericAction::new(10i32, "Action A"),
            NumericAction::new(10i32, "Action B"),
        ];
        
        let id0 = actions.get(0).unwrap().id();
        let id1 = actions.get(1).unwrap().id();
        
        let mut policy =
            EpsilonGreedyPolicy::<NumericAction<i32>, DummyReward, DummyContext>::new(1.0, &actions)
                .unwrap();

        let dummy_context = DummyContext;

        policy.update(
            &dummy_context,
            &actions.get(0).unwrap(),
            &DummyReward(10.0),
        );
        policy.update(
            &dummy_context,
            &actions.get(1).unwrap(),
            &DummyReward(20.0),
        );

        assert_eq!(policy.total_pulls, 2);
        assert_eq!(*policy.counts.get(&id0).unwrap(), 1);
        assert_eq!(*policy.counts.get(&id1).unwrap(), 1);
        assert_eq!(*policy.sum_rewards.get(&id0).unwrap(), 10.0);
        assert_eq!(*policy.sum_rewards.get(&id1).unwrap(), 20.0);

        policy.reset();
        assert_eq!(policy.total_pulls, 0);
        for action_id in policy.action_map.keys() {
            assert_eq!(*policy.counts.get(&action_id).unwrap(), 0);
            assert_eq!(*policy.sum_rewards.get(&action_id).unwrap(), 0.0);
        }
    }
}
