use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Beta, Distribution};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Mutex;

use crate::traits::entities::{Action, ActionStorage, Context, Reward};
use crate::traits::policy::BanditPolicy;
use crate::utils::error::OctopusError;

/// Thompson Sampling policy for Multi-Armed Bandit problems.
#[derive(Debug)]
pub struct ThompsonSamplingPolicy<A, R, C>
where
    C: Context,
    A: Action,
    R: Reward,
{
    alpha_params: HashMap<u32, f64>,
    beta_params: HashMap<u32, f64>,
    action_map: ActionStorage<A>,
    rng: Mutex<StdRng>,
    _phantom: PhantomData<(R, C)>,
}

impl<A, R, C> ThompsonSamplingPolicy<A, R, C>
where
    C: Context,
    A: Action,
    R: Reward,
{
    /// Create new ThompsonSamplingPolicy with seeded RNG
    pub fn new(initial_actions: &[A], seed: u64) -> Result<Self, OctopusError> {
        if initial_actions.is_empty() {
            return Err(OctopusError::InvalidParameter {
                parameter_name: "initial_actions".to_string(),
                value: "empty".to_string(),
                expected_range: "non-empty slice of actions".to_string(),
            });
        }

        let alpha_params: HashMap<u32, f64> =
            initial_actions.iter().map(|action| (action.id(), 1.0)).collect();
        let beta_params: HashMap<u32, f64> =
            initial_actions.iter().map(|action| (action.id(), 1.0)).collect();

        // Expand u64 seed to [u8; 32]
        let mut seed_bytes = [0u8; 32];
        seed_bytes[..8].copy_from_slice(&seed.to_le_bytes());
        let rng = StdRng::from_seed(seed_bytes);

        Ok(ThompsonSamplingPolicy {
            alpha_params,
            beta_params,
            action_map: ActionStorage::new(initial_actions)?,
            rng: Mutex::new(rng),
            _phantom: PhantomData,
        })
    }
}

impl<A, R, C> Clone for ThompsonSamplingPolicy<A, R, C>
where
    C: Context,
    A: Action + Clone,
    R: Reward,
{
    fn clone(&self) -> Self {
        // Use a new seed or replicate seed as needed
        let mut seed_bytes = [0u8; 32];
        let seed = rand::random::<u64>();
        seed_bytes[..8].copy_from_slice(&seed.to_le_bytes());

        ThompsonSamplingPolicy {
            alpha_params: self.alpha_params.clone(),
            beta_params: self.beta_params.clone(),
            action_map: self.action_map.clone(),
            rng: Mutex::new(StdRng::from_seed(seed_bytes)),
            _phantom: PhantomData,
        }
    }
}

impl<A, R, C> BanditPolicy<A, R, C> for ThompsonSamplingPolicy<A, R, C>
where
    C: Context,
    A: Action + 'static,
    R: Reward,
    ThompsonSamplingPolicy<A, R, C>: Clone,
{
    fn choose_action(&self, _context: &C) -> A {
        let mut rng = self.rng.lock().unwrap();
        let mut best_action_id = *self.action_map.keys().next().unwrap();
        let mut max_sampled_reward = -1.0;

        for &action_id in self.action_map.keys() {
            let alpha = *self.alpha_params.get(&action_id).unwrap_or(&1.0);
            let beta = *self.beta_params.get(&action_id).unwrap_or(&1.0);

            if alpha <= 0.0 || beta <= 0.0 {
                panic!("Invalid Beta parameters: alpha = {}, beta = {}", alpha, beta);
            }

            let beta_dist = Beta::new(alpha, beta)
                .expect("Beta distribution parameters must be positive.");
            let sampled_reward = beta_dist.sample(&mut *rng);

            if sampled_reward > max_sampled_reward {
                max_sampled_reward = sampled_reward;
                best_action_id = action_id;
            }
        }

        self.action_map.get(&best_action_id).unwrap().clone()
    }

    fn update(&mut self, _context: &C, action: &A, reward: &R) {
        let action_id = action.id();
        let reward_value = reward.value();

        if reward_value >= 0.5 {
            *self.alpha_params.entry(action_id).or_insert(1.0) += 1.0;
        } else {
            *self.beta_params.entry(action_id).or_insert(1.0) += 1.0;
        }
    }

    fn reset(&mut self) {
        for &action_id in self.action_map.keys() {
            *self.alpha_params.get_mut(&action_id).unwrap() = 1.0;
            *self.beta_params.get_mut(&action_id).unwrap() = 1.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::entities::{DummyContext, NumericAction};
    use crate::utils::error::OctopusError;

    #[derive(Debug, Clone, PartialEq)]
    struct DummyReward(f64);

    impl Reward for DummyReward {
        fn value(&self) -> f64 {
            self.0
        }
    }

    #[test]
    fn test_thompson_init_success() {
        let actions = vec![
            NumericAction::new(10i32, "A"),
            NumericAction::new(20i32, "B"),
        ];
        let policy = ThompsonSamplingPolicy::<NumericAction<i32>, DummyReward, DummyContext>::new(&actions, 42)
            .unwrap();
        assert_eq!(policy.alpha_params.len(), 2);
        assert_eq!(policy.beta_params.len(), 2);
        for a in actions {
            assert_eq!(*policy.alpha_params.get(&a.id()).unwrap(), 1.0);
            assert_eq!(*policy.beta_params.get(&a.id()).unwrap(), 1.0);
        }
    }

    #[test]
    fn test_thompson_init_empty_error() {
        let actions: Vec<NumericAction<i32>> = vec![];
        let err = ThompsonSamplingPolicy::<NumericAction<i32>, DummyReward, DummyContext>::new(&actions, 42).unwrap_err();
        assert_eq!(
            err,
            OctopusError::InvalidParameter {
                parameter_name: "initial_actions".to_string(),
                value: "empty".to_string(),
                expected_range: "non-empty slice of actions".to_string()
            }
        );
    }

    #[test]
    fn test_thompson_choose_action_does_not_panic() {
        let actions = vec![
            NumericAction::new(10i32, "A"),
            NumericAction::new(20i32, "B"),
        ];
        let policy = ThompsonSamplingPolicy::<NumericAction<i32>, DummyReward, DummyContext>::new(&actions, 12345)
            .unwrap();
        let ctx = DummyContext;
        let action = policy.choose_action(&ctx);
        assert!(actions.contains(&action));
    }

    #[test]
    fn test_thompson_update_modifies_params() {
        let actions = vec![
            NumericAction::new(10i32, "A"),
            NumericAction::new(20i32, "B"),
        ];
        let id0 = actions.get(0).unwrap().id();
        
        let mut policy = ThompsonSamplingPolicy::<NumericAction<i32>, DummyReward, DummyContext>::new(&actions, 777)
            .unwrap();
        let ctx = DummyContext;

        let a = actions.get(0).unwrap(); 

        // Simulate a reward of 1.0 (success)
        policy.update(&ctx, a, &DummyReward(1.0));
        assert_eq!(*policy.alpha_params.get(&id0).unwrap(), 2.0);
        assert_eq!(*policy.beta_params.get(&id0).unwrap(), 1.0);

        // Simulate a reward of 0.0 (failure)
        policy.update(&ctx, a, &DummyReward(0.0));
        assert_eq!(*policy.alpha_params.get(&id0).unwrap(), 2.0);
        assert_eq!(*policy.beta_params.get(&id0).unwrap(), 2.0);
    }

    #[test]
    fn test_thompson_reset() {
        let actions = vec![
            NumericAction::new(10i32, "A"),
            NumericAction::new(20i32, "B"),
        ];
        
        let id0 = actions.get(0).unwrap().id();
        
        let mut policy = ThompsonSamplingPolicy::<NumericAction<i32>, DummyReward, DummyContext>::new(&actions, 42)
            .unwrap();
        let ctx = DummyContext;
        let a = actions.get(0).unwrap();

        policy.update(&ctx, &a, &DummyReward(1.0));
        policy.update(&ctx, &a, &DummyReward(0.0));
        assert_ne!(*policy.alpha_params.get(&id0).unwrap(), 1.0);
        assert_ne!(*policy.beta_params.get(&id0).unwrap(), 1.0);

        policy.reset();
        for id in policy.action_map.keys() {
            assert_eq!(*policy.alpha_params.get(&id).unwrap(), 1.0);
            assert_eq!(*policy.beta_params.get(&id).unwrap(), 1.0);
        }
    }

    #[test]
    fn test_thompson_sampling_is_reproducible() {
        let actions = vec![
            NumericAction::new(10i32, "A"),
            NumericAction::new(20i32, "B"),
        ];
        let ctx = DummyContext;

        let policy1 = ThompsonSamplingPolicy::<NumericAction<i32>, DummyReward, DummyContext>::new(&actions, 1234)
            .unwrap();
        let policy2 = ThompsonSamplingPolicy::<NumericAction<i32>, DummyReward, DummyContext>::new(&actions, 1234)
            .unwrap();

        let chosen1 = policy1.choose_action(&ctx);
        let chosen2 = policy2.choose_action(&ctx);

        assert_eq!(chosen1, chosen2, "Same seed should produce same result");
    }
}
