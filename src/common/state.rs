use crate::common::{arm::Arm, error::StateError, reward::Reward, reward::RewardAggregator};
use dashmap::DashMap;
use rayon::prelude::*;

#[derive(Debug)]
pub struct ArmState<A: Arm, RA: RewardAggregator>
where
    A: Send + Sync,
    RA: Send + Sync,
{
    pub arm: A,
    pub reward_aggregator: RA,
    pub n_pulls: usize,
}

impl<A: Arm, RA: RewardAggregator> ArmState<A, RA>
where
    A: Send + Sync,
    RA: Send + Sync,
{
    pub fn new(arm: A, reward_aggregator: RA) -> Self {
        Self {
            arm,
            reward_aggregator,
            n_pulls: 0,
        }
    }

    pub fn update<R: Reward>(&mut self, reward: R) -> Result<(), StateError> {
        let value = reward.get_value()?;
        self.reward_aggregator.update(value)?;
        self.n_pulls += 1;
        Ok(())
    }

    pub fn estimate(&self) -> Result<f64, StateError> {
        self.reward_aggregator.mean().map_err(|e| StateError::RewardError(e))
    }

    pub fn pulls(&self) -> usize {
        self.n_pulls
    }
}

#[derive(Debug)]
pub struct StateStore<A: Arm, RA: RewardAggregator>
where
    A: Send + Sync,
    RA: Send + Sync,
{
    pub states: DashMap<A, ArmState<A, RA>>,
}

impl<A: Arm + std::fmt::Debug, RA: RewardAggregator> StateStore<A, RA>
where
    A: Send + Sync,
    RA: Send + Sync,
{
    pub fn new() -> Self {
        Self {
            states: DashMap::new(),
        }
    }

    pub fn add_arm(&self, arm: A, reward_aggregator: RA) -> Result<(), StateError> {
        if self.states.contains_key(&arm) {
            return Err(StateError::ArmAlreadyExists);
        }
        self.states.insert(arm.clone(), ArmState::new(arm, reward_aggregator));
        Ok(())
    }

    pub fn update<R: Reward>(&self, arm: A, reward: R) -> Result<(), StateError> {
        let mut state = self.states.get_mut(&arm).ok_or(StateError::ArmNotFound)?;
        state.update(reward)
    }

    pub fn estimate(&self, arm: A) -> Result<f64, StateError> {
        let state = self.states.get(&arm).ok_or(StateError::ArmNotFound)?;
        state.estimate()
    }

    pub fn pulls(&self, arm: A) -> Result<usize, StateError> {
        let state = self.states.get(&arm).ok_or(StateError::ArmNotFound)?;
        Ok(state.pulls())
    }

    pub fn total_pulls(&self) -> usize {
        self.states.iter().map(|entry| entry.pulls()).sum()
    }

    pub fn best_arm(&self) -> Result<A, StateError> {
        if self.states.len() == 0 {
            return Err(StateError::NoArmsAvailable);
        }

        // If we have arms, we'll always return one, even if all estimates are the same
        let mut best_arm = None;
        let mut best_estimate = f64::NEG_INFINITY;

        for entry in self.states.iter() {
            // If estimate fails, we treat it as the worst possible estimate
            let estimate = entry.estimate().unwrap_or(f64::NEG_INFINITY);
            if estimate >= best_estimate {
                best_estimate = estimate;
                best_arm = Some(entry.key().clone());
            }
        }

        // We know this is Some because we checked !is_empty() above
        Ok(best_arm.unwrap())
    }

    pub fn print_state(&self) {
        self.states.iter().for_each(|entry| {
            println!(
                "Arm: {:?}, Estimate: {:?}, Pulls: {:?}",
                entry.key(),
                entry.estimate(),
                entry.pulls()
            );
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::arm::{NumericArm, StringArm};
    use crate::common::reward::{BinaryReward, MeanAggregator, NumericReward};

    #[test]
    fn test_add_arm() {
        let store: StateStore<NumericArm, MeanAggregator> = StateStore::new();

        // Add first arm
        let arm1 = NumericArm::new("test1".to_string());
        assert!(store.add_arm(arm1.clone(), MeanAggregator::new()).is_ok());

        // Add second arm
        let arm2 = NumericArm::new("test2".to_string());
        assert!(store.add_arm(arm2.clone(), MeanAggregator::new()).is_ok());

        // Try to add duplicate arm
        assert!(matches!(
            store.add_arm(arm1, MeanAggregator::new()),
            Err(StateError::ArmAlreadyExists)
        ));
    }

    #[test]
    fn test_update_and_estimate() {
        let store: StateStore<NumericArm, MeanAggregator> = StateStore::new();
        let arm = NumericArm::new("test".to_string());

        // Add arm
        store.add_arm(arm.clone(), MeanAggregator::new()).unwrap();

        // Update with numeric rewards
        store.update(arm.clone(), NumericReward::new(10.0).unwrap()).unwrap();
        store.update(arm.clone(), NumericReward::new(20.0).unwrap()).unwrap();

        // Check estimate
        assert_eq!(store.estimate(arm).unwrap(), 15.0); // (10 + 20) / 2
    }

    #[test]
    fn test_update_with_binary_rewards() {
        let store: StateStore<StringArm, MeanAggregator> = StateStore::new();
        let arm = StringArm::new("test".to_string());

        store.add_arm(arm.clone(), MeanAggregator::new()).unwrap();

        // Update with binary rewards (success = 1.0, failure = 0.0)
        store.update(arm.clone(), BinaryReward::success()).unwrap();
        store.update(arm.clone(), BinaryReward::success()).unwrap();
        store.update(arm.clone(), BinaryReward::failure()).unwrap();

        // Check estimate (2 successes, 1 failure = 2/3)
        assert!((store.estimate(arm).unwrap() - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_pulls_counting() {
        let store: StateStore<NumericArm, MeanAggregator> = StateStore::new();
        let arm1 = NumericArm::new("test1".to_string());
        let arm2 = NumericArm::new("test2".to_string());

        store.add_arm(arm1.clone(), MeanAggregator::new()).unwrap();
        store.add_arm(arm2.clone(), MeanAggregator::new()).unwrap();

        // Pull arm1 twice
        store.update(arm1.clone(), NumericReward::new(1.0).unwrap()).unwrap();
        store.update(arm1.clone(), NumericReward::new(2.0).unwrap()).unwrap();

        // Pull arm2 once
        store.update(arm2.clone(), NumericReward::new(3.0).unwrap()).unwrap();

        // Check individual pull counts
        assert_eq!(store.pulls(arm1).unwrap(), 2);
        assert_eq!(store.pulls(arm2).unwrap(), 1);

        // Check total pulls
        assert_eq!(store.total_pulls(), 3);
    }

    #[test]
    fn test_best_arm() {
        let store: StateStore<NumericArm, MeanAggregator> = StateStore::new();
        let arm1 = NumericArm::new("test1".to_string());
        let arm2 = NumericArm::new("test2".to_string());
        let arm3 = NumericArm::new("test3".to_string());

        store.add_arm(arm1.clone(), MeanAggregator::new()).unwrap();
        store.add_arm(arm2.clone(), MeanAggregator::new()).unwrap();
        store.add_arm(arm3.clone(), MeanAggregator::new()).unwrap();

        // Update arms with different rewards
        store.update(arm1.clone(), NumericReward::new(1.0).unwrap()).unwrap();
        store.update(arm2.clone(), NumericReward::new(2.0).unwrap()).unwrap();
        store.update(arm3.clone(), NumericReward::new(3.0).unwrap()).unwrap();

        // arm3 should be the best (highest mean)
        let best = store.best_arm().unwrap();
        assert_eq!(best.id, arm3.id);
        assert_eq!(best.name, arm3.name);
    }

    #[test]
    fn test_error_cases() {
        let store: StateStore<NumericArm, MeanAggregator> = StateStore::new();
        let arm = NumericArm::new("test".to_string());

        // Try to update non-existent arm
        assert!(matches!(
            store.update(arm.clone(), NumericReward::new(1.0).unwrap()),
            Err(StateError::ArmNotFound)
        ));

        // Try to get estimate for non-existent arm
        assert!(matches!(
            store.estimate(arm.clone()),
            Err(StateError::ArmNotFound)
        ));

        // Try to get pulls for non-existent arm
        assert!(matches!(store.pulls(arm), Err(StateError::ArmNotFound)));
    }

    #[test]
    fn test_empty_store() {
        let store: StateStore<NumericArm, MeanAggregator> = StateStore::new();

        // Total pulls should be 0 for empty store
        assert_eq!(store.total_pulls(), 0);

        // raise error if no arms are available for best arm
        assert!(matches!(store.best_arm(), Err(StateError::NoArmsAvailable)));
    }
}
