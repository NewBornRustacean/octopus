use crate::common::{Arm, Reward, error::StateError};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::{Arc, RwLock};

/// Thread-safe state management for bandit algorithms
pub struct BanditState<A: Arm, R: Reward> {
    // Count of times each arm has been pulled
    counts: Arc<RwLock<HashMap<A, usize>>>,
    // Total reward for each arm
    rewards: Arc<RwLock<HashMap<A, R>>>,
}

impl<A: Arm + Debug, R: Reward> BanditState<A, R> {
    /// Creates a new empty bandit state
    pub fn new() -> Self {
        Self {
            counts: Arc::new(RwLock::new(HashMap::new())),
            rewards: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Gets the number of times an arm has been pulled
    pub fn get_count(&self, arm: &A) -> Result<usize, StateError> {
        self.counts
            .read()
            .map_err(|e| {
                StateError::ConcurrentStateAccessError(format!("Failed to read counts: {}", e))
            })?
            .get(arm)
            .copied()
            .ok_or_else(|| {
                StateError::ArmNotFoundInState(format!("Arm {:?} not found in state", arm))
            })
    }

    /// Increments the count for an arm
    pub fn increment_count(&self, arm: &A) -> Result<(), StateError> {
        let mut counts = self.counts.write().map_err(|e| {
            StateError::ConcurrentStateAccessError(format!("Failed to write counts: {}", e))
        })?;
        *counts.entry(arm.clone()).or_insert(0) += 1;
        Ok(())
    }

    /// Gets the total reward for an arm
    pub fn get_reward(&self, arm: &A) -> Result<R, StateError> {
        self.rewards
            .read()
            .map_err(|e| {
                StateError::ConcurrentStateAccessError(format!("Failed to read rewards: {}", e))
            })?
            .get(arm)
            .cloned()
            .ok_or_else(|| {
                StateError::ArmNotFoundInState(format!("Arm {:?} not found in state", arm))
            })
    }

    /// Updates the reward for an arm
    pub fn update_reward(&self, arm: &A, reward: R) -> Result<(), StateError> {
        let mut rewards = self.rewards.write().map_err(|e| {
            StateError::ConcurrentStateAccessError(format!("Failed to write rewards: {}", e))
        })?;
        rewards.insert(arm.clone(), reward);
        Ok(())
    }

    /// Gets the average reward for an arm
    pub fn get_average_reward(&self, arm: &A) -> Result<f64, StateError> {
        let count = self.get_count(arm)?;
        if count == 0 {
            return Ok(0.0);
        }

        let reward = self.get_reward(arm)?;
        let value = reward.get_value().map_err(|e| {
            StateError::InvalidRewardUpdate(format!("Failed to get reward value: {}", e))
        })?;

        Ok(value / count as f64)
    }

    /// Resets the state for all arms
    pub fn reset(&self) -> Result<(), StateError> {
        let mut counts = self.counts.write().map_err(|e| {
            StateError::ConcurrentStateAccessError(format!("Failed to write counts: {}", e))
        })?;
        let mut rewards = self.rewards.write().map_err(|e| {
            StateError::ConcurrentStateAccessError(format!("Failed to write rewards: {}", e))
        })?;

        counts.clear();
        rewards.clear();

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{NumericArm, NumericReward};

    #[test]
    fn test_state_creation() {
        let state = BanditState::<NumericArm, NumericReward<f64>>::new();
        assert!(state.counts.read().unwrap().is_empty());
        assert!(state.rewards.read().unwrap().is_empty());
    }

    #[test]
    fn test_arm_count() {
        let state = BanditState::<NumericArm, NumericReward<f64>>::new();
        let arm = NumericArm::new(1);

        // Test initial count
        assert!(state.get_count(&arm).is_err());

        // Test increment
        state.increment_count(&arm).unwrap();
        assert_eq!(state.get_count(&arm).unwrap(), 1);

        // Test multiple increments
        state.increment_count(&arm).unwrap();
        assert_eq!(state.get_count(&arm).unwrap(), 2);
    }

    #[test]
    fn test_reward_update() {
        let state = BanditState::<NumericArm, NumericReward<f64>>::new();
        let arm = NumericArm::new(1);
        let reward = NumericReward::new(1.0).unwrap();

        // Test initial reward
        assert!(state.get_reward(&arm).is_err());

        // Test update
        state.update_reward(&arm, reward.clone()).unwrap();
        assert_eq!(state.get_reward(&arm).unwrap().get_value().unwrap(), 1.0);
    }

    #[test]
    fn test_average_reward() {
        let state = BanditState::<NumericArm, NumericReward<f64>>::new();
        let arm = NumericArm::new(1);
        let reward = NumericReward::new(1.0).unwrap();

        // Test initial average
        assert!(state.get_average_reward(&arm).is_err());

        // Test after update
        state.update_reward(&arm, reward).unwrap();
        state.increment_count(&arm).unwrap();
        assert_eq!(state.get_average_reward(&arm).unwrap(), 1.0);
    }

    #[test]
    fn test_reset() {
        let state = BanditState::<NumericArm, NumericReward<f64>>::new();
        let arm = NumericArm::new(1);
        let reward = NumericReward::new(1.0).unwrap();

        // Set up state
        state.update_reward(&arm, reward).unwrap();
        state.increment_count(&arm).unwrap();

        // Reset
        state.reset().unwrap();

        // Verify reset
        assert!(state.get_count(&arm).is_err());
        assert!(state.get_reward(&arm).is_err());
    }
}
