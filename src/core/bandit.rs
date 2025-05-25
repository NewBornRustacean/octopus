use crate::common::{Arm, Reward, state::BanditState};
use crate::core::error::BanditError;
use std::fmt::Debug;

/// Core trait for multi-armed bandit algorithms.
///
/// This trait defines the essential operations that any bandit algorithm must implement:
/// - Selecting an arm to pull
/// - Updating the state with observed rewards
/// - Accessing and managing the algorithm's state
/// - Resetting the algorithm to its initial state
pub trait Bandit<A: Arm, R: Reward> {
    /// Select an arm to pull based on the current state.
    ///
    /// # Returns
    /// - `Ok(A)`: The selected arm
    /// - `Err(BanditError)`: If arm selection fails
    fn select_arm(&self) -> Result<A, BanditError>;

    /// Update the state with the reward from pulling an arm.
    ///
    /// # Arguments
    /// * `arm` - The arm that was pulled
    /// * `reward` - The reward received from pulling the arm
    ///
    /// # Returns
    /// - `Ok(())`: If the update was successful
    /// - `Err(BanditError)`: If the update fails
    fn update(&self, arm: &A, reward: R) -> Result<(), BanditError>;

    /// Get the current state of the bandit.
    ///
    /// # Returns
    /// A reference to the current state of the bandit
    fn get_state(&self) -> &BanditState<A, R>;

    /// Reset the bandit to its initial state.
    ///
    /// # Returns
    /// - `Ok(())`: If the reset was successful
    /// - `Err(BanditError)`: If the reset fails
    fn reset(&self) -> Result<(), BanditError>;
}

/// Configuration trait for bandit algorithms.
///
/// This trait defines the configuration options that can be used to initialize
/// and customize bandit algorithms.
pub trait BanditConfig: Debug + Send + Sync {
    /// Validate the configuration.
    ///
    /// # Returns
    /// - `Ok(())`: If the configuration is valid
    /// - `Err(BanditError)`: If the configuration is invalid
    fn validate(&self) -> Result<(), BanditError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{NumericArm, NumericReward};

    // Mock implementation for testing
    struct MockBandit {
        state: BanditState<NumericArm, NumericReward<f64>>,
    }

    impl Bandit<NumericArm, NumericReward<f64>> for MockBandit {
        fn select_arm(&self) -> Result<NumericArm, BanditError> {
            Ok(NumericArm::new(1))
        }

        fn update(&self, arm: &NumericArm, reward: NumericReward<f64>) -> Result<(), BanditError> {
            self.state.update_reward(arm, reward)?;
            self.state.increment_count(arm)?;
            Ok(())
        }

        fn get_state(&self) -> &BanditState<NumericArm, NumericReward<f64>> {
            &self.state
        }

        fn reset(&self) -> Result<(), BanditError> {
            self.state.reset()?;
            Ok(())
        }
    }

    #[test]
    fn test_mock_bandit_operations() {
        let bandit = MockBandit {
            state: BanditState::new(),
        };

        // Test arm selection
        let arm = bandit.select_arm().unwrap();
        assert_eq!(arm.0, 1);

        // Test update
        let reward = NumericReward::new(1.0).unwrap();
        bandit.update(&arm, reward).unwrap();

        // Test state access
        let state = bandit.get_state();
        assert_eq!(state.get_count(&arm).unwrap(), 1);
        assert_eq!(state.get_reward(&arm).unwrap().get_value().unwrap(), 1.0);

        // Test reset
        bandit.reset().unwrap();
        assert!(state.get_count(&arm).is_err());
    }
}
