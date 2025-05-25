use crate::common::error::RewardError;
use num_traits::Float;

/// Represents a reward in the bandit problem.
///
/// A reward must be:
/// - `Clone`: For creating copies when needed
/// - `Send`: For thread safety (can be transferred between threads)
/// - `Sync`: For thread safety (can be shared between threads)
pub trait Reward: Clone + Send + Sync {
    /// Validates if the reward is in a valid state.
    ///
    /// # Returns
    /// `Result<(), RewardError>` - Ok if valid, Err with reason if invalid
    fn is_valid(&self) -> Result<(), RewardError>;

    /// Gets the numeric value of the reward.
    ///
    /// # Returns
    /// `Result<f64, RewardError>` - The reward value as f64, or an error if conversion fails
    fn get_value(&self) -> Result<f64, RewardError>;
}

/// A simple numeric reward implementation.
///
/// This is the most basic implementation of a reward, using a numeric value.
#[derive(Debug, Clone)]
pub struct NumericReward<T: Float> {
    pub value: T,
}

impl<T: Float + Send + Sync> Reward for NumericReward<T> {
    fn is_valid(&self) -> Result<(), RewardError> {
        if self.value.is_finite() {
            Ok(())
        } else {
            Err(RewardError::InvalidRewardValue)
        }
    }

    fn get_value(&self) -> Result<f64, RewardError> {
        Ok(self.value.to_f64().ok_or_else(|| {
            RewardError::InvalidRewardValue
        })?)
    }
}

impl<T: Float + Send + Sync> NumericReward<T> {
    /// Creates a new numeric reward with the given value.
    pub fn new(value: T) -> Result<Self, RewardError> {
        let reward = Self { value };
        reward.is_valid()?;
        Ok(reward)
    }
}

/// A binary reward implementation.
/// 
/// This is used for binary outcomes (success/failure, 0/1) in bandit problems.
#[derive(Debug, Clone)]
pub struct BinaryReward {
    pub value: bool,
}

impl Reward for BinaryReward {
    fn is_valid(&self) -> Result<(), RewardError> {
        Ok(()) // Binary rewards are always valid
    }

    fn get_value(&self) -> Result<f64, RewardError> {
        Ok(if self.value { 1.0 } else { 0.0 })
    }
}

impl BinaryReward {
    /// Creates a new binary reward.
    pub fn new(value: bool) -> Self {
        Self { value }
    }

    /// Creates a success reward (1.0).
    pub fn success() -> Self {
        Self { value: true }
    }

    /// Creates a failure reward (0.0).
    pub fn failure() -> Self {
        Self { value: false }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numeric_reward_creation() {
        let reward = NumericReward::new(42.0).unwrap();
        assert!(reward.is_valid().is_ok());
        assert_eq!(reward.get_value().unwrap(), 42.0);
    }

    #[test]
    fn test_numeric_reward_invalid() {
        let reward = NumericReward::new(f64::INFINITY);
        assert!(reward.is_err());
        assert_eq!(matches!(reward, Err(RewardError::InvalidRewardValue)), true);

        let reward = NumericReward::new(f64::NAN);
        assert!(reward.is_err());
        assert_eq!(matches!(reward, Err(RewardError::InvalidRewardValue)), true);

        let reward = NumericReward::new(f64::NEG_INFINITY);
        assert!(reward.is_err());
        assert_eq!(matches!(reward, Err(RewardError::InvalidRewardValue)), true);
    }

    #[test]
    fn test_binary_reward_creation() {
        let reward = BinaryReward::new(true);
        assert!(reward.is_valid().is_ok());
        assert_eq!(reward.get_value().unwrap(), 1.0);
        assert!(reward.value); // Testing the value field directly

        let reward = BinaryReward::new(false);
        assert!(reward.is_valid().is_ok());
        assert_eq!(reward.get_value().unwrap(), 0.0);
        assert!(!reward.value); // Testing the value field directly
    }

    #[test]
    fn test_binary_reward_helpers() {
        let success = BinaryReward::success();
        assert_eq!(success.get_value().unwrap(), 1.0);
        assert!(success.value); // Testing the value field directly

        let failure = BinaryReward::failure();
        assert_eq!(failure.get_value().unwrap(), 0.0);
        assert!(!failure.value); // Testing the value field directly
    }
}
