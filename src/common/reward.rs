use crate::common::error::RewardError;

/// Represents a reward in the bandit problem.
pub trait Reward: Send + Sync {
    fn is_valid(&self) -> Result<(), RewardError>;
    fn get_value(&self) -> Result<f64, RewardError>;
}

/// Trait for aggregating rewards over time (e.g., mean, sum).
pub trait RewardAggregator: Send + Sync {
    fn update(&mut self, reward: f64) -> Result<(), RewardError>;
    fn mean(&self) -> Result<f64, RewardError>;
}

/// A simple numeric reward implementation.
#[derive(Debug, Clone, Copy)]
pub struct NumericReward {
    value: f64,
}

impl Reward for NumericReward {
    fn is_valid(&self) -> Result<(), RewardError> {
        if self.value.is_finite() {
            Ok(())
        } else {
            Err(RewardError::InvalidRewardValue)
        }
    }

    fn get_value(&self) -> Result<f64, RewardError> {
        Ok(self.value)
    }
}

impl NumericReward {
    pub fn new(value: f64) -> Result<Self, RewardError> {
        let reward = Self { value };
        reward.is_valid()?;
        Ok(reward)
    }
}

/// A binary reward implementation.
#[derive(Debug, Clone, Copy)]
pub struct BinaryReward {
    value: bool,
}

impl Reward for BinaryReward {
    fn is_valid(&self) -> Result<(), RewardError> {
        Ok(())
    }

    fn get_value(&self) -> Result<f64, RewardError> {
        Ok(if self.value { 1.0 } else { 0.0 })
    }
}

impl BinaryReward {
    pub fn new(value: bool) -> Self {
        Self { value }
    }

    pub fn success() -> Self {
        Self::new(true)
    }

    pub fn failure() -> Self {
        Self::new(false)
    }
}

/// Aggregator that maintains a running mean.
#[derive(Debug)]
pub struct MeanAggregator {
    count: usize,
    total: f64,
}

impl MeanAggregator {
    pub fn new() -> Self {
        Self {
            count: 0,
            total: 0.0,
        }
    }
}

impl RewardAggregator for MeanAggregator {
    fn update(&mut self, reward: f64) -> Result<(), RewardError> {
        if !reward.is_finite() {
            return Err(RewardError::InvalidRewardValue);
        }
        self.total += reward;
        self.count += 1;
        Ok(())
    }

    fn mean(&self) -> Result<f64, RewardError> {
        if self.count == 0 {
            Err(RewardError::InvalidRewardValue)
        } else {
            Ok(self.total / self.count as f64)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64;

    mod numeric_reward_tests {
        use super::*;

        #[test]
        fn test_invalid_values() {
            // NaN
            assert!(NumericReward::new(f64::NAN).is_err());

            // Infinity
            assert!(NumericReward::new(f64::INFINITY).is_err());
            assert!(NumericReward::new(f64::NEG_INFINITY).is_err());
        }

        #[test]
        fn test_valid_values() {
            // Zero
            let zero = NumericReward::new(0.0).unwrap();
            assert_eq!(zero.get_value().unwrap(), 0.0);

            // Positive number
            let positive = NumericReward::new(42.0).unwrap();
            assert_eq!(positive.get_value().unwrap(), 42.0);

            // Negative number
            let negative = NumericReward::new(-42.0).unwrap();
            assert_eq!(negative.get_value().unwrap(), -42.0);

            // Small number
            let small = NumericReward::new(1e-10).unwrap();
            assert_eq!(small.get_value().unwrap(), 1e-10);

            // Large number
            let large = NumericReward::new(1e10).unwrap();
            assert_eq!(large.get_value().unwrap(), 1e10);
        }
    }

    mod binary_reward_tests {
        use super::*;

        #[test]
        fn test_creation() {
            // True
            let true_reward = BinaryReward::new(true);
            assert_eq!(true_reward.get_value().unwrap(), 1.0);

            // False
            let false_reward = BinaryReward::new(false);
            assert_eq!(false_reward.get_value().unwrap(), 0.0);

            // Success helper
            let success = BinaryReward::success();
            assert_eq!(success.get_value().unwrap(), 1.0);

            // Failure helper
            let failure = BinaryReward::failure();
            assert_eq!(failure.get_value().unwrap(), 0.0);
        }
    }

    mod mean_aggregator_tests {
        use super::*;

        #[test]
        fn test_invalid_updates() {
            let mut agg = MeanAggregator::new();

            // NaN
            assert!(agg.update(f64::NAN).is_err());

            // Infinity
            assert!(agg.update(f64::INFINITY).is_err());
            assert!(agg.update(f64::NEG_INFINITY).is_err());
        }

        #[test]
        fn test_valid_updates() {
            let mut agg = MeanAggregator::new();

            // Zero
            assert!(agg.update(0.0).is_ok());
            assert_eq!(agg.mean().unwrap(), 0.0);

            // Positive number
            let mut agg = MeanAggregator::new();
            assert!(agg.update(42.0).is_ok());
            assert_eq!(agg.mean().unwrap(), 42.0);

            // Negative number
            let mut agg = MeanAggregator::new();
            assert!(agg.update(-42.0).is_ok());
            assert_eq!(agg.mean().unwrap(), -42.0);
        }

        #[test]
        fn test_multiple_updates() {
            let mut agg = MeanAggregator::new();

            // Multiple positive numbers
            assert!(agg.update(10.0).is_ok());
            assert!(agg.update(20.0).is_ok());
            assert!(agg.update(30.0).is_ok());
            assert_eq!(agg.mean().unwrap(), 20.0);

            // Mixed positive and negative
            let mut agg = MeanAggregator::new();
            assert!(agg.update(10.0).is_ok());
            assert!(agg.update(-10.0).is_ok());
            assert_eq!(agg.mean().unwrap(), 0.0);
        }

        #[test]
        fn test_no_updates() {
            let agg = MeanAggregator::new();
            assert!(agg.mean().is_err());
        }

        #[test]
        fn test_edge_cases() {
            let mut agg = MeanAggregator::new();

            // Maximum f64
            assert!(agg.update(f64::MAX).is_ok());
            assert_eq!(agg.mean().unwrap(), f64::MAX);

            // Minimum f64
            let mut agg = MeanAggregator::new();
            assert!(agg.update(f64::MIN).is_ok());
            assert_eq!(agg.mean().unwrap(), f64::MIN);

            // Epsilon
            let mut agg = MeanAggregator::new();
            assert!(agg.update(f64::EPSILON).is_ok());
            assert_eq!(agg.mean().unwrap(), f64::EPSILON);
        }
    }
}
