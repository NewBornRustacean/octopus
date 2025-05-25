use crate::common::{Arm, Reward, state::BanditState};
use crate::core::{Bandit, BanditConfig, BanditError};
use std::fmt::Debug as FmtDebug;

/// Configuration for the EpsilonGreedy algorithm.
///
/// # Parameters
/// * `epsilon` - Probability of exploration (0.0 to 1.0)
/// * `arms` - Initial set of arms to choose from
#[derive(Debug, Clone)]
pub struct EpsilonGreedyConfig<A: Arm> {
    /// Probability of exploration (0.0 to 1.0)
    pub epsilon: f64,
    /// Initial set of arms to choose from
    pub arms: Vec<A>,
}

impl<A: Arm + FmtDebug> BanditConfig for EpsilonGreedyConfig<A> {
    fn validate(&self) -> Result<(), BanditError> {
        if self.epsilon < 0.0 || self.epsilon > 1.0 {
            return Err(BanditError::InvalidConfig(
                "Epsilon must be between 0.0 and 1.0".to_string(),
            ));
        }
        if self.arms.is_empty() {
            return Err(BanditError::InvalidConfig(
                "At least one arm must be provided".to_string(),
            ));
        }
        Ok(())
    }
}

/// EpsilonGreedy bandit algorithm implementation.
///
/// This algorithm balances exploration and exploitation using a fixed probability (epsilon)
/// of choosing a random arm (exploration) versus choosing the best known arm (exploitation).
pub struct EpsilonGreedy<A: Arm, R: Reward> {
    config: EpsilonGreedyConfig<A>,
    state: BanditState<A, R>,
}

impl<A: Arm + FmtDebug, R: Reward> EpsilonGreedy<A, R> {
    /// Creates a new EpsilonGreedy bandit with the given configuration.
    ///
    /// # Arguments
    /// * `config` - The configuration for the algorithm
    ///
    /// # Returns
    /// - `Ok(Self)`: The configured bandit
    /// - `Err(BanditError)`: If the configuration is invalid
    pub fn new(config: EpsilonGreedyConfig<A>) -> Result<Self, BanditError> {
        config.validate()?;
        Ok(Self {
            config,
            state: BanditState::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{NumericArm, NumericReward};

    #[test]
    fn test_config_validation() {
        // Test valid config
        let config = EpsilonGreedyConfig {
            epsilon: 0.1,
            arms: vec![NumericArm::new(1), NumericArm::new(2)],
        };
        assert!(config.validate().is_ok());

        // Test invalid epsilon
        let config = EpsilonGreedyConfig {
            epsilon: 1.5,
            arms: vec![NumericArm::new(1)],
        };
        assert!(config.validate().is_err());

        // Test empty arms - this should fail validation
        let empty_config = EpsilonGreedyConfig::<NumericArm> {
            epsilon: 0.1,
            arms: Vec::new(),
        };
        let validation_result = empty_config.validate();
        assert!(validation_result.is_err());
        assert!(matches!(
            validation_result,
            Err(BanditError::InvalidConfig(_))
        ));
    }

    #[test]
    fn test_bandit_creation() {
        let config = EpsilonGreedyConfig {
            epsilon: 0.1,
            arms: vec![NumericArm::new(1), NumericArm::new(2)],
        };
        let bandit = EpsilonGreedy::<NumericArm, NumericReward<f64>>::new(config);
        assert!(bandit.is_ok());
    }
}
