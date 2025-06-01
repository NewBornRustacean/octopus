use crate::algorithm::error::BanditError;
use crate::common::{arm::Arm, error::StateError, reward::RewardAggregator, state::StateStore};
use rand::Rng;

/// Epsilon-greedy bandit algorithm implementation.
///
/// This algorithm balances exploration and exploitation by:
/// - With probability ε: randomly select an arm (exploration)
/// - With probability 1-ε: select the arm with the highest mean reward (exploitation)
#[derive(Debug)]
pub struct EpsilonGreedy {
    epsilon: f64,
}

impl EpsilonGreedy {
    /// Creates a new epsilon-greedy bandit algorithm.
    ///
    /// # Arguments
    /// * `epsilon` - The exploration rate (0.0 to 1.0)
    ///
    /// # Returns
    /// * `Result<Self, BanditError>` - The algorithm instance or an error if epsilon is invalid
    pub fn new(epsilon: f64) -> Result<Self, BanditError> {
        if !(0.0..=1.0).contains(&epsilon) {
            return Err(BanditError::InvalidEpsilon(epsilon));
        }
        Ok(Self { epsilon })
    }

    /// Selects an arm using the epsilon-greedy strategy.
    ///
    /// # Arguments
    /// * `state` - The current state of all arms
    ///
    /// # Returns
    /// * `Result<A, BanditError>` - The selected arm or an error
    pub fn select_arm<A: Arm, RA: RewardAggregator>(
        &self,
        state: &StateStore<A, RA>,
    ) -> Result<A, BanditError> {
        if state.states.len() == 0 {
            return Err(BanditError::StateError(StateError::NoArmsAvailable));
        }

        let mut rng = rand::thread_rng();
        if rng.gen_bool(self.epsilon) {
            // Exploration: randomly select an arm
            let arms: Vec<_> = state.states.iter().map(|entry| entry.key().clone()).collect();
            let random_idx = rng.gen_range(0..arms.len());
            Ok(arms[random_idx].clone())
        } else {
            // Exploitation: select the best arm
            state.best_arm().map_err(|e| match e {
                StateError::NoArmsAvailable => BanditError::StateError(e),
                _ => BanditError::StateError(e),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{
        arm::NumericArm,
        reward::{MeanAggregator, NumericReward},
    };

    #[test]
    fn test_invalid_epsilon() {
        assert!(matches!(
            EpsilonGreedy::new(-0.1),
            Err(BanditError::InvalidEpsilon(-0.1))
        ));
        assert!(matches!(
            EpsilonGreedy::new(1.1),
            Err(BanditError::InvalidEpsilon(1.1))
        ));
    }

    #[test]
    fn test_valid_epsilon() {
        assert!(EpsilonGreedy::new(0.0).is_ok());
        assert!(EpsilonGreedy::new(0.5).is_ok());
        assert!(EpsilonGreedy::new(1.0).is_ok());
    }

    #[test]
    fn test_empty_state() {
        let state: StateStore<NumericArm, MeanAggregator> = StateStore::new();
        let bandit = EpsilonGreedy::new(0.1).unwrap();
        assert!(matches!(
            bandit.select_arm(&state),
            Err(BanditError::StateError(StateError::NoArmsAvailable))
        ));
    }

    #[test]
    fn test_exploitation() {
        let state: StateStore<NumericArm, MeanAggregator> = StateStore::new();
        let bandit = EpsilonGreedy::new(0.0).unwrap(); // Always exploit

        // Add arms with different rewards
        let arm1 = NumericArm::new("arm1".to_string());
        let arm2 = NumericArm::new("arm2".to_string());
        let arm3 = NumericArm::new("arm3".to_string());

        state.add_arm(arm1.clone(), MeanAggregator::new()).unwrap();
        state.add_arm(arm2.clone(), MeanAggregator::new()).unwrap();
        state.add_arm(arm3.clone(), MeanAggregator::new()).unwrap();

        // Update arms with different rewards
        state.update(arm1.clone(), NumericReward::new(1.0).unwrap()).unwrap();
        state.update(arm2.clone(), NumericReward::new(2.0).unwrap()).unwrap();
        state.update(arm3.clone(), NumericReward::new(3.0).unwrap()).unwrap();

        // Should always select arm3 (highest mean reward)
        for _ in 0..100 {
            let selected = bandit.select_arm(&state).unwrap();
            assert_eq!(selected.id, arm3.id);
        }
    }

    #[test]
    fn test_exploration() {
        let state: StateStore<NumericArm, MeanAggregator> = StateStore::new();
        let bandit = EpsilonGreedy::new(1.0).unwrap(); // Always explore

        // Add arms
        let arm1 = NumericArm::new("arm1".to_string());
        let arm2 = NumericArm::new("arm2".to_string());
        let arm3 = NumericArm::new("arm3".to_string());

        state.add_arm(arm1.clone(), MeanAggregator::new()).unwrap();
        state.add_arm(arm2.clone(), MeanAggregator::new()).unwrap();
        state.add_arm(arm3.clone(), MeanAggregator::new()).unwrap();

        // Update arms with different rewards
        state.update(arm1.clone(), NumericReward::new(1.0).unwrap()).unwrap();
        state.update(arm2.clone(), NumericReward::new(2.0).unwrap()).unwrap();
        state.update(arm3.clone(), NumericReward::new(3.0).unwrap()).unwrap();

        // Count selections of each arm
        let mut counts = [0, 0, 0];
        let arms = [&arm1, &arm2, &arm3];
        let n_trials = 1000;

        for _ in 0..n_trials {
            let selected = bandit.select_arm(&state).unwrap();
            for (i, arm) in arms.iter().enumerate() {
                if selected.id == arm.id {
                    counts[i] += 1;
                    break;
                }
            }
        }

        // Each arm should be selected roughly 1/3 of the time
        let expected = n_trials as f64 / 3.0;
        let tolerance = 0.1 * expected; // 10% tolerance

        for count in counts {
            assert!((count as f64 - expected).abs() < tolerance);
        }
    }
}
