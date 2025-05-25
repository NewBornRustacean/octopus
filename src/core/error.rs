use crate::common::error::{RewardError, StateError};
use thiserror::Error;

/// Errors that can occur in bandit algorithms
#[derive(Debug, Error)]
pub enum BanditError {
    /// Errors related to state management
    #[error("State error: {0}")]
    StateError(#[from] StateError),

    /// Errors related to reward handling
    #[error("Reward error: {0}")]
    RewardError(#[from] RewardError),

    /// Errors related to invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Errors related to algorithm-specific issues
    #[error("Algorithm error: {0}")]
    AlgorithmError(String),

    /// Errors related to arm selection
    #[error("Arm selection error: {0}")]
    ArmSelectionError(String),

    /// Errors related to invalid state transitions
    #[error("Invalid state transition: {0}")]
    InvalidStateTransition(String),
}
