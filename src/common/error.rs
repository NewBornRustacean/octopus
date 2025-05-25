use thiserror::Error;

#[derive(Error, Debug)]
pub enum ArmError {
    #[error("Invalid arm identifier")]
    InvalidArmIdentifier,

    #[error("Arm not found")]
    ArmNotFound,
    // Add more error types as needed
}

#[derive(Error, Debug)]
pub enum RewardError {
    #[error("Invalid reward value")]
    InvalidRewardValue,

    #[error("Reward calculation failed")]
    RewardCalculationFailed,

    #[error("Reward type mismatch: expected {expected}, got {actual}")]
    RewardTypeMismatch { expected: String, actual: String },
    // Add more error types as needed
}

#[derive(Error, Debug)]
pub enum StateError {
    #[error("Invalid state operation: {0}")]
    InvalidStateOperation(String),

    #[error("State update failed")]
    StateUpdateFailed,

    #[error("Arm not found in state: {0}")]
    ArmNotFoundInState(String),

    #[error("State initialization failed: {0}")]
    StateInitializationFailed(String),

    #[error("State reset failed: {0}")]
    StateResetFailed(String),

    #[error("Concurrent state access error: {0}")]
    ConcurrentStateAccessError(String),

    #[error("Invalid reward update: {0}")]
    InvalidRewardUpdate(String),
}