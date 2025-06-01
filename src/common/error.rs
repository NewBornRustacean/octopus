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
    #[error("Arm not found")]
    ArmNotFound,

    #[error("Arm already exists")]
    ArmAlreadyExists,

    #[error("Reward error: {0}")]
    RewardError(#[from] RewardError),

    #[error("No arms available")]
    NoArmsAvailable,
}
