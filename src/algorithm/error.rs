use crate::common::{ArmError, RewardError, StateError};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum BanditError {
    #[error("Invalid epsilon value: {0}")]
    InvalidEpsilon(f64),

    #[error("State error: {0}")]
    StateError(#[from] StateError),

    #[error("Arm error: {0}")]
    ArmError(#[from] ArmError),

    #[error("Reward error: {0}")]
    RewardError(#[from] RewardError),
}
