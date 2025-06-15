use thiserror::Error;

#[derive(Error, Debug, PartialEq)]
pub enum OctopusError {
    /// Error indicating that a parameter received an invalid value.
    ///
    /// # Fields
    /// - `parameter_name`: The name of the parameter that was invalid.
    /// - `value`: The actual value received for the parameter (as a string).
    /// - `expected_range`: A description of the valid range or expected values.
    #[error("Invalid parameter '{parameter_name}': received '{value}', expected {expected_range}")]
    InvalidParameter {
        parameter_name: String,
        value: String,
        expected_range: String,
    },

    /// Error indicating that a required collection (e.g., a list of actions) was empty.
    ///
    /// # Fields
    /// - `collection_name`: The name of the empty collection.
    #[error("Collection '{collection_name}' cannot be empty.")]
    EmptyCollection { collection_name: String },
    // can add more specific error types here as the library grows, e.g.:
    // #[error("Algorithm specific error: {0}")]
    // AlgorithmError(String),
    // #[error("Simulation error: {0}")]
    // SimulationError(String),
    // #[error("Data processing error: {0}")]
    // DataProcessingError(String),
}
