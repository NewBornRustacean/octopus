use std::hash::Hash;

/// Represents an arm in the bandit problem.
///
/// An arm must be:
/// - `Clone`: For creating copies when needed
/// - `Eq`: For equality comparison
/// - `Hash`: For use in hash-based collections
/// - `Send`: For thread safety (can be transferred between threads)
/// - `Sync`: For thread safety (can be shared between threads)
pub trait Arm: Clone + Eq + Hash + Send + Sync {
    /// Validates if the arm is in a valid state.
    ///
    /// # Returns
    /// `true` if the arm is valid, `false` otherwise.
    fn is_valid(&self) -> bool;
}

/// A simple numeric arm implementation.
///
/// This is the most basic implementation of an arm, using a numeric identifier.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct NumericArm(pub usize);

impl Arm for NumericArm {
    fn is_valid(&self) -> bool {
        true // Numeric arms are always valid
    }
}

impl NumericArm {
    /// Creates a new numeric arm with the given identifier.
    pub fn new(id: usize) -> Self {
        Self(id)
    }
}

/// A string-based arm implementation.
///
/// Useful for named arms like "red", "blue", "green" as shown in the README.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct StringArm(pub String);

impl Arm for StringArm {
    fn is_valid(&self) -> bool {
        !self.0.is_empty() // String arms are valid if they're not empty
    }
}

impl StringArm {
    pub fn new(id: String) -> Self {
        Self(id)
    }
}

/// A generic wrapper for custom arm types.
///
/// This allows users to create their own arm types while ensuring they meet the required trait bounds.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct CustomArm<T: Clone + Eq + Hash + Send + Sync>(pub T);

impl<T: Clone + Eq + Hash + Send + Sync> Arm for CustomArm<T> {
    fn is_valid(&self) -> bool {
        true // Custom validation can be added if needed
    }
}

impl<T: Clone + Eq + Hash + Send + Sync> CustomArm<T> {
    pub fn new(value: T) -> Self {
        Self(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numeric_arm_creation() {
        let arm = NumericArm::new(1);
        assert!(arm.is_valid());
        assert_eq!(arm.0, 1);
    }

    #[test]
    fn test_numeric_arm_equality() {
        let arm1 = NumericArm::new(1);
        let arm2 = NumericArm::new(1);
        let arm3 = NumericArm::new(2);

        assert_eq!(arm1, arm2);
        assert_ne!(arm1, arm3);
    }

    #[test]
    fn test_numeric_arm_clone() {
        let arm1 = NumericArm::new(1);
        let arm2 = arm1.clone();
        assert_eq!(arm1, arm2);
    }

    #[test]
    fn test_string_arm_creation() {
        let arm = StringArm::new("red".to_string());
        assert!(arm.is_valid());
        assert_eq!(arm.0, "red");
    }

    #[test]
    fn test_string_arm_equality() {
        let arm1 = StringArm::new("red".to_string());
        let arm2 = StringArm::new("red".to_string());
        let arm3 = StringArm::new("blue".to_string());

        assert_eq!(arm1, arm2);
        assert_ne!(arm1, arm3);
    }

    #[test]
    fn test_string_arm_invalid() {
        let arm = StringArm::new("".to_string());
        assert!(!arm.is_valid());
    }

    #[test]
    fn test_custom_arm() {
        #[derive(Debug, Clone, Eq, PartialEq, Hash)]
        struct MyArm {
            id: String,
            value: i32,
        }

        let custom_arm = CustomArm::new(MyArm {
            id: "test".to_string(),
            value: 42,
        });

        assert!(custom_arm.is_valid());
        assert_eq!(custom_arm.0.id, "test");
        assert_eq!(custom_arm.0.value, 42);
    }
}
