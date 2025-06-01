use rand::Rng;
use std::hash::Hash;

/// Represents an arm in the bandit problem.
///
/// An arm must be:
/// - `Clone`: For creating copies when needed
/// - `Eq`: For equality comparison
/// - `Hash`: For use in hash-based collections
/// - `Send`: For thread safety (can be transferred between threads)
/// - `Sync`: For thread safety (can be shared between threads)
/// - `Debug`: For debugging purposes
pub trait Arm: Clone + Eq + Hash + Send + Sync + std::fmt::Debug {
    /// Validates if the arm is in a valid state.
    ///
    /// # Returns
    /// `true` if the arm is valid, `false` otherwise.
    fn is_valid(&self) -> bool;

    /// Returns the name of the arm.
    fn name(&self) -> &str;

    /// Returns the unique identifier of the arm
    fn id(&self) -> String;
}

/// A simple numeric arm implementation.
///
/// This is the most basic implementation of an arm, using a numeric identifier.
#[derive(Debug, Clone, Hash)]
pub struct NumericArm {
    pub id: usize,
    pub name: String,
}

impl PartialEq for NumericArm {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.name == other.name
    }
}

impl Eq for NumericArm {}

impl Arm for NumericArm {
    fn is_valid(&self) -> bool {
        true // Numeric arms are always valid
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn id(&self) -> String {
        self.id.to_string()
    }
}

impl NumericArm {
    /// Creates a new numeric arm. id is generated automatically.
    pub fn new(name: String) -> Self {
        let id = rand::rng().random_range(0..1000000);
        Self { id, name }
    }
}

/// A string-based arm implementation.
///
/// Useful for named arms like "red", "blue", "green" as shown in the README.
#[derive(Debug, Clone, Hash)]
pub struct StringArm {
    pub id: usize,
    pub name: String,
}

impl PartialEq for StringArm {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.name == other.name
    }
}

impl Eq for StringArm {}

impl Arm for StringArm {
    fn is_valid(&self) -> bool {
        !self.name.is_empty() // String arms are valid if they're not empty
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn id(&self) -> String {
        self.id.to_string()
    }
}

impl StringArm {
    pub fn new(name: String) -> Self {
        let id = rand::rng().random_range(0..1000000);
        Self { id, name }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numeric_arm_creation() {
        let arm = NumericArm::new("test".to_string());
        assert!(arm.is_valid());
        assert_eq!(arm.name(), "test");
    }

    #[test]
    fn test_numeric_arm_equality() {
        let arm1 = NumericArm {
            id: 1,
            name: "test".to_string(),
        };
        let arm2 = NumericArm {
            id: 1,
            name: "test".to_string(),
        };
        let arm3 = NumericArm {
            id: 2,
            name: "test".to_string(),
        };
        let arm4 = NumericArm {
            id: 1,
            name: "test2".to_string(),
        };

        assert_eq!(arm1, arm2); // same id and name
        assert_ne!(arm1, arm3); // different id
        assert_ne!(arm1, arm4); // different name
    }

    #[test]
    fn test_numeric_arm_clone() {
        let arm1 = NumericArm::new("test".to_string());
        let arm2 = arm1.clone();
        assert_eq!(arm1, arm2);
    }

    #[test]
    fn test_string_arm_creation() {
        let arm = StringArm::new("red".to_string());
        assert!(arm.is_valid());
        assert_eq!(arm.name(), "red");
    }

    #[test]
    fn test_string_arm_equality() {
        let arm1 = StringArm {
            id: 1,
            name: "red".to_string(),
        };
        let arm2 = StringArm {
            id: 1,
            name: "red".to_string(),
        };
        let arm3 = StringArm {
            id: 2,
            name: "red".to_string(),
        };
        let arm4 = StringArm {
            id: 1,
            name: "blue".to_string(),
        };

        assert_eq!(arm1, arm2); // same id and name
        assert_ne!(arm1, arm3); // different id
        assert_ne!(arm1, arm4); // different name
    }

    #[test]
    fn test_string_arm_invalid() {
        let arm = StringArm::new("".to_string());
        assert!(!arm.is_valid());
    }
}
