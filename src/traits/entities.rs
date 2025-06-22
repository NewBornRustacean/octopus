use crate::utils::error::OctopusError;
use ndarray::{Array, Array1, Dimension, Ix1};
use std::collections::HashMap;
use std::hash::Hash; // For 1-dimensional feature vectors
use std::ops::{Deref, DerefMut};
use rand::{rng, Rng};


/// Represents an action (or arm) in a Multi-Armed Bandit problem.
///
/// Implementors must be clonable, equatable, hashable, and thread-safe.
/// The associated `ValueType` allows for flexible action payloads.
pub trait Action: Clone + Eq + Hash + Send + Sync + 'static {
    /// The type of value carried by this action (e.g., an identifier, label, or struct).
    type ValueType;

    /// Returns a unique, stable identifier for this action instance.
    fn id(&self) -> u32;

    /// Returns a human-readable name for this action (for logging/debugging).
    fn name(&self) -> String {
        format!("Action-{}", self.id())
    }

    /// Returns the value associated with this action.
    fn value(&self) -> Self::ValueType;
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct NumericAction<T>
where
    T: Copy + PartialEq + Eq + Hash + Send + Sync + 'static,
{
    id: u32,
    value: T,
    name: String,
}

impl<T> NumericAction<T>
where
    T: Copy + PartialEq + Eq + Hash + Send + Sync + 'static,
{
    /// Create a new NumericAction with a random ID
    pub fn new(value: T, name:&str) -> Self {
        let mut rng = rng();
        let id = rng.random::<u32>();
        Self { id, value, name: name.to_string() }
    }
    
    /// Create a new NumericAction with a given ID
    /// This is for test cases. 
    pub fn with_id(id:u32, value: T, name: &str) -> Self {
        Self { id, value, name: name.to_string() }
    }
}

impl<T> Action for NumericAction<T>
where
    T: Copy + Eq + Hash + Send + Sync + 'static,
{
    type ValueType = T;

    fn id(&self) -> u32 {
        self.id
    }

    fn value(&self) -> T {
        self.value
    }

    fn name(&self) -> String {
        self.name.clone()
    }
}


/// Stores a collection of actions, indexed by their unique ID.
#[derive(Debug, Clone)]
pub struct ActionStorage<A: Action>(HashMap<u32, A>);

impl<A: Action + Clone> ActionStorage<A> {
    /// Creates a new ActionStorage from a slice of actions.
    pub fn new(initial_actions: &[A]) -> Result<Self, OctopusError> {
        let actions = initial_actions
            .into_iter()
            .map(|action| (action.id(), action.clone()))
            .collect();
        Ok(ActionStorage { 0: actions })
    }
    /// Returns all actions as a vector.
    pub fn get_all_actions(&self) -> Vec<A> {
        self.0.values().cloned().collect()
    }
}

impl<A: Action> Deref for ActionStorage<A> {
    type Target = HashMap<u32, A>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<A: Action> DerefMut for ActionStorage<A> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Represents a reward signal received after taking an action in a context.
///
/// The reward is always interpreted as a value to maximize (higher is better).
pub trait Reward: Clone + Send + Sync + 'static {
    /// Returns the scalar value of the reward.
    fn value(&self) -> f64;
}

/// Represents the contextual information available to the bandit algorithm.
///
/// The context is typically converted to an ndarray for use in contextual algorithms.
pub trait Context: Clone + Send + Sync + 'static {
    /// The ndarray dimension type for this context.
    type DimType: Dimension;
    /// Converts the context into an ndarray of features (usually 1D, but extensible).
    fn to_ndarray(&self) -> Array<f64, Self::DimType>;
}

/// Dummy context for non-contextual bandits or testing.
#[derive(Debug, Clone, PartialEq)]
pub struct DummyContext;

impl Context for DummyContext {
    type DimType = Ix1;
    fn to_ndarray(&self) -> Array<f64, Self::DimType> {
        Array1::from_vec(vec![0.0])
    }
}
