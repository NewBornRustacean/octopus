//! Bandit algorithm implementations.
//!
//! This module contains concrete implementations of bandit policies, such as Epsilon-Greedy.
//! All algorithms implement the BanditPolicy trait and are generic over action, reward, and context types.

pub mod epsilon_greedy;
