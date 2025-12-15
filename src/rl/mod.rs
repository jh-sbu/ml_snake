//! Reinforcement learning environment for Snake game
//!
//! Provides:
//! - 4-channel grid observations (head, body, food, walls)
//! - Burn-compatible RL environment interface
//! - Backend-agnostic tensor operations

pub mod environment;
pub mod observation;

pub use environment::SnakeEnvironment;
pub use observation::create_observation;
