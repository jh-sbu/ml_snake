//! Reinforcement learning environment for Snake game
//!
//! Provides:
//! - 4-channel grid observations (head, body, food, walls)
//! - Burn-compatible RL environment interface
//! - Backend-agnostic tensor operations
//! - Actor-Critic neural network for PPO training

pub mod environment;
pub mod network;
pub mod observation;

pub use environment::SnakeEnvironment;
pub use network::{ActorCriticConfig, ActorCriticNetwork};
pub use observation::create_observation;
