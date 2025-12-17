//! Reinforcement learning environment for Snake game
//!
//! Provides:
//! - 4-channel grid observations (head, body, food, walls)
//! - Burn-compatible RL environment interface
//! - Backend-agnostic tensor operations
//! - Actor-Critic neural network for PPO training
//! - PPO algorithm configuration and training
//! - Model persistence for saving/loading trained agents

pub mod backend;
pub mod buffer;
pub mod config;
pub mod environment;
pub mod network;
pub mod observation;
pub mod persistence;
pub mod ppo;

pub use backend::{default_device, InferenceBackend, TrainingBackend};
pub use buffer::RolloutBuffer;
pub use config::PPOConfig;
pub use environment::SnakeEnvironment;
pub use network::{ActorCriticConfig, ActorCriticNetwork};
pub use observation::create_observation;
pub use persistence::{load_network, save_model, ModelMetadata};
pub use ppo::PPOAgent;
