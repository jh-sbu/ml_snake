//! Core game logic module for Snake
//!
//! This module contains all the game logic without any I/O or rendering dependencies.
//! It can be used programmatically for both human play and RL training.

pub mod action;
pub mod config;
pub mod engine;
pub mod state;

// Re-export commonly used types
pub use action::{Action, Direction};
pub use config::GameConfig;
pub use engine::{GameEngine, StepInfo, StepResult};
pub use state::{CollisionType, GameState, Position, Snake};
