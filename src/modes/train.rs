//! Training mode for PPO agent
//!
//! This module implements the training loop for the PPO agent. It collects
//! experiences by running episodes in the Snake environment, updates the agent
//! using PPO, and periodically saves checkpoints.
//!
//! # Example
//!
//! ```rust,ignore
//! use ml_snake::modes::{TrainMode, TrainConfig};
//! use ml_snake::game::GameConfig;
//! use ml_snake::rl::{PPOConfig, default_device, TrainingBackend};
//! use std::path::PathBuf;
//!
//! let train_config = TrainConfig {
//!     num_episodes: 10000,
//!     save_path: PathBuf::from("models/snake.bin"),
//!     checkpoint_frequency: 1000,
//!     log_frequency: 100,
//!     game_config: GameConfig::default(),
//!     ppo_config: PPOConfig::default(),
//! };
//!
//! let device = default_device();
//! let mut train_mode = TrainMode::<TrainingBackend>::new(train_config, device);
//! train_mode.run()?;
//! ```

use anyhow::{Context, Result};
use burn::tensor::backend::AutodiffBackend;
use std::iter::repeat;
use std::path::{Path, PathBuf};

use crate::game::GameConfig;
use crate::metrics::TrainingStats;
use crate::rl::{
    save_model, ActorCriticConfig, PPOAgent, PPOConfig, SnakeEnvironment,
};

/// Configuration for training mode
#[derive(Debug, Clone)]
pub struct TrainConfig {
    /// Number of episodes to train
    pub num_episodes: usize,

    /// Path to save the final trained model
    pub save_path: PathBuf,

    /// Save a checkpoint every N episodes
    pub checkpoint_frequency: usize,

    /// Log training progress every N episodes
    pub log_frequency: usize,

    /// Game configuration (grid size, rewards)
    pub game_config: GameConfig,

    /// PPO hyperparameters
    pub ppo_config: PPOConfig,
}

impl TrainConfig {
    /// Create a new training configuration with defaults
    ///
    /// # Arguments
    ///
    /// * `num_episodes` - Number of episodes to train
    /// * `save_path` - Path to save the final model
    ///
    /// # Example
    ///
    /// ```rust
    /// use ml_snake::modes::TrainConfig;
    /// use std::path::PathBuf;
    ///
    /// let config = TrainConfig::new(10000, PathBuf::from("models/snake.bin"));
    /// ```
    pub fn new(num_episodes: usize, save_path: PathBuf) -> Self {
        Self {
            num_episodes,
            save_path,
            checkpoint_frequency: 1000,
            log_frequency: 100,
            game_config: GameConfig::default(),
            ppo_config: PPOConfig::default(),
        }
    }
}

/// Training mode for PPO agent
///
/// Runs the training loop, collecting experiences and updating the agent using PPO.
/// Periodically logs progress and saves checkpoints.
pub struct TrainMode<B: AutodiffBackend> {
    /// PPO agent being trained
    agent: PPOAgent<B>,

    /// Snake environment for experience collection
    env: SnakeEnvironment<B::InnerBackend>,

    /// Training statistics tracker
    stats: TrainingStats,

    /// Training configuration
    config: TrainConfig,

    /// Current episode number
    current_episode: usize,

    /// Total steps across all episodes
    total_steps: usize,
}

impl<B: AutodiffBackend> TrainMode<B> {
    /// Create a new training mode
    ///
    /// # Arguments
    ///
    /// * `config` - Training configuration
    /// * `device` - Device for computation (CPU/GPU)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use ml_snake::modes::{TrainMode, TrainConfig};
    /// use ml_snake::rl::{default_device, TrainingBackend};
    ///
    /// let config = TrainConfig::new(10000, "models/snake.bin".into());
    /// let device = default_device();
    /// let mut train_mode = TrainMode::<TrainingBackend>::new(config, device);
    /// ```
    pub fn new(config: TrainConfig, device: B::Device) -> Self {
        // Initialize network
        let network_config = ActorCriticConfig::new(
            config.game_config.grid_height,
            config.game_config.grid_width,
        );
        let network = network_config.init::<B>(&device);

        // Initialize agent
        let agent = PPOAgent::new(
            network,
            config.ppo_config.clone(),
            config.game_config.grid_height,
            config.game_config.grid_width,
            device.clone(),
        );

        // Initialize environment
        let env = SnakeEnvironment::new(config.game_config.clone(), device);

        // Initialize stats tracker (100-episode rolling window)
        let stats = TrainingStats::new(100);

        Self {
            agent,
            env,
            stats,
            config,
            current_episode: 0,
            total_steps: 0,
        }
    }

    /// Run the training loop
    ///
    /// Trains the agent for the specified number of episodes, logging progress
    /// and saving checkpoints periodically.
    ///
    /// # Returns
    ///
    /// `Ok(())` on successful completion
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut train_mode = TrainMode::new(config, device);
    /// train_mode.run()?;
    /// ```
    pub fn run(&mut self) -> Result<()> {
        self.print_header();

        for episode in 0..self.config.num_episodes {
            self.current_episode = episode;

            // Run one episode
            let (episode_reward, episode_steps, episode_score) = self.run_episode()?;

            // Record episode stats
            self.stats
                .record_episode(episode_reward, episode_steps, episode_score);

            // Increment agent's episode counter
            self.agent.increment_episode();

            // Log progress
            if (episode + 1) % self.config.log_frequency == 0 {
                self.print_progress(episode + 1);
            }

            // Save checkpoint
            if (episode + 1) % self.config.checkpoint_frequency == 0 {
                self.save_checkpoint()?;
            }
        }

        // Final save
        self.save_model()?;

        println!("\nTraining complete!");
        println!("Final model saved to: {:?}", self.config.save_path);
        println!("\nFinal Statistics:");
        println!("{}", self.stats.format_summary());

        Ok(())
    }

    /// Run a single training episode
    ///
    /// Collects experiences by running the agent in the environment, storing
    /// transitions in the buffer. When the buffer is full, performs PPO updates.
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// - Total episode reward
    /// - Number of steps in the episode
    /// - Final score (food eaten)
    fn run_episode(&mut self) -> Result<(f32, usize, u32)> {
        let mut obs = self.env.reset();
        let mut episode_reward = 0.0;
        let mut episode_steps = 0;
        let mut done = false;

        while !done {
            // Select action
            let (action, log_prob, value) = self.agent.select_action(obs.clone());

            // Step environment
            let (next_obs, reward, terminated) = self.env.step(action);

            // Store transition
            self.agent
                .store_transition(obs, action, log_prob, reward, value, terminated);

            // Update counters
            episode_reward += reward;
            episode_steps += 1;
            self.total_steps += 1;
            done = terminated;
            obs = next_obs;

            // PPO update if buffer is full
            if self.agent.should_update() {
                // Get last value for bootstrapping
                let (_, _, last_value) = self.agent.select_action(obs.clone());

                // Perform PPO update
                let (policy_loss, value_loss, entropy, _total_loss) =
                    self.agent.update(last_value, done);

                // Record training metrics
                self.stats
                    .record_update(policy_loss, value_loss, entropy);
            }
        }

        let episode_score = self.env.state().score;

        Ok((episode_reward, episode_steps, episode_score))
    }

    /// Save a checkpoint of the current model
    fn save_checkpoint(&self) -> Result<()> {
        let checkpoint_path = self
            .config
            .save_path
            .parent()
            .unwrap_or(Path::new("."))
            .join(format!(
                "checkpoint_ep{}.bin",
                self.current_episode + 1
            ));

        save_model(&self.agent, &checkpoint_path)
            .with_context(|| format!("Failed to save checkpoint to {:?}", checkpoint_path))?;

        println!("  Checkpoint saved: {:?}", checkpoint_path);

        Ok(())
    }

    /// Save the final trained model
    fn save_model(&self) -> Result<()> {
        save_model(&self.agent, &self.config.save_path).with_context(|| {
            format!(
                "Failed to save final model to {:?}",
                self.config.save_path
            )
        })?;

        Ok(())
    }

    /// Print training header information
    fn print_header(&self) {
        println!("{}", "=".repeat(70));
        println!("PPO Training - ML Snake");
        println!("{}", "=".repeat(70));
        println!("Episodes: {}", self.config.num_episodes);
        println!(
            "Game Config: {}x{} grid",
            self.config.game_config.grid_width, self.config.game_config.grid_height
        );
        println!("PPO Config:");
        println!("  Learning rate: {}", self.config.ppo_config.learning_rate);
        println!("  Gamma: {}", self.config.ppo_config.gamma);
        println!("  GAE lambda: {}", self.config.ppo_config.gae_lambda);
        println!("  Clip epsilon: {}", self.config.ppo_config.clip_epsilon);
        println!(
            "  Update frequency: {} steps",
            self.config.ppo_config.update_frequency
        );
        println!("  Batch size: {}", self.config.ppo_config.batch_size);
        println!("  Epochs per update: {}", self.config.ppo_config.n_epochs);
        println!("Checkpoints: Every {} episodes", self.config.checkpoint_frequency);
        println!("Logging: Every {} episodes", self.config.log_frequency);
        println!("Save path: {:?}", self.config.save_path);
        println!("{}", "=".repeat(70));
        println!();
    }

    /// Print training progress
    fn print_progress(&self, episode: usize) {
        println!(
            "[Episode {}/{}] {}",
            episode,
            self.config.num_episodes,
            self.stats.format_summary()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rl::{default_device, TrainingBackend};
    use tempfile::TempDir;

    #[test]
    fn test_train_config_creation() {
        let config = TrainConfig::new(1000, PathBuf::from("test.bin"));
        assert_eq!(config.num_episodes, 1000);
        assert_eq!(config.save_path, PathBuf::from("test.bin"));
    }

    #[test]
    fn test_train_mode_creation() {
        let temp_dir = TempDir::new().unwrap();
        let save_path = temp_dir.path().join("model.bin");

        let mut config = TrainConfig::new(10, save_path);
        config.game_config = GameConfig::new(10, 10); // Small grid for test

        let device = default_device();
        let _train_mode = TrainMode::<TrainingBackend>::new(config, device);
        // If this doesn't panic, creation succeeded
    }

    #[test]
    fn test_run_single_episode() {
        let temp_dir = TempDir::new().unwrap();
        let save_path = temp_dir.path().join("model.bin");

        let mut config = TrainConfig::new(1, save_path);
        config.game_config = GameConfig::new(10, 10);
        config.ppo_config.update_frequency = 1000; // Don't update during test

        let device = default_device();
        let mut train_mode = TrainMode::<TrainingBackend>::new(config, device);

        // Run a single episode
        let result = train_mode.run_episode();
        assert!(result.is_ok());

        let (reward, steps, score) = result.unwrap();
        assert!(steps > 0);
        assert!(reward < 0.0 || score > 0); // Either died or ate food
    }
}
