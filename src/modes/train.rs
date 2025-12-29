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
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::game::GameConfig;
use crate::metrics::{PerformanceMetrics, TimingKey, TrainingStats};
use crate::rl::{ActorCriticConfig, PPOAgent, PPOConfig, SnakeEnvironment, save_model};

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

    /// Maximum steps per episode (prevents infinite loops)
    pub max_steps_per_episode: usize,

    /// Game configuration (grid size, rewards)
    pub game_config: GameConfig,

    /// PPO hyperparameters
    pub ppo_config: PPOConfig,

    /// Enable performance metrics collection
    pub perf_metrics_enabled: bool,

    /// Enable fine-grained performance metrics (higher overhead)
    pub perf_fine_grained: bool,

    /// Optional path to export performance metrics as CSV
    pub perf_output_path: Option<PathBuf>,
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
            max_steps_per_episode: 1000,
            game_config: GameConfig::default(),
            ppo_config: PPOConfig::default(),
            perf_metrics_enabled: false,
            perf_fine_grained: false,
            perf_output_path: None,
        }
    }

    /// Validate training configuration
    ///
    /// Checks that PPO config is valid and warns about memory-intensive settings.
    ///
    /// # Returns
    ///
    /// `Ok(())` if configuration is valid, `Err` with detailed message otherwise.
    pub fn validate(&self) -> Result<()> {
        // Validate PPO config
        self.ppo_config
            .validate()
            .map_err(|e| anyhow::anyhow!("Invalid PPO configuration: {}", e))?;

        // Warn about memory-intensive configurations
        let buffer_memory_mb = estimate_buffer_memory_mb(
            self.ppo_config.update_frequency,
            self.game_config.grid_height,
            self.game_config.grid_width,
        );

        if buffer_memory_mb > 500.0 {
            eprintln!(
                "WARNING: Replay buffer will use ~{:.0} MB of memory",
                buffer_memory_mb
            );
            eprintln!(
                "  This may cause GPU memory allocation errors on systems with limited VRAM."
            );
            eprintln!(
                "  Consider using smaller --update-frequency (current: {})",
                self.ppo_config.update_frequency
            );
        }

        let batch_memory_mb = estimate_batch_memory_mb(
            self.ppo_config.batch_size,
            self.game_config.grid_height,
            self.game_config.grid_width,
        );

        if batch_memory_mb > 50.0 {
            eprintln!(
                "WARNING: Each training batch will use ~{:.0} MB of GPU memory",
                batch_memory_mb
            );
            eprintln!("  This may cause errors on GPUs with less than 4 GB VRAM.");
            eprintln!(
                "  Consider using smaller --batch-size (current: {})",
                self.ppo_config.batch_size
            );
        }

        Ok(())
    }
}

/// Estimate memory usage of replay buffer in MB
fn estimate_buffer_memory_mb(capacity: usize, height: usize, width: usize) -> f32 {
    // Each transition stores:
    // - observation: [4, H, W] f32
    // - action: usize
    // - log_prob: f32
    // - reward: f32
    // - value: f32
    // - done: bool
    // - advantage: f32 (after GAE)
    // - return: f32 (after GAE)

    let obs_bytes = 4 * height * width * 4; // [4, H, W] × f32 (4 bytes)
    let metadata_bytes = 8 + 4 + 4 + 4 + 1 + 4 + 4; // Other fields
    let bytes_per_transition = obs_bytes + metadata_bytes;
    let total_bytes = capacity * bytes_per_transition;

    total_bytes as f32 / (1024.0 * 1024.0)
}

/// Estimate memory usage of a single batch in MB
fn estimate_batch_memory_mb(batch_size: usize, height: usize, width: usize) -> f32 {
    // Batch tensor: [batch_size, 4, H, W] f32
    let batch_bytes = batch_size * 4 * height * width * 4;
    batch_bytes as f32 / (1024.0 * 1024.0)
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

    /// Performance metrics tracker
    perf_metrics: PerformanceMetrics,

    /// Training configuration
    config: TrainConfig,

    /// Current episode number
    current_episode: usize,

    /// Total steps across all episodes
    total_steps: usize,

    /// Training start time (for elapsed time calculation)
    start_time: Instant,
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

        // Initialize performance metrics tracker
        let perf_metrics = PerformanceMetrics::new(
            config.perf_metrics_enabled,
            config.perf_fine_grained,
        );

        Self {
            agent,
            env,
            stats,
            perf_metrics,
            config,
            current_episode: 0,
            total_steps: 0,
            start_time: Instant::now(),
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

        // Validate configuration before starting training
        self.config.validate()?;

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

        let timestamp = Self::format_timestamp();
        let elapsed = Self::format_elapsed_time(self.start_time.elapsed());

        println!("\n{}", "=".repeat(70));
        println!("Training Complete!");
        println!("{}", "=".repeat(70));
        println!("[{}] [Total Elapsed: {}]", timestamp, elapsed);
        println!("Final model saved to: {:?}", self.config.save_path);
        println!("\nFinal Statistics:");
        println!("{}", self.stats.format_summary());

        // Print detailed performance metrics if enabled
        if self.config.perf_metrics_enabled {
            println!("\n{}", "=".repeat(70));
            println!("Performance Metrics:");
            println!("{}", "=".repeat(70));
            println!("{}", self.perf_metrics.format_detailed());

            // Export to CSV if path provided
            if let Some(ref path) = self.config.perf_output_path {
                self.perf_metrics.export_csv(path)?;
                println!("\nPerformance metrics exported to: {:?}", path);
            }
        }

        println!("{}", "=".repeat(70));

        Ok(())
    }

    /// Run a single training episode
    ///
    /// Collects experiences by running the agent in the environment, storing
    /// transitions in the buffer. When the buffer is full, performs PPO updates.
    ///
    /// Episodes terminate when:
    /// - The snake dies (wall/self collision)
    /// - Maximum steps per episode is reached (prevents infinite loops)
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// - Total episode reward
    /// - Number of steps in the episode
    /// - Final score (food eaten)
    fn run_episode(&mut self) -> Result<(f32, usize, u32)> {
        let _episode_timer = self.perf_metrics.start_scope(TimingKey::Episode);

        let mut obs = self.env.reset();
        let mut episode_reward = 0.0;
        let mut episode_steps = 0;
        let mut done = false;

        while !done && episode_steps < self.config.max_steps_per_episode {
            // Select action (optionally timed for fine-grained metrics)
            let (action, log_prob, value) = if self.config.perf_fine_grained {
                let _timer = self.perf_metrics.start_scope(TimingKey::NetworkForward);
                self.agent.select_action(obs.clone())
            } else {
                self.agent.select_action(obs.clone())
            };

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
                let _update_timer = self.perf_metrics.start_scope(TimingKey::PpoUpdate);

                // Get last value for bootstrapping
                let (_, _, last_value) = self.agent.select_action(obs.clone());

                // Perform PPO update
                let (policy_loss, value_loss, entropy, _total_loss) =
                    self.agent.update(last_value, done);

                // Record training metrics
                self.stats.record_update(policy_loss, value_loss, entropy);
            }
        }

        let episode_score = self.env.state().score;

        Ok((episode_reward, episode_steps, episode_score))
    }

    /// Save a checkpoint of the current model
    fn save_checkpoint(&mut self) -> Result<()> {
        let _timer = self.perf_metrics.start_scope(TimingKey::ModelSave);

        let checkpoint_path = self
            .config
            .save_path
            .parent()
            .unwrap_or(Path::new("."))
            .join(format!("checkpoint_ep{}.bin", self.current_episode + 1));

        save_model(&self.agent, &checkpoint_path)
            .with_context(|| format!("Failed to save checkpoint to {:?}", checkpoint_path))?;

        println!("  Checkpoint saved: {:?}", checkpoint_path);

        Ok(())
    }

    /// Save the final trained model
    fn save_model(&mut self) -> Result<()> {
        let _timer = self.perf_metrics.start_scope(TimingKey::ModelSave);

        save_model(&self.agent, &self.config.save_path).with_context(|| {
            format!("Failed to save final model to {:?}", self.config.save_path)
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
        println!(
            "Max steps per episode: {}",
            self.config.max_steps_per_episode
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
        println!(
            "Checkpoints: Every {} episodes",
            self.config.checkpoint_frequency
        );
        println!("Logging: Every {} episodes", self.config.log_frequency);
        println!("Save path: {:?}", self.config.save_path);
        println!("{}", "=".repeat(70));
        println!();
    }

    /// Print training progress
    fn print_progress(&self, episode: usize) {
        let timestamp = Self::format_timestamp();
        let elapsed = Self::format_elapsed_time(self.start_time.elapsed());

        println!(
            "[{}] [Elapsed: {}] [Episode {}/{}] {}",
            timestamp,
            elapsed,
            episode,
            self.config.num_episodes,
            self.stats.format_summary()
        );

        // Print performance metrics if enabled and we have enough data
        if self.config.perf_metrics_enabled && episode >= self.config.log_frequency {
            println!("\n  {}\n", self.perf_metrics.format_summary());
        }
    }

    /// Format current timestamp as HH:MM:SS
    fn format_timestamp() -> String {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();

        let total_seconds = now.as_secs();
        let hours = (total_seconds / 3600) % 24;
        let minutes = (total_seconds / 60) % 60;
        let seconds = total_seconds % 60;

        format!("{:02}:{:02}:{:02}", hours, minutes, seconds)
    }

    /// Format elapsed time as HH:MM:SS
    fn format_elapsed_time(duration: std::time::Duration) -> String {
        let total_seconds = duration.as_secs();
        let hours = total_seconds / 3600;
        let minutes = (total_seconds / 60) % 60;
        let seconds = total_seconds % 60;

        format!("{:02}:{:02}:{:02}", hours, minutes, seconds)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rl::{TrainingBackend, default_device};
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
