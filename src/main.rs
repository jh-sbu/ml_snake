use anyhow::Result;
use clap::{Parser, ValueEnum};
use ml_snake::game::GameConfig;
use ml_snake::modes::{HumanMode, TrainConfig, TrainMode, VisualizeMode};
use ml_snake::rl::{InferenceBackend, PPOConfig, TrainingBackend, default_device};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "ml_snake")]
#[command(version, about = "Snake game with ML capabilities")]
struct Cli {
    /// Game mode
    #[arg(long, default_value = "human")]
    mode: Mode,

    /// Grid width
    #[arg(long, default_value = "20")]
    width: usize,

    /// Grid height
    #[arg(long, default_value = "20")]
    height: usize,

    // Training mode arguments
    /// Number of episodes to train (train mode only)
    #[arg(long, default_value = "10000")]
    episodes: usize,

    /// Path to save trained model (train mode only)
    #[arg(long, default_value = "models/snake.bin")]
    save_path: PathBuf,

    /// Checkpoint save frequency in episodes (train mode only)
    #[arg(long, default_value = "1000")]
    checkpoint_freq: usize,

    /// Log progress frequency in episodes (train mode only)
    #[arg(long, default_value = "100")]
    log_freq: usize,

    /// Maximum steps per episode - prevents infinite loops (train mode only)
    #[arg(long, default_value = "1000")]
    max_steps: usize,

    // Visualization mode arguments
    /// Path to trained model (visualize mode only)
    #[arg(long, default_value = "models/snake.bin")]
    model_path: PathBuf,
}

#[derive(Clone, ValueEnum)]
enum Mode {
    /// Play snake with keyboard controls
    Human,
    /// Train RL agent with PPO
    Train,
    /// Visualize trained agent playing
    Visualize,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Create game configuration from CLI arguments
    let config = GameConfig::new(cli.width, cli.height);

    // Dispatch to appropriate mode
    match cli.mode {
        Mode::Human => {
            let mut human_mode = HumanMode::new(config);
            human_mode.run().await?;
        }
        Mode::Train => {
            // Create training configuration
            let mut train_config = TrainConfig::new(cli.episodes, cli.save_path);
            train_config.checkpoint_frequency = cli.checkpoint_freq;
            train_config.log_frequency = cli.log_freq;
            train_config.max_steps_per_episode = cli.max_steps;
            train_config.game_config = config;
            train_config.ppo_config = PPOConfig::default();

            // Create device and run training
            let device = default_device();
            let mut train_mode = TrainMode::<TrainingBackend>::new(train_config, device);
            train_mode.run()?;
        }
        Mode::Visualize => {
            // Create device and run visualization
            let device = default_device();
            let mut visualize_mode =
                VisualizeMode::<InferenceBackend>::new(&cli.model_path, config, device)?;
            visualize_mode.run().await?;
        }
    }

    Ok(())
}
