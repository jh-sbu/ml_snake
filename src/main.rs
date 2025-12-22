use anyhow::Result;
use clap::{Parser, ValueEnum};
use ml_snake::game::GameConfig;
use ml_snake::modes::{HumanMode, TrainConfig, TrainMode, VisualizeMode};
use ml_snake::rl::{gpu_available, PPOConfig};
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

    // Backend selection arguments
    /// Backend to use: cpu, gpu, or auto (default: auto)
    #[arg(long, default_value = "auto", value_enum)]
    backend: BackendChoice,

    /// GPU device ID (0 = first discrete GPU, 1 = second, etc.)
    #[arg(long)]
    gpu_device: Option<usize>,

    // Performance metrics arguments
    /// Enable performance metrics collection (train mode only)
    #[arg(long)]
    perf: bool,

    /// Enable fine-grained performance metrics - higher overhead (train mode only)
    #[arg(long)]
    perf_fine_grained: bool,

    /// Path to export performance metrics as CSV (train mode only)
    #[arg(long)]
    perf_output: Option<PathBuf>,
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

#[derive(Clone, Copy, ValueEnum)]
enum BackendChoice {
    /// Use CPU backend (NdArray)
    Cpu,
    /// Use GPU backend (Wgpu)
    Gpu,
    /// Auto-detect best available backend
    Auto,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Create game configuration from CLI arguments
    let config = GameConfig::new(cli.width, cli.height);

    // Determine backend based on CLI
    let use_gpu = match cli.backend {
        BackendChoice::Cpu => {
            eprintln!("Using CPU backend (NdArray)");
            false
        }
        BackendChoice::Gpu => {
            if !gpu_available() {
                anyhow::bail!(
                    "GPU backend requested but GPU is not available.\n\
                     Please ensure you have:\n\
                     - Up-to-date graphics drivers installed\n\
                     - Vulkan/Metal/DirectX12 support"
                );
            }
            eprintln!("Using GPU backend (Wgpu)");
            true
        }
        BackendChoice::Auto => {
            if gpu_available() {
                eprintln!("Auto-detected GPU, using Wgpu backend");
                true
            } else {
                eprintln!("No GPU detected, using CPU backend (NdArray)");
                false
            }
        }
    };

    // Dispatch to appropriate mode and backend
    match (&cli.mode, use_gpu) {
        (Mode::Human, _) => {
            // Human mode doesn't use ML, no backend needed
            let mut human_mode = HumanMode::new(config);
            human_mode.run().await?;
        }
        (Mode::Train, false) => {
            run_train_cpu(&cli, config)?;
        }
        (Mode::Train, true) => {
            run_train_gpu(&cli, config)?;
        }
        (Mode::Visualize, false) => {
            run_visualize_cpu(&cli, config).await?;
        }
        (Mode::Visualize, true) => {
            run_visualize_gpu(&cli, config).await?;
        }
    }

    Ok(())
}

fn run_train_cpu(cli: &Cli, config: GameConfig) -> Result<()> {
    use ml_snake::rl::cpu;

    let device = cpu::device();

    // Create training configuration with CPU-optimized batch size
    let mut train_config = TrainConfig::new(cli.episodes, cli.save_path.clone());
    train_config.checkpoint_frequency = cli.checkpoint_freq;
    train_config.log_frequency = cli.log_freq;
    train_config.max_steps_per_episode = cli.max_steps;
    train_config.game_config = config;
    train_config.ppo_config = PPOConfig::default(); // batch_size=32 for CPU
    train_config.perf_metrics_enabled = cli.perf;
    train_config.perf_fine_grained = cli.perf_fine_grained;
    train_config.perf_output_path = cli.perf_output.clone();

    let mut train_mode = TrainMode::<cpu::TrainingBackend>::new(train_config, device);
    train_mode.run()
}

fn run_train_gpu(cli: &Cli, config: GameConfig) -> Result<()> {
    use ml_snake::rl::gpu;

    let device = gpu::device(cli.gpu_device);

    // Create training configuration with GPU-optimized batch size
    let mut train_config = TrainConfig::new(cli.episodes, cli.save_path.clone());
    train_config.checkpoint_frequency = cli.checkpoint_freq;
    train_config.log_frequency = cli.log_freq;
    train_config.max_steps_per_episode = cli.max_steps;
    train_config.game_config = config;
    // Auto-adjust batch size for GPU for better utilization
    train_config.ppo_config = PPOConfig {
        batch_size: 256,        // vs 32 on CPU
        update_frequency: 4096, // vs 2048 on CPU
        ..Default::default()
    };
    train_config.perf_metrics_enabled = cli.perf;
    train_config.perf_fine_grained = cli.perf_fine_grained;
    train_config.perf_output_path = cli.perf_output.clone();

    let mut train_mode = TrainMode::<gpu::TrainingBackend>::new(train_config, device);
    train_mode.run()
}

async fn run_visualize_cpu(cli: &Cli, config: GameConfig) -> Result<()> {
    use ml_snake::rl::cpu;

    let device = cpu::device();
    let mut visualize_mode =
        VisualizeMode::<cpu::InferenceBackend>::new(&cli.model_path, config, device)?;
    visualize_mode.run().await
}

async fn run_visualize_gpu(cli: &Cli, config: GameConfig) -> Result<()> {
    use ml_snake::rl::gpu;

    let device = gpu::device(cli.gpu_device);
    let mut visualize_mode =
        VisualizeMode::<gpu::InferenceBackend>::new(&cli.model_path, config, device)?;
    visualize_mode.run().await
}
