use anyhow::Result;
use clap::{Parser, ValueEnum};
use ml_snake::game::GameConfig;
use ml_snake::modes::HumanMode;

#[derive(Parser)]
#[command(name = "ml_snake")]
#[command(version, about = "Snake game with ML capabilities")]
struct Cli {
    /// Game mode (currently only 'human' is implemented)
    #[arg(long, default_value = "human")]
    mode: Mode,

    /// Grid width
    #[arg(long, default_value = "20")]
    width: usize,

    /// Grid height
    #[arg(long, default_value = "20")]
    height: usize,
}

#[derive(Clone, ValueEnum)]
enum Mode {
    /// Play snake with keyboard controls
    Human,
    // Future modes:
    // Train,
    // Visualize,
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
    }

    Ok(())
}
