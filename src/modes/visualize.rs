//! Visualization mode for watching trained agents
//!
//! This module implements a TUI-based visualization mode that loads a trained
//! model and displays the agent playing Snake. Users can control playback speed,
//! pause, and reset episodes.
//!
//! # Controls
//!
//! - Space: Pause/unpause
//! - R: Reset episode
//! - 1-4: Speed control (1=slow, 2=normal, 3=fast, 4=very fast)
//! - Q/Esc: Quit
//!
//! # Example
//!
//! ```rust,ignore
//! use ml_snake::modes::VisualizeMode;
//! use ml_snake::game::GameConfig;
//! use ml_snake::rl::{default_device, InferenceBackend};
//! use std::path::Path;
//!
//! let config = GameConfig::default();
//! let device = default_device();
//! let mut visualize_mode = VisualizeMode::<InferenceBackend>::new(
//!     Path::new("models/snake.bin"),
//!     config,
//!     device,
//! )?;
//! visualize_mode.run().await?;
//! ```

use anyhow::{Context, Result};
use burn::module::AutodiffModule;
use burn::tensor::{
    activation::softmax,
    backend::Backend,
    Tensor,
};
use crossterm::{
    event::{Event, EventStream, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use futures::StreamExt;
use ratatui::{backend::CrosstermBackend, Terminal};
use std::{
    io::{stderr, Stderr},
    iter::repeat,
    path::Path,
    time::Duration,
};
use tokio::time::{interval, Interval};

use crate::game::GameConfig;
use crate::render::Renderer;
use crate::rl::{load_network, ActorCriticNetwork, ModelMetadata, SnakeEnvironment};

/// Visualization speed settings
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisualizationSpeed {
    /// Slow: 2 Hz (500ms per step)
    Slow,
    /// Normal: 8 Hz (125ms per step) - same as human mode
    Normal,
    /// Fast: 20 Hz (50ms per step)
    Fast,
    /// Very Fast: 60 Hz (16ms per step)
    VeryFast,
}

impl VisualizationSpeed {
    /// Get the tick interval for this speed
    fn tick_interval(&self) -> Duration {
        match self {
            Self::Slow => Duration::from_millis(500),
            Self::Normal => Duration::from_millis(125),
            Self::Fast => Duration::from_millis(50),
            Self::VeryFast => Duration::from_millis(16),
        }
    }

    /// Get a string representation of the speed
    fn as_str(&self) -> &'static str {
        match self {
            Self::Slow => "Slow",
            Self::Normal => "Normal",
            Self::Fast => "Fast",
            Self::VeryFast => "Very Fast",
        }
    }
}

/// Visualization mode for watching trained agents
pub struct VisualizeMode<B: Backend> {
    /// Trained neural network (in inference mode)
    network: ActorCriticNetwork<B>,

    /// Snake environment
    env: SnakeEnvironment<B>,

    /// Renderer for TUI display
    renderer: Renderer,

    /// Model metadata
    metadata: ModelMetadata,

    /// Whether to quit the visualization
    should_quit: bool,

    /// Whether playback is paused
    paused: bool,

    /// Current playback speed
    speed: VisualizationSpeed,

    /// Number of episodes completed
    episode_count: usize,
}

impl<B: Backend> VisualizeMode<B> {
    /// Create a new visualization mode
    ///
    /// Loads a trained model from the specified path and initializes the
    /// visualization environment.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to the saved model file
    /// * `config` - Game configuration (must match training config)
    /// * `device` - Device for computation
    ///
    /// # Returns
    ///
    /// A new VisualizeMode instance or an error if loading fails
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use ml_snake::modes::VisualizeMode;
    /// use ml_snake::game::GameConfig;
    /// use ml_snake::rl::{default_device, InferenceBackend};
    /// use std::path::Path;
    ///
    /// let config = GameConfig::default();
    /// let device = default_device();
    /// let visualize_mode = VisualizeMode::<InferenceBackend>::new(
    ///     Path::new("models/snake.bin"),
    ///     config,
    ///     device,
    /// )?;
    /// ```
    pub fn new(model_path: &Path, config: GameConfig, device: B::Device) -> Result<Self> {
        // Load trained network
        use burn::backend::Autodiff;
        let (network, metadata) = load_network::<Autodiff<B>>(model_path, &device)
            .with_context(|| format!("Failed to load model from {:?}", model_path))?;

        // Convert to inference mode
        let network = network.valid();

        // Print loaded model info
        println!("{}", "=".repeat(60));
        println!("Loaded Model Information");
        println!("{}", "=".repeat(60));
        println!("Model path: {:?}", model_path);
        println!("Episodes trained: {}", metadata.episodes_trained);
        println!("Training steps: {}", metadata.training_steps);
        println!("Grid size: {}x{}", metadata.grid_width, metadata.grid_height);
        println!("Version: {}", metadata.version);
        println!("{}", "=".repeat(60));
        println!();
        println!("Starting visualization...");
        println!();

        // Create environment
        let env = SnakeEnvironment::new(config, device);

        Ok(Self {
            network,
            env,
            renderer: Renderer::new(),
            metadata,
            should_quit: false,
            paused: false,
            speed: VisualizationSpeed::Normal,
            episode_count: 0,
        })
    }

    /// Run the visualization loop
    ///
    /// Sets up the terminal, runs the main visualization loop, and cleans up
    /// on exit.
    ///
    /// # Returns
    ///
    /// `Ok(())` on successful completion
    pub async fn run(&mut self) -> Result<()> {
        // Setup terminal
        enable_raw_mode().context("Failed to enable raw mode")?;
        let mut stderr = stderr();
        execute!(stderr, EnterAlternateScreen).context("Failed to enter alternate screen")?;
        let backend = CrosstermBackend::new(stderr);
        let mut terminal = Terminal::new(backend).context("Failed to create terminal")?;
        terminal.hide_cursor().context("Failed to hide cursor")?;
        terminal.clear().context("Failed to clear terminal")?;

        // Run visualization loop
        let result = self.run_visualization_loop(&mut terminal).await;

        // Cleanup terminal
        self.cleanup_terminal(&mut terminal)?;

        result
    }

    /// Main visualization loop
    async fn run_visualization_loop(
        &mut self,
        terminal: &mut Terminal<CrosstermBackend<Stderr>>,
    ) -> Result<()> {
        let mut event_stream = EventStream::new();

        // Game ticks based on speed
        let mut tick_timer = interval(self.speed.tick_interval());

        // Render at 30 FPS
        let render_interval = Duration::from_millis(33);
        let mut render_timer = interval(render_interval);

        // Reset environment
        let mut obs = self.env.reset();
        let mut done = false;

        loop {
            tokio::select! {
                // Handle keyboard input
                maybe_event = event_stream.next() => {
                    if let Some(Ok(event)) = maybe_event {
                        self.handle_event(event, &mut tick_timer)?;
                    }
                }

                // Game logic tick
                _ = tick_timer.tick() => {
                    if !self.paused {
                        if done {
                            // Auto-restart
                            obs = self.env.reset();
                            done = false;
                            self.episode_count += 1;
                        } else {
                            // Step agent
                            obs = self.step_agent(obs)?;
                            done = !self.env.state().is_alive;
                        }
                    }
                }

                // Render frame
                _ = render_timer.tick() => {
                    terminal.draw(|frame| {
                        self.render_frame(frame);
                    }).context("Failed to draw frame")?;
                }

                // Ctrl+C
                _ = tokio::signal::ctrl_c() => {
                    self.should_quit = true;
                }
            }

            if self.should_quit {
                break;
            }
        }

        Ok(())
    }

    /// Step the agent forward one action
    ///
    /// Uses the trained network to select an action (greedy policy) and steps
    /// the environment.
    fn step_agent(&mut self, obs: Tensor<B, 3>) -> Result<Tensor<B, 3>> {
        // Add batch dimension
        let obs_batch = obs.unsqueeze_dim(0); // [1, 4, H, W]

        // Forward pass
        let (action_logits, _value) = self.network.forward(obs_batch);

        // Select best action (argmax of probabilities)
        let action_probs = softmax(action_logits, 1);
        let action_idx = argmax_action(&action_probs);

        // Step environment
        let (next_obs, _reward, _done) = self.env.step(action_idx);

        Ok(next_obs)
    }

    /// Handle keyboard events
    fn handle_event(&mut self, event: Event, tick_timer: &mut Interval) -> Result<()> {
        if let Event::Key(key) = event {
            // Only process key press events
            if key.kind != KeyEventKind::Press {
                return Ok(());
            }

            match key.code {
                KeyCode::Char('q') | KeyCode::Esc => {
                    self.should_quit = true;
                }
                KeyCode::Char(' ') => {
                    self.paused = !self.paused;
                }
                KeyCode::Char('r') => {
                    // Manual reset
                    self.env.reset();
                    self.episode_count += 1;
                }
                KeyCode::Char('1') => {
                    self.change_speed(VisualizationSpeed::Slow, tick_timer);
                }
                KeyCode::Char('2') => {
                    self.change_speed(VisualizationSpeed::Normal, tick_timer);
                }
                KeyCode::Char('3') => {
                    self.change_speed(VisualizationSpeed::Fast, tick_timer);
                }
                KeyCode::Char('4') => {
                    self.change_speed(VisualizationSpeed::VeryFast, tick_timer);
                }
                _ => {}
            }
        }

        Ok(())
    }

    /// Change the visualization speed
    fn change_speed(&mut self, new_speed: VisualizationSpeed, tick_timer: &mut Interval) {
        self.speed = new_speed;
        tick_timer.reset_after(self.speed.tick_interval());
    }

    /// Render the current frame
    fn render_frame(&self, frame: &mut ratatui::Frame) {
        // Use existing renderer for game state
        // Note: This uses a dummy metrics object since we're not tracking human play metrics
        use crate::metrics::GameMetrics;
        let dummy_metrics = GameMetrics::new();

        self.renderer.render(frame, self.env.state(), &dummy_metrics);

        // TODO: Add visualization-specific overlays:
        // - "VISUALIZE MODE" indicator
        // - Episode counter
        // - Speed setting
        // - Pause indicator
        // - Controls help text
        //
        // This can be done by extending the Renderer or creating overlay widgets
    }

    /// Cleanup terminal state
    fn cleanup_terminal(
        &mut self,
        terminal: &mut Terminal<CrosstermBackend<Stderr>>,
    ) -> Result<()> {
        disable_raw_mode().context("Failed to disable raw mode")?;
        execute!(terminal.backend_mut(), LeaveAlternateScreen)
            .context("Failed to leave alternate screen")?;
        terminal.show_cursor().context("Failed to show cursor")?;
        Ok(())
    }
}

/// Select the action with highest probability (argmax)
///
/// # Arguments
///
/// * `probs` - Action probabilities [1, num_actions]
///
/// # Returns
///
/// Index of the action with highest probability
fn argmax_action<B: Backend>(probs: &Tensor<B, 2>) -> usize {
    let probs_data = probs.to_data();
    let probs_vec: Vec<f32> = probs_data.to_vec().expect("Failed to convert probs to vec");

    probs_vec
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rl::{default_device, InferenceBackend, TrainingBackend};
    use crate::rl::{save_model, ActorCriticConfig, PPOAgent, PPOConfig};
    use tempfile::TempDir;

    #[test]
    fn test_visualization_speed() {
        assert_eq!(VisualizationSpeed::Slow.tick_interval(), Duration::from_millis(500));
        assert_eq!(VisualizationSpeed::Normal.tick_interval(), Duration::from_millis(125));
        assert_eq!(VisualizationSpeed::Fast.tick_interval(), Duration::from_millis(50));
        assert_eq!(VisualizationSpeed::VeryFast.tick_interval(), Duration::from_millis(16));
    }

    #[test]
    fn test_argmax_action() {
        use burn::tensor::Tensor;

        let device = default_device();
        // Create probabilities: [0.1, 0.6, 0.2, 0.1]
        let probs = Tensor::<InferenceBackend, 2>::from_floats(
            [[0.1, 0.6, 0.2, 0.1]],
            &device,
        );

        let action = argmax_action(&probs);
        assert_eq!(action, 1); // Index 1 has highest probability (0.6)
    }

    #[test]
    fn test_visualize_mode_creation() {
        let temp_dir = TempDir::new().unwrap();
        let model_path = temp_dir.path().join("test_model.bin");

        // Create and save a test model
        let device = default_device();
        let network_config = ActorCriticConfig::new(10, 10);
        let network = network_config.init::<TrainingBackend>(&device);
        let agent = PPOAgent::new(
            network,
            PPOConfig::default(),
            10,
            10,
            device.clone(),
        );

        save_model(&agent, &model_path).unwrap();

        // Load in visualize mode
        let config = GameConfig::new(10, 10);
        let visualize_mode = VisualizeMode::<InferenceBackend>::new(
            &model_path,
            config,
            device,
        );

        assert!(visualize_mode.is_ok());
        let mode = visualize_mode.unwrap();
        assert_eq!(mode.episode_count, 0);
        assert!(!mode.paused);
        assert_eq!(mode.speed, VisualizationSpeed::Normal);
    }
}
