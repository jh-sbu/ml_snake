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
    path::Path,
    time::Duration,
};
use tokio::time::{interval, Interval};

use crate::game::GameConfig;
use crate::rl::{load_network, ActorCriticNetwork, ModelMetadata, SnakeEnvironment};

/// Agent decision information for visualization
#[derive(Debug, Clone)]
pub struct AgentInfo {
    /// Action probabilities [Up, Down, Left, Right]
    pub action_probs: [f32; 4],

    /// Critic's value estimate for current state
    pub value: f32,

    /// Index of chosen action (0-3)
    pub chosen_action: usize,
}

impl Default for AgentInfo {
    fn default() -> Self {
        Self {
            action_probs: [0.25, 0.25, 0.25, 0.25],
            value: 0.0,
            chosen_action: 0,
        }
    }
}

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

    /// Current agent decision info (for rendering)
    agent_info: AgentInfo,
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
            metadata,
            should_quit: false,
            paused: false,
            speed: VisualizationSpeed::Normal,
            episode_count: 0,
            agent_info: AgentInfo::default(),
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
                            self.agent_info = AgentInfo::default();
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
    /// the environment. Also extracts and stores agent decision info for rendering.
    fn step_agent(&mut self, obs: Tensor<B, 3>) -> Result<Tensor<B, 3>> {
        // Add batch dimension
        let obs_batch = obs.unsqueeze_dim(0); // [1, 4, H, W]

        // Forward pass (DON'T discard value!)
        let (action_logits, value) = self.network.forward(obs_batch);

        // Compute action probabilities
        let action_probs = softmax(action_logits.clone(), 1);
        let action_idx = argmax_action(&action_probs);

        // Extract data for rendering (convert Backend tensors to f32)
        let probs_data = action_probs.to_data();
        let probs_vec: Vec<f32> = probs_data
            .to_vec()
            .expect("Failed to convert action probs to vec");

        let value_data = value.to_data();
        let value_vec: Vec<f32> = value_data
            .to_vec()
            .expect("Failed to convert value to vec");

        // Store agent info for rendering
        self.agent_info = AgentInfo {
            action_probs: [probs_vec[0], probs_vec[1], probs_vec[2], probs_vec[3]],
            value: value_vec[0],
            chosen_action: action_idx,
        };

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
                    self.agent_info = AgentInfo::default();
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

    /// Render the current frame with agent insights
    fn render_frame(&self, frame: &mut ratatui::Frame) {
        use ratatui::layout::{Constraint, Direction, Layout};

        // Main layout: Header | Body | Footer
        let main_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3), // Header
                Constraint::Min(0),    // Body
                Constraint::Length(3), // Footer
            ])
            .split(frame.area());

        // Render header with visualization info
        self.render_viz_header(main_chunks[0], frame);

        // Body layout: Game (70%) | Sidebar (30%)
        let body_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(70), // Game area
                Constraint::Percentage(30), // Sidebar
            ])
            .split(main_chunks[1]);

        // Render game grid
        let grid_widget = self.render_grid_widget();
        frame.render_widget(grid_widget, body_chunks[0]);

        // Sidebar layout: Agent Info (60%) | Model Info (40%)
        let sidebar_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Percentage(60), // Agent decision
                Constraint::Percentage(40), // Model info
            ])
            .split(body_chunks[1]);

        // Render sidebar sections
        self.render_agent_info(sidebar_chunks[0], frame);
        self.render_model_info(sidebar_chunks[1], frame);

        // Render footer with controls
        self.render_viz_controls(main_chunks[2], frame);
    }

    /// Render enhanced header with visualization mode info
    fn render_viz_header(&self, area: ratatui::layout::Rect, frame: &mut ratatui::Frame) {
        use ratatui::layout::Alignment;
        use ratatui::style::{Color, Modifier, Style};
        use ratatui::text::{Line, Span};
        use ratatui::widgets::{Block, Borders, Paragraph};

        // Build status line
        let mode_indicator = Span::styled(
            "VISUALIZE MODE",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        );

        let episode_info = Span::styled(
            format!(" | Episode: {} ", self.episode_count),
            Style::default().fg(Color::Yellow),
        );

        let speed_info = Span::styled(
            format!(" | Speed: {} ", self.speed.as_str()),
            Style::default().fg(Color::Green),
        );

        let pause_indicator = if self.paused {
            Span::styled(
                " | PAUSED",
                Style::default()
                    .fg(Color::Red)
                    .add_modifier(Modifier::BOLD),
            )
        } else {
            Span::raw("")
        };

        let header = Paragraph::new(Line::from(vec![
            mode_indicator,
            episode_info,
            speed_info,
            pause_indicator,
        ]))
        .alignment(Alignment::Center)
        .block(
            Block::default()
                .borders(Borders::BOTTOM)
                .border_style(Style::default().fg(Color::DarkGray)),
        );

        frame.render_widget(header, area);
    }

    /// Render agent decision info sidebar
    fn render_agent_info(&self, area: ratatui::layout::Rect, frame: &mut ratatui::Frame) {
        use ratatui::style::{Color, Modifier, Style};
        use ratatui::text::{Line, Span};
        use ratatui::widgets::{Block, BorderType, Borders, Paragraph};

        // Value estimate with context
        let (value_color, value_label) = if self.agent_info.value > 10.0 {
            (Color::Green, "Good")
        } else if self.agent_info.value > 0.0 {
            (Color::Yellow, "Okay")
        } else {
            (Color::Red, "Poor")
        };

        let value_line = Line::from(vec![
            Span::styled("Value: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{:+.1}", self.agent_info.value),
                Style::default()
                    .fg(value_color)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!(" [{}]", value_label),
                Style::default().fg(value_color),
            ),
        ]);

        // Action probabilities with bars
        let action_names = ["↑ Up:   ", "↓ Down: ", "← Left: ", "→ Right:"];
        let mut lines = vec![
            value_line,
            Line::from(""),
            Line::styled(
                "Action Probabilities:",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
        ];

        for (idx, (name, &prob)) in action_names
            .iter()
            .zip(self.agent_info.action_probs.iter())
            .enumerate()
        {
            let is_chosen = idx == self.agent_info.chosen_action;
            let color = if is_chosen { Color::Green } else { Color::Gray };

            // Create bar chart (10 chars = 100%)
            let filled = ((prob * 10.0).round() as usize).min(10);
            let empty = 10 - filled;
            let bar = format!("{}{}", "█".repeat(filled), "░".repeat(empty));

            let check = if is_chosen { " ✓" } else { "" };

            lines.push(Line::from(vec![
                Span::styled(*name, Style::default().fg(color)),
                Span::styled(bar, Style::default().fg(color)),
                Span::styled(
                    format!(" {:3.0}%{}", prob * 100.0, check),
                    Style::default().fg(color),
                ),
            ]));
        }

        let widget = Paragraph::new(lines).block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(Color::Cyan))
                .title(" Agent Decision "),
        );

        frame.render_widget(widget, area);
    }

    /// Render model metadata info
    fn render_model_info(&self, area: ratatui::layout::Rect, frame: &mut ratatui::Frame) {
        use ratatui::style::{Color, Style};
        use ratatui::text::{Line, Span};
        use ratatui::widgets::{Block, BorderType, Borders, Paragraph};

        let lines = vec![
            Line::from(vec![
                Span::styled("Episodes: ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format!("{}", self.metadata.episodes_trained),
                    Style::default().fg(Color::White),
                ),
            ]),
            Line::from(vec![
                Span::styled("Steps: ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format!("{}", self.metadata.training_steps),
                    Style::default().fg(Color::White),
                ),
            ]),
            Line::from(vec![
                Span::styled("Grid: ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format!("{}x{}", self.metadata.grid_width, self.metadata.grid_height),
                    Style::default().fg(Color::White),
                ),
            ]),
        ];

        let widget = Paragraph::new(lines).block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(Color::Yellow))
                .title(" Model Info "),
        );

        frame.render_widget(widget, area);
    }

    /// Render visualization-specific controls
    fn render_viz_controls(&self, area: ratatui::layout::Rect, frame: &mut ratatui::Frame) {
        use ratatui::layout::Alignment;
        use ratatui::style::{Color, Style};
        use ratatui::text::{Line, Span};
        use ratatui::widgets::Paragraph;

        let text = vec![Line::from(vec![
            Span::styled("SPACE", Style::default().fg(Color::Cyan)),
            Span::raw("=Pause | "),
            Span::styled("R", Style::default().fg(Color::Cyan)),
            Span::raw("=Reset | "),
            Span::styled("1-4", Style::default().fg(Color::Cyan)),
            Span::raw("=Speed | "),
            Span::styled("Q", Style::default().fg(Color::Red)),
            Span::raw("=Quit"),
        ])];

        let widget = Paragraph::new(text).alignment(Alignment::Center);
        frame.render_widget(widget, area);
    }

    /// Create grid widget (extracted from Renderer for inline use)
    fn render_grid_widget(&self) -> ratatui::widgets::Paragraph<'static> {
        use crate::game::Position;
        use ratatui::layout::Alignment;
        use ratatui::style::{Color, Modifier, Style};
        use ratatui::text::{Line, Span};
        use ratatui::widgets::{Block, BorderType, Borders, Paragraph};

        let state = self.env.state();
        let mut lines = Vec::new();

        // Add score at top
        let score_line = Line::from(vec![
            Span::styled("Score: ", Style::default().fg(Color::Yellow)),
            Span::styled(
                state.score.to_string(),
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("  "),
            Span::styled("Steps: ", Style::default().fg(Color::Yellow)),
            Span::styled(state.steps.to_string(), Style::default().fg(Color::White)),
        ]);

        lines.push(score_line);
        lines.push(Line::from(""));

        // Render grid
        for y in 0..state.grid_height {
            let mut spans = Vec::new();

            for x in 0..state.grid_width {
                let pos = Position::new(x as i32, y as i32);

                let cell = if pos == state.snake.head() {
                    // Snake head
                    Span::styled(
                        "■ ",
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    )
                } else if state.snake.body.contains(&pos) {
                    // Snake body
                    Span::styled("□ ", Style::default().fg(Color::Green))
                } else if pos == state.food {
                    // Food
                    Span::styled(
                        "O ",
                        Style::default()
                            .fg(Color::Red)
                            .add_modifier(Modifier::BOLD),
                    )
                } else {
                    // Empty cell
                    Span::styled(". ", Style::default().fg(Color::DarkGray))
                };

                spans.push(cell);
            }

            lines.push(Line::from(spans));
        }

        Paragraph::new(lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_type(BorderType::Double)
                    .border_style(Style::default().fg(Color::White))
                    .title(" Snake Game "),
            )
            .alignment(Alignment::Center)
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
