use anyhow::{Context, Result};
use crossterm::{
    event::{Event, EventStream, KeyEventKind},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use futures::StreamExt;
use ratatui::{Terminal, backend::CrosstermBackend};
use std::io::{Stderr, stderr};
use std::time::Duration;
use tokio::time::interval;

use crate::game::{Action, Direction, GameConfig, GameEngine, GameState};
use crate::input::{InputHandler, KeyAction};
use crate::metrics::GameMetrics;
use crate::render::Renderer;

pub struct HumanMode {
    engine: GameEngine,
    state: GameState,
    metrics: GameMetrics,
    renderer: Renderer,
    input_handler: InputHandler,
    should_quit: bool,
    pending_direction: Option<Direction>,
}

impl HumanMode {
    pub fn new(config: GameConfig) -> Self {
        let mut engine = GameEngine::new(config);
        let state = engine.reset();

        Self {
            engine,
            state,
            metrics: GameMetrics::new(),
            renderer: Renderer::new(),
            input_handler: InputHandler::new(),
            should_quit: false,
            pending_direction: None,
        }
    }

    pub async fn run(&mut self) -> Result<()> {
        // Setup terminal
        enable_raw_mode().context("Failed to enable raw mode")?;
        let mut stderr = stderr();
        execute!(stderr, EnterAlternateScreen).context("Failed to enter alternate screen")?;
        let backend = CrosstermBackend::new(stderr);
        let mut terminal = Terminal::new(backend).context("Failed to create terminal")?;
        terminal.hide_cursor().context("Failed to hide cursor")?;
        terminal.clear().context("Failed to clear terminal")?;

        // Run game loop with cleanup
        let result = self.run_game_loop(&mut terminal).await;

        // Cleanup terminal
        self.cleanup_terminal(&mut terminal)?;

        result
    }

    async fn run_game_loop(
        &mut self,
        terminal: &mut Terminal<CrosstermBackend<Stderr>>,
    ) -> Result<()> {
        let mut event_stream = EventStream::new();

        // Game ticks at 8 Hz (125ms per tick)
        let tick_interval = Duration::from_millis(125);
        let mut tick_timer = interval(tick_interval);

        // Render at 30 FPS (33ms per frame)
        let render_interval = Duration::from_millis(33);
        let mut render_timer = interval(render_interval);

        loop {
            tokio::select! {
                // Handle terminal events
                maybe_event = event_stream.next() => {
                    if let Some(Ok(event)) = maybe_event {
                        self.handle_event(event)?;
                    }
                }

                // Game logic tick
                _ = tick_timer.tick() => {
                    if self.state.is_alive {
                        self.update_game()?;
                    }
                }

                // Render frame
                _ = render_timer.tick() => {
                    self.metrics.update();
                    terminal.draw(|frame| {
                        self.renderer.render(frame, &self.state, &self.metrics);
                    }).context("Failed to draw frame")?;
                }

                // Handle Ctrl+C
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

    fn handle_event(&mut self, event: Event) -> Result<()> {
        if let Event::Key(key) = event {
            // Only process key press events, not release
            if key.kind != KeyEventKind::Press {
                return Ok(());
            }

            let action = self.input_handler.handle_key_event(key);

            match action {
                KeyAction::GameAction(Action::Move(dir)) => {
                    self.pending_direction = Some(dir);
                }
                KeyAction::GameAction(Action::Continue) => {
                    // No action needed
                }
                KeyAction::Restart => {
                    self.reset_game();
                }
                KeyAction::Quit => {
                    self.should_quit = true;
                }
                KeyAction::None => {}
            }
        }

        Ok(())
    }

    fn update_game(&mut self) -> Result<()> {
        let action = self
            .pending_direction
            .map(Action::Move)
            .unwrap_or(Action::Continue);

        self.pending_direction = None;

        let result = self.engine.step(&mut self.state, action);

        // Track game over
        if result.terminated && !self.state.is_alive {
            self.metrics.on_game_over(self.state.score);
        }

        Ok(())
    }

    fn reset_game(&mut self) {
        self.state = self.engine.reset();
        self.metrics.on_game_start();
        self.pending_direction = None;
    }

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_game_initialization() {
        let config = GameConfig::default();
        let mode = HumanMode::new(config);
        assert!(mode.state.is_alive);
        assert_eq!(mode.state.score, 0);
    }

    #[test]
    fn test_game_reset() {
        let mut mode = HumanMode::new(GameConfig::default());
        mode.state.score = 10;
        mode.state.is_alive = false;
        mode.reset_game();
        assert_eq!(mode.state.score, 0);
        assert!(mode.state.is_alive);
    }
}
