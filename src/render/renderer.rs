use ratatui::{
    Frame,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Paragraph},
};

use crate::game::{GameState, Position};
use crate::metrics::GameMetrics;

pub struct Renderer;

impl Renderer {
    pub fn new() -> Self {
        Self
    }

    pub fn render(&self, frame: &mut Frame, state: &GameState, metrics: &GameMetrics) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3), // Header
                Constraint::Min(0),    // Game area
                Constraint::Length(3), // Footer
            ])
            .split(frame.area());

        // Render header with basic stats
        let stats = self.render_stats(chunks[0], state, metrics);
        frame.render_widget(stats, chunks[0]);

        // Center the game grid horizontally
        let game_area = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(10),
                Constraint::Percentage(80),
                Constraint::Percentage(10),
            ])
            .split(chunks[1])[1];

        // Render game grid or game over screen
        if state.is_alive {
            let grid = self.render_grid(game_area, state);
            frame.render_widget(grid, game_area);
        } else {
            let game_over = self.render_game_over(game_area, state);
            frame.render_widget(game_over, game_area);
        }

        // Render footer with controls
        let controls = self.render_controls(chunks[2]);
        frame.render_widget(controls, chunks[2]);
    }

    fn render_grid(&self, _area: Rect, state: &GameState) -> Paragraph<'_> {
        let mut lines = Vec::new();

        for y in 0..state.grid_height {
            let mut spans = Vec::new();

            for x in 0..state.grid_width {
                let pos = Position::new(x as i32, y as i32);

                let cell = if pos == state.snake.head() {
                    // Snake head - distinct color
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
                        Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
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
                    .title(" Snake "),
            )
            .alignment(Alignment::Center)
    }

    fn render_stats(&self, _area: Rect, state: &GameState, metrics: &GameMetrics) -> Paragraph<'_> {
        let text = vec![Line::from(vec![
            Span::styled("Score: ", Style::default().fg(Color::Yellow)),
            Span::styled(
                state.score.to_string(),
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("    "),
            Span::styled("Steps: ", Style::default().fg(Color::Yellow)),
            Span::styled(state.steps.to_string(), Style::default().fg(Color::White)),
            Span::raw("    "),
            Span::styled("Time: ", Style::default().fg(Color::Yellow)),
            Span::styled(metrics.format_time(), Style::default().fg(Color::White)),
        ])];

        Paragraph::new(text).alignment(Alignment::Center)
    }

    fn render_game_over(&self, _area: Rect, state: &GameState) -> Paragraph<'_> {
        let text = vec![
            Line::from(""),
            Line::from(vec![Span::styled(
                "GAME OVER",
                Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
            )]),
            Line::from(""),
            Line::from(vec![
                Span::styled("Final Score: ", Style::default().fg(Color::Yellow)),
                Span::styled(
                    state.score.to_string(),
                    Style::default()
                        .fg(Color::White)
                        .add_modifier(Modifier::BOLD),
                ),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled("Press ", Style::default().fg(Color::Gray)),
                Span::styled(
                    "R",
                    Style::default()
                        .fg(Color::Green)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(" to restart or ", Style::default().fg(Color::Gray)),
                Span::styled(
                    "Q",
                    Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
                ),
                Span::styled(" to quit", Style::default().fg(Color::Gray)),
            ]),
        ];

        Paragraph::new(text).alignment(Alignment::Center).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Red)),
        )
    }

    fn render_controls(&self, _area: Rect) -> Paragraph<'_> {
        let text = vec![Line::from(vec![
            Span::styled("↑↓←→", Style::default().fg(Color::Cyan)),
            Span::raw(" or "),
            Span::styled("WASD", Style::default().fg(Color::Cyan)),
            Span::raw(" to move | "),
            Span::styled("Q", Style::default().fg(Color::Red)),
            Span::raw(" to quit"),
        ])];

        Paragraph::new(text).alignment(Alignment::Center)
    }
}

impl Default for Renderer {
    fn default() -> Self {
        Self::new()
    }
}
