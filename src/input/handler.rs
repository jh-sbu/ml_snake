use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use crate::game::{Action, Direction};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KeyAction {
    GameAction(Action),
    Restart,
    Quit,
    None,
}

pub struct InputHandler;

impl InputHandler {
    pub fn new() -> Self {
        Self
    }

    pub fn handle_key_event(&self, key: KeyEvent) -> KeyAction {
        // Handle Ctrl+C
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
            return KeyAction::Quit;
        }

        match key.code {
            // Movement - Arrow keys
            KeyCode::Up => KeyAction::GameAction(Action::Move(Direction::Up)),
            KeyCode::Down => KeyAction::GameAction(Action::Move(Direction::Down)),
            KeyCode::Left => KeyAction::GameAction(Action::Move(Direction::Left)),
            KeyCode::Right => KeyAction::GameAction(Action::Move(Direction::Right)),

            // Movement - WASD
            KeyCode::Char('w') | KeyCode::Char('W') => {
                KeyAction::GameAction(Action::Move(Direction::Up))
            }
            KeyCode::Char('s') | KeyCode::Char('S') => {
                KeyAction::GameAction(Action::Move(Direction::Down))
            }
            KeyCode::Char('a') | KeyCode::Char('A') => {
                KeyAction::GameAction(Action::Move(Direction::Left))
            }
            KeyCode::Char('d') | KeyCode::Char('D') => {
                KeyAction::GameAction(Action::Move(Direction::Right))
            }

            // Controls
            KeyCode::Char('q') | KeyCode::Char('Q') | KeyCode::Esc => KeyAction::Quit,
            KeyCode::Char('r') | KeyCode::Char('R') => KeyAction::Restart,

            _ => KeyAction::None,
        }
    }
}

impl Default for InputHandler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arrow_keys() {
        let handler = InputHandler::new();

        let up = KeyEvent::new(KeyCode::Up, KeyModifiers::NONE);
        assert_eq!(
            handler.handle_key_event(up),
            KeyAction::GameAction(Action::Move(Direction::Up))
        );

        let down = KeyEvent::new(KeyCode::Down, KeyModifiers::NONE);
        assert_eq!(
            handler.handle_key_event(down),
            KeyAction::GameAction(Action::Move(Direction::Down))
        );

        let left = KeyEvent::new(KeyCode::Left, KeyModifiers::NONE);
        assert_eq!(
            handler.handle_key_event(left),
            KeyAction::GameAction(Action::Move(Direction::Left))
        );

        let right = KeyEvent::new(KeyCode::Right, KeyModifiers::NONE);
        assert_eq!(
            handler.handle_key_event(right),
            KeyAction::GameAction(Action::Move(Direction::Right))
        );
    }

    #[test]
    fn test_wasd_keys() {
        let handler = InputHandler::new();

        let w = KeyEvent::new(KeyCode::Char('w'), KeyModifiers::NONE);
        assert_eq!(
            handler.handle_key_event(w),
            KeyAction::GameAction(Action::Move(Direction::Up))
        );

        let a = KeyEvent::new(KeyCode::Char('a'), KeyModifiers::NONE);
        assert_eq!(
            handler.handle_key_event(a),
            KeyAction::GameAction(Action::Move(Direction::Left))
        );

        let s = KeyEvent::new(KeyCode::Char('s'), KeyModifiers::NONE);
        assert_eq!(
            handler.handle_key_event(s),
            KeyAction::GameAction(Action::Move(Direction::Down))
        );

        let d = KeyEvent::new(KeyCode::Char('d'), KeyModifiers::NONE);
        assert_eq!(
            handler.handle_key_event(d),
            KeyAction::GameAction(Action::Move(Direction::Right))
        );
    }

    #[test]
    fn test_wasd_uppercase() {
        let handler = InputHandler::new();

        let w_upper = KeyEvent::new(KeyCode::Char('W'), KeyModifiers::SHIFT);
        assert_eq!(
            handler.handle_key_event(w_upper),
            KeyAction::GameAction(Action::Move(Direction::Up))
        );
    }

    #[test]
    fn test_quit_keys() {
        let handler = InputHandler::new();

        let q = KeyEvent::new(KeyCode::Char('q'), KeyModifiers::NONE);
        assert_eq!(handler.handle_key_event(q), KeyAction::Quit);

        let q_upper = KeyEvent::new(KeyCode::Char('Q'), KeyModifiers::SHIFT);
        assert_eq!(handler.handle_key_event(q_upper), KeyAction::Quit);

        let esc = KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE);
        assert_eq!(handler.handle_key_event(esc), KeyAction::Quit);
    }

    #[test]
    fn test_restart_key() {
        let handler = InputHandler::new();

        let r = KeyEvent::new(KeyCode::Char('r'), KeyModifiers::NONE);
        assert_eq!(handler.handle_key_event(r), KeyAction::Restart);

        let r_upper = KeyEvent::new(KeyCode::Char('R'), KeyModifiers::SHIFT);
        assert_eq!(handler.handle_key_event(r_upper), KeyAction::Restart);
    }

    #[test]
    fn test_unknown_key() {
        let handler = InputHandler::new();

        let x = KeyEvent::new(KeyCode::Char('x'), KeyModifiers::NONE);
        assert_eq!(handler.handle_key_event(x), KeyAction::None);
    }

    #[test]
    fn test_ctrl_c() {
        let handler = InputHandler::new();

        let ctrl_c = KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL);
        assert_eq!(handler.handle_key_event(ctrl_c), KeyAction::Quit);
    }
}
