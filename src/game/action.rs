/// Direction the snake can move
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Direction {
    Up,
    Down,
    Left,
    Right,
}

impl Direction {
    /// Returns true if turning from self to other would be a 180-degree turn
    pub fn is_opposite(&self, other: Direction) -> bool {
        matches!(
            (self, other),
            (Direction::Up, Direction::Down)
                | (Direction::Down, Direction::Up)
                | (Direction::Left, Direction::Right)
                | (Direction::Right, Direction::Left)
        )
    }

    /// Returns the delta (dx, dy) for moving in this direction
    pub fn delta(&self) -> (i32, i32) {
        match self {
            Direction::Up => (0, -1),
            Direction::Down => (0, 1),
            Direction::Left => (-1, 0),
            Direction::Right => (1, 0),
        }
    }
}

/// Action that can be taken in the game
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    /// Move in a specific direction
    Move(Direction),
    /// Continue in current direction (useful for RL)
    Continue,
}

impl From<Direction> for Action {
    fn from(direction: Direction) -> Self {
        Action::Move(direction)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opposite_directions() {
        assert!(Direction::Up.is_opposite(Direction::Down));
        assert!(Direction::Down.is_opposite(Direction::Up));
        assert!(Direction::Left.is_opposite(Direction::Right));
        assert!(Direction::Right.is_opposite(Direction::Left));

        assert!(!Direction::Up.is_opposite(Direction::Left));
        assert!(!Direction::Up.is_opposite(Direction::Right));
    }

    #[test]
    fn test_direction_delta() {
        assert_eq!(Direction::Up.delta(), (0, -1));
        assert_eq!(Direction::Down.delta(), (0, 1));
        assert_eq!(Direction::Left.delta(), (-1, 0));
        assert_eq!(Direction::Right.delta(), (1, 0));
    }
}
