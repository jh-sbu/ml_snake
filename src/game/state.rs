use super::action::Direction;

/// A position on the game grid
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Position {
    pub x: i32,
    pub y: i32,
}

impl Position {
    pub fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }

    /// Move position by delta
    pub fn moved_by(&self, dx: i32, dy: i32) -> Self {
        Self {
            x: self.x + dx,
            y: self.y + dy,
        }
    }

    /// Move position in a direction
    pub fn moved_in_direction(&self, direction: Direction) -> Self {
        let (dx, dy) = direction.delta();
        self.moved_by(dx, dy)
    }
}

/// The snake in the game
#[derive(Debug, Clone, PartialEq)]
pub struct Snake {
    /// Body segments, with head at index 0
    pub body: Vec<Position>,
    /// Current direction of movement
    pub direction: Direction,
}

impl Snake {
    /// Create a new snake with given starting position and direction
    pub fn new(head: Position, direction: Direction, length: usize) -> Self {
        let mut body = vec![head];

        // Add initial body segments behind the head
        let (dx, dy) = direction.delta();
        let (back_dx, back_dy) = (-dx, -dy);

        for i in 1..length {
            let prev = body[i - 1];
            body.push(prev.moved_by(back_dx, back_dy));
        }

        Self { body, direction }
    }

    /// Get the head position
    pub fn head(&self) -> Position {
        self.body[0]
    }

    /// Get the tail position (last segment)
    pub fn tail(&self) -> Position {
        *self.body.last().unwrap()
    }

    /// Get body segments (excluding head)
    pub fn body_segments(&self) -> &[Position] {
        &self.body[1..]
    }

    /// Check if position collides with snake body (excluding head)
    pub fn collides_with_body(&self, pos: Position) -> bool {
        self.body_segments().contains(&pos)
    }

    /// Move snake in current direction, growing if should_grow is true
    pub fn move_snake(&mut self, should_grow: bool) {
        let new_head = self.head().moved_in_direction(self.direction);
        self.body.insert(0, new_head);

        if !should_grow {
            self.body.pop();
        }
    }

    /// Get the length of the snake
    pub fn len(&self) -> usize {
        self.body.len()
    }

    /// Check if the snake is empty (should never happen in practice)
    pub fn is_empty(&self) -> bool {
        self.body.is_empty()
    }
}

/// Type of collision that occurred
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollisionType {
    /// Snake hit a wall
    Wall,
    /// Snake hit itself
    SelfCollision,
}

/// Complete game state
#[derive(Debug, Clone, PartialEq)]
pub struct GameState {
    pub snake: Snake,
    pub food: Position,
    pub grid_width: usize,
    pub grid_height: usize,
    pub score: u32,
    pub steps: u32,
    pub is_alive: bool,
}

impl GameState {
    /// Create a new game state
    pub fn new(snake: Snake, food: Position, grid_width: usize, grid_height: usize) -> Self {
        Self {
            snake,
            food,
            grid_width,
            grid_height,
            score: 0,
            steps: 0,
            is_alive: true,
        }
    }

    /// Check if a position is within the grid bounds
    pub fn is_in_bounds(&self, pos: Position) -> bool {
        pos.x >= 0
            && pos.x < self.grid_width as i32
            && pos.y >= 0
            && pos.y < self.grid_height as i32
    }

    /// Check if a position is occupied by the snake
    pub fn is_occupied_by_snake(&self, pos: Position) -> bool {
        self.snake.body.contains(&pos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_movement() {
        let pos = Position::new(5, 5);
        assert_eq!(pos.moved_by(1, 0), Position::new(6, 5));
        assert_eq!(pos.moved_by(-1, 0), Position::new(4, 5));
        assert_eq!(pos.moved_by(0, 1), Position::new(5, 6));
        assert_eq!(pos.moved_by(0, -1), Position::new(5, 4));
    }

    #[test]
    fn test_snake_creation() {
        let snake = Snake::new(Position::new(5, 5), Direction::Right, 3);
        assert_eq!(snake.len(), 3);
        assert_eq!(snake.head(), Position::new(5, 5));
        assert_eq!(snake.body[1], Position::new(4, 5));
        assert_eq!(snake.body[2], Position::new(3, 5));
    }

    #[test]
    fn test_snake_movement() {
        let mut snake = Snake::new(Position::new(5, 5), Direction::Right, 3);

        // Move without growing
        snake.move_snake(false);
        assert_eq!(snake.len(), 3);
        assert_eq!(snake.head(), Position::new(6, 5));

        // Move with growing
        snake.move_snake(true);
        assert_eq!(snake.len(), 4);
        assert_eq!(snake.head(), Position::new(7, 5));
    }

    #[test]
    fn test_collision_detection() {
        let snake = Snake::new(Position::new(5, 5), Direction::Right, 3);
        assert!(!snake.collides_with_body(Position::new(5, 5))); // head
        assert!(snake.collides_with_body(Position::new(4, 5))); // body
        assert!(!snake.collides_with_body(Position::new(10, 10))); // empty
    }

    #[test]
    fn test_bounds_checking() {
        let state = GameState::new(
            Snake::new(Position::new(5, 5), Direction::Right, 3),
            Position::new(10, 10),
            20,
            20,
        );

        assert!(state.is_in_bounds(Position::new(0, 0)));
        assert!(state.is_in_bounds(Position::new(19, 19)));
        assert!(!state.is_in_bounds(Position::new(-1, 0)));
        assert!(!state.is_in_bounds(Position::new(20, 0)));
        assert!(!state.is_in_bounds(Position::new(0, 20)));
    }
}
