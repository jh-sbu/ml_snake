use super::{
    action::{Action, Direction},
    config::GameConfig,
    state::{CollisionType, GameState, Position, Snake},
};
use rand::Rng;

/// Information about a step
#[derive(Debug, Clone, PartialEq)]
pub struct StepInfo {
    /// Whether the snake ate food this step
    pub ate_food: bool,
    /// Type of collision if one occurred
    pub collision_type: Option<CollisionType>,
}

/// Result of a game step
#[derive(Debug, Clone, PartialEq)]
pub struct StepResult {
    /// Reward for this step (for RL training)
    pub reward: f32,
    /// Whether the game has terminated
    pub terminated: bool,
    /// Additional information about the step
    pub info: StepInfo,
}

/// The game engine that handles all game logic
pub struct GameEngine {
    config: GameConfig,
    rng: rand::rngs::ThreadRng,
}

impl GameEngine {
    /// Create a new game engine with the given configuration
    pub fn new(config: GameConfig) -> Self {
        Self {
            config,
            rng: rand::thread_rng(),
        }
    }

    /// Reset the game to initial state
    pub fn reset(&mut self) -> GameState {
        let center_x = (self.config.grid_width / 2) as i32;
        let center_y = (self.config.grid_height / 2) as i32;

        let snake = Snake::new(
            Position::new(center_x, center_y),
            Direction::Right,
            self.config.initial_snake_length,
        );

        let food = self.spawn_food_avoid_snake(&snake);

        GameState::new(snake, food, self.config.grid_width, self.config.grid_height)
    }

    /// Execute one step of the game
    pub fn step(&mut self, state: &mut GameState, action: Action) -> StepResult {
        if !state.is_alive {
            return StepResult {
                reward: 0.0,
                terminated: true,
                info: StepInfo {
                    ate_food: false,
                    collision_type: None,
                },
            };
        }

        // Update direction based on action (prevent 180° turns)
        match action {
            Action::Move(new_direction) => {
                if !state.snake.direction.is_opposite(new_direction) {
                    state.snake.direction = new_direction;
                }
            }
            Action::Continue => {
                // Keep current direction
            }
        }

        // Calculate new head position
        let new_head = state.snake.head().moved_in_direction(state.snake.direction);

        // Check for collisions
        if let Some(collision_type) = self.check_collision(state, new_head) {
            state.is_alive = false;
            state.steps += 1;

            return StepResult {
                reward: self.config.death_penalty,
                terminated: true,
                info: StepInfo {
                    ate_food: false,
                    collision_type: Some(collision_type),
                },
            };
        }

        // Check if snake ate food
        let ate_food = new_head == state.food;

        // Move snake (grow if ate food)
        state.snake.move_snake(ate_food);

        // Update score and spawn new food if needed
        let mut reward = self.config.step_penalty;

        if ate_food {
            state.score += 1;
            state.food = self.spawn_food_avoid_snake(&state.snake);
            reward += self.config.food_reward;
        }

        state.steps += 1;

        StepResult {
            reward,
            terminated: false,
            info: StepInfo {
                ate_food,
                collision_type: None,
            },
        }
    }

    /// Check if the new head position causes a collision
    fn check_collision(&self, state: &GameState, pos: Position) -> Option<CollisionType> {
        // Check wall collision
        if !state.is_in_bounds(pos) {
            return Some(CollisionType::Wall);
        }

        // Check self-collision
        if state.snake.collides_with_body(pos) {
            return Some(CollisionType::SelfCollision);
        }

        None
    }

    /// Spawn food at a random empty position
    fn spawn_food_avoid_snake(&mut self, snake: &Snake) -> Position {
        loop {
            let x = self.rng.gen_range(0..self.config.grid_width) as i32;
            let y = self.rng.gen_range(0..self.config.grid_height) as i32;
            let pos = Position::new(x, y);

            if !snake.body.contains(&pos) {
                return pos;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reset() {
        let mut engine = GameEngine::new(GameConfig::default());
        let state = engine.reset();

        assert!(state.is_alive);
        assert_eq!(state.score, 0);
        assert_eq!(state.steps, 0);
        assert_eq!(state.snake.len(), 3);
    }

    #[test]
    fn test_basic_movement() {
        let mut engine = GameEngine::new(GameConfig::small());
        let mut state = engine.reset();
        let initial_head = state.snake.head();

        let result = engine.step(&mut state, Action::Continue);

        assert!(!result.terminated);
        assert!(!result.info.ate_food);
        assert_eq!(state.steps, 1);
        assert_ne!(state.snake.head(), initial_head);
    }

    #[test]
    fn test_food_consumption() {
        let mut engine = GameEngine::new(GameConfig::small());
        let mut state = engine.reset();

        // Place food directly in front of snake
        let head = state.snake.head();
        state.food = head.moved_in_direction(state.snake.direction);
        let initial_length = state.snake.len();

        let result = engine.step(&mut state, Action::Continue);

        assert!(result.info.ate_food);
        assert_eq!(state.score, 1);
        assert_eq!(state.snake.len(), initial_length + 1);
        assert!(result.reward > 0.0); // Should get food reward
    }

    #[test]
    fn test_wall_collision() {
        let mut engine = GameEngine::new(GameConfig::small());
        let mut state = GameState::new(
            Snake::new(Position::new(0, 5), Direction::Left, 3),
            Position::new(5, 5),
            10,
            10,
        );

        let result = engine.step(&mut state, Action::Continue);

        assert!(result.terminated);
        assert!(!state.is_alive);
        assert_eq!(result.info.collision_type, Some(CollisionType::Wall));
    }

    #[test]
    fn test_self_collision() {
        let mut engine = GameEngine::new(GameConfig::small());

        // Create a longer snake that will collide with itself
        // Snake at (5, 5) going Right with length 4
        // Body: (5,5), (4,5), (3,5), (2,5)
        let snake = Snake::new(Position::new(5, 5), Direction::Right, 4);
        let mut state = GameState::new(snake, Position::new(8, 8), 10, 10);

        // Move in a pattern that will cause self-collision:
        // Right: (6,5), (5,5), (4,5), (3,5)
        engine.step(&mut state, Action::Continue);
        // Down: (6,6), (6,5), (5,5), (4,5)
        engine.step(&mut state, Action::Move(Direction::Down));
        // Left: (5,6), (6,6), (6,5), (5,5)
        engine.step(&mut state, Action::Move(Direction::Left));
        // Up: (5,5) - this should collide with body at (5,5)!
        let result = engine.step(&mut state, Action::Move(Direction::Up));

        assert!(result.terminated);
        assert_eq!(
            result.info.collision_type,
            Some(CollisionType::SelfCollision)
        );
    }

    #[test]
    fn test_prevent_180_degree_turn() {
        let mut engine = GameEngine::new(GameConfig::small());
        let mut state = engine.reset();
        state.snake.direction = Direction::Right;

        // Try to turn 180 degrees (should be ignored)
        engine.step(&mut state, Action::Move(Direction::Left));

        assert_eq!(state.snake.direction, Direction::Right);
    }

    #[test]
    fn test_terminated_game_no_update() {
        let mut engine = GameEngine::new(GameConfig::small());
        let mut state = engine.reset();
        state.is_alive = false;
        let steps_before = state.steps;

        let result = engine.step(&mut state, Action::Continue);

        assert!(result.terminated);
        assert_eq!(state.steps, steps_before); // Should not increment
    }
}
