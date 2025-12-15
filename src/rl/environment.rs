use super::observation::create_observation;
use crate::game::{Action, Direction, GameConfig, GameEngine, GameState};
use burn::tensor::{Tensor, backend::Backend};

/// Snake environment for reinforcement learning
///
/// Wraps the game engine and provides a Burn-compatible RL interface with:
/// - Tensor observations (4-channel grid)
/// - Discrete action space (5 actions: Up, Down, Left, Right, Continue)
/// - Standard RL interface (reset, step)
pub struct SnakeEnvironment<B: Backend> {
    engine: GameEngine,
    state: GameState,
    device: B::Device,
}

impl<B: Backend> SnakeEnvironment<B> {
    /// Create a new Snake environment
    pub fn new(config: GameConfig, device: B::Device) -> Self {
        let mut engine = GameEngine::new(config);
        let state = engine.reset();
        Self {
            engine,
            state,
            device,
        }
    }

    /// Reset the environment and return initial observation
    ///
    /// Returns: Tensor<B, 3> with shape [4, height, width]
    pub fn reset(&mut self) -> Tensor<B, 3> {
        self.state = self.engine.reset();
        create_observation(&self.state, &self.device)
    }

    /// Step the environment with a discrete action
    ///
    /// Actions:
    /// - 0: Move Up
    /// - 1: Move Down
    /// - 2: Move Left
    /// - 3: Move Right
    /// - 4: Continue (keep current direction)
    ///
    /// Returns: (observation, reward, done)
    /// - observation: Tensor<B, 3> with shape [4, height, width]
    /// - reward: f32 (food_reward, step_penalty, or death_penalty)
    /// - done: bool (true if game terminated)
    pub fn step(&mut self, action_idx: usize) -> (Tensor<B, 3>, f32, bool) {
        let action = action_from_index(action_idx);
        let step_result = self.engine.step(&mut self.state, action);

        let observation = create_observation(&self.state, &self.device);
        let reward = step_result.reward;
        let done = step_result.terminated;

        (observation, reward, done)
    }

    /// Get current observation without stepping
    ///
    /// Returns: Tensor<B, 3> with shape [4, height, width]
    pub fn get_observation(&self) -> Tensor<B, 3> {
        create_observation(&self.state, &self.device)
    }

    /// Get the device used by this environment
    pub fn device(&self) -> &B::Device {
        &self.device
    }

    /// Get reference to current game state (for testing/debugging)
    pub fn state(&self) -> &GameState {
        &self.state
    }
}

/// Convert discrete action index to game Action
///
/// - 0 → Move Up
/// - 1 → Move Down
/// - 2 → Move Left
/// - 3 → Move Right
/// - 4 → Continue
/// - other → Continue (default)
fn action_from_index(idx: usize) -> Action {
    match idx {
        0 => Action::Move(Direction::Up),
        1 => Action::Move(Direction::Down),
        2 => Action::Move(Direction::Left),
        3 => Action::Move(Direction::Right),
        4 => Action::Continue,
        _ => Action::Continue, // Default to Continue for invalid indices
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::state::Position;
    use burn::backend::NdArray;
    use burn::backend::ndarray::NdArrayDevice;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_environment_creation() {
        let device = NdArrayDevice::default();
        let config = GameConfig::default();
        let env = SnakeEnvironment::<TestBackend>::new(config, device);

        assert!(env.state().is_alive);
        assert_eq!(env.state().score, 0);
        assert_eq!(env.state().steps, 0);
    }

    #[test]
    fn test_reset_returns_valid_observation() {
        let device = NdArrayDevice::default();
        let config = GameConfig::default();
        let mut env = SnakeEnvironment::<TestBackend>::new(config, device);

        let obs = env.reset();
        let shape = obs.shape().dims;

        assert_eq!(shape, [4, 20, 20]);
    }

    #[test]
    fn test_step_with_continue_action() {
        let device = NdArrayDevice::default();
        let config = GameConfig::small();
        let mut env = SnakeEnvironment::<TestBackend>::new(config, device);

        let initial_steps = env.state().steps;
        let (obs, reward, done) = env.step(4); // Continue

        assert_eq!(obs.shape().dims, [4, 10, 10]);
        assert!(reward <= 0.0); // step penalty
        assert!(!done); // shouldn't terminate on first step
        assert_eq!(env.state().steps, initial_steps + 1);
    }

    #[test]
    fn test_step_with_directional_actions() {
        let device = NdArrayDevice::default();
        let config = GameConfig::small();
        let mut env = SnakeEnvironment::<TestBackend>::new(config, device);

        // Test all directional actions
        for action_idx in 0..4 {
            env.reset();
            let (obs, _reward, _done) = env.step(action_idx);
            assert_eq!(obs.shape().dims, [4, 10, 10]);
        }
    }

    #[test]
    fn test_step_returns_correct_shapes() {
        let device = NdArrayDevice::default();
        let config = GameConfig::default();
        let mut env = SnakeEnvironment::<TestBackend>::new(config, device);

        let (obs, reward, done) = env.step(4);

        // Check types
        assert_eq!(obs.shape().dims, [4, 20, 20]);
        assert!(reward.is_finite());
        assert!(!done || done); // Just checking it's a bool
    }

    #[test]
    fn test_terminal_state_handling() {
        let device = NdArrayDevice::default();
        let config = GameConfig::small();
        let mut env = SnakeEnvironment::<TestBackend>::new(config, device);

        // Force terminal state by setting snake near wall and moving into it
        env.state.snake.direction = Direction::Left;
        env.state.snake.body[0] = Position::new(0, 5);

        let (_obs, _reward, done) = env.step(4); // Continue left into wall

        assert!(done);
        assert!(!env.state().is_alive);
    }

    #[test]
    fn test_food_reward() {
        let device = NdArrayDevice::default();
        let config = GameConfig::small();
        let mut env = SnakeEnvironment::<TestBackend>::new(config, device);

        // Place food directly in front of snake
        let head = env.state().snake.head();
        let direction = env.state().snake.direction;
        env.state.food = head.moved_in_direction(direction);

        let initial_score = env.state().score;
        let (_, reward, _) = env.step(4); // Continue to eat food

        assert!(reward > 0.0); // Should get food reward
        assert_eq!(env.state().score, initial_score + 1);
    }

    #[test]
    fn test_observation_changes_after_step() {
        let device = NdArrayDevice::default();
        let config = GameConfig::small();
        let mut env = SnakeEnvironment::<TestBackend>::new(config, device);

        let obs1 = env.get_observation();
        env.step(4); // Take a step
        let obs2 = env.get_observation();

        // Observations should be different (snake moved)
        let data1 = obs1.to_data();
        let data2 = obs2.to_data();

        assert_ne!(
            data1.as_slice::<f32>().unwrap(),
            data2.as_slice::<f32>().unwrap()
        );
    }

    #[test]
    fn test_action_mapping() {
        // Test action_from_index function
        assert_eq!(action_from_index(0), Action::Move(Direction::Up));
        assert_eq!(action_from_index(1), Action::Move(Direction::Down));
        assert_eq!(action_from_index(2), Action::Move(Direction::Left));
        assert_eq!(action_from_index(3), Action::Move(Direction::Right));
        assert_eq!(action_from_index(4), Action::Continue);
        assert_eq!(action_from_index(999), Action::Continue); // Invalid → Continue
    }

    #[test]
    fn test_multiple_episodes() {
        let device = NdArrayDevice::default();
        let config = GameConfig::small();
        let mut env = SnakeEnvironment::<TestBackend>::new(config, device);

        // Run two episodes
        for _ in 0..2 {
            env.reset();
            let mut steps = 0;
            let mut done = false;

            // Run until termination or 100 steps
            while !done && steps < 100 {
                let (_obs, _reward, terminated) = env.step(4);
                done = terminated;
                steps += 1;
            }

            // Should either terminate or reach step limit
            assert!(done || steps == 100);
        }
    }

    #[test]
    fn test_device_access() {
        let device = NdArrayDevice::default();
        let config = GameConfig::default();
        let env = SnakeEnvironment::<TestBackend>::new(config, device.clone());

        // Device should be accessible
        let _env_device = env.device();
    }
}
