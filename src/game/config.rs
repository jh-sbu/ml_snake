use serde::{Deserialize, Serialize};

/// Configuration for the game
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameConfig {
    /// Width of the game grid
    pub grid_width: usize,
    /// Height of the game grid
    pub grid_height: usize,
    /// Initial length of the snake
    pub initial_snake_length: usize,

    // Rewards (for RL)
    /// Reward for eating food
    pub food_reward: f32,
    /// Penalty for each step (encourages efficiency)
    pub step_penalty: f32,
    /// Penalty for dying
    pub death_penalty: f32,
}

impl Default for GameConfig {
    fn default() -> Self {
        Self {
            grid_width: 20,
            grid_height: 20,
            initial_snake_length: 3,
            food_reward: 10.0,
            step_penalty: -0.01,
            death_penalty: -10.0,
        }
    }
}

impl GameConfig {
    /// Create a new configuration with custom grid size
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            grid_width: width,
            grid_height: height,
            ..Default::default()
        }
    }

    /// Create a small grid for testing
    pub fn small() -> Self {
        Self::new(10, 10)
    }

    /// Create a large grid
    pub fn large() -> Self {
        Self::new(30, 30)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = GameConfig::default();
        assert_eq!(config.grid_width, 20);
        assert_eq!(config.grid_height, 20);
        assert_eq!(config.initial_snake_length, 3);
    }

    #[test]
    fn test_custom_config() {
        let config = GameConfig::new(15, 15);
        assert_eq!(config.grid_width, 15);
        assert_eq!(config.grid_height, 15);
    }
}
