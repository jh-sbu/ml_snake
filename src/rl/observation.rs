use burn::tensor::{Tensor, TensorData, backend::Backend};

use crate::game::GameState;

/// Create a 4-channel observation tensor from game state
///
/// Channels:
/// - 0: Snake head (1.0 at head position)
/// - 1: Snake body (1.0 at body positions, excluding head)
/// - 2: Food location (1.0 at food position)
/// - 3: Walls/boundaries (1.0 at grid edges)
///
/// Returns: Tensor<B, 3> with shape [4, height, width]
pub fn create_observation<B: Backend>(state: &GameState, device: &B::Device) -> Tensor<B, 3> {
    let head_channel = create_head_channel(state, device);
    let body_channel = create_body_channel(state, device);
    let food_channel = create_food_channel(state, device);
    let walls_channel = create_walls_channel(state, device);

    // Stack channels along dimension 0: [4, height, width]
    // Each channel is [height, width], stacking creates [4, height, width]
    Tensor::stack(
        vec![head_channel, body_channel, food_channel, walls_channel],
        0,
    )
}

/// Create channel with snake head position (1.0 at head, 0.0 elsewhere)
fn create_head_channel<B: Backend>(state: &GameState, device: &B::Device) -> Tensor<B, 2> {
    let mut data = vec![0.0; state.grid_height * state.grid_width];

    let head = state.snake.head();
    let idx = (head.y as usize) * state.grid_width + (head.x as usize);
    data[idx] = 1.0;

    let tensor_data = TensorData::new(data, [state.grid_height, state.grid_width]);

    Tensor::<B, 2>::from_data(tensor_data, device)
}

/// Create channel with snake body positions (1.0 at body, 0.0 elsewhere)
/// Excludes the head position
fn create_body_channel<B: Backend>(state: &GameState, device: &B::Device) -> Tensor<B, 2> {
    let mut data = vec![0.0; state.grid_height * state.grid_width];

    // Get body segments (excluding head)
    for &pos in state.snake.body_segments() {
        let idx = (pos.y as usize) * state.grid_width + (pos.x as usize);
        data[idx] = 1.0;
    }

    let tensor_data = TensorData::new(data, [state.grid_height, state.grid_width]);

    Tensor::<B, 2>::from_data(tensor_data, device)
}

/// Create channel with food position (1.0 at food, 0.0 elsewhere)
fn create_food_channel<B: Backend>(state: &GameState, device: &B::Device) -> Tensor<B, 2> {
    let mut data = vec![0.0; state.grid_height * state.grid_width];

    let food = state.food;
    let idx = (food.y as usize) * state.grid_width + (food.x as usize);
    data[idx] = 1.0;

    let tensor_data = TensorData::new(data, [state.grid_height, state.grid_width]);

    Tensor::<B, 2>::from_data(tensor_data, device)
}

/// Create channel with walls/boundaries (1.0 at edges, 0.0 in interior)
fn create_walls_channel<B: Backend>(state: &GameState, device: &B::Device) -> Tensor<B, 2> {
    let mut data = vec![0.0; state.grid_height * state.grid_width];

    let width = state.grid_width;
    let height = state.grid_height;

    // Top and bottom edges
    for x in 0..width {
        data[x] = 1.0; // Top edge
        data[(height - 1) * width + x] = 1.0; // Bottom edge
    }

    // Left and right edges
    for y in 0..height {
        data[y * width] = 1.0; // Left edge
        data[y * width + (width - 1)] = 1.0; // Right edge
    }

    let tensor_data = TensorData::new(data, [height, width]);

    Tensor::<B, 2>::from_data(tensor_data, device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::{GameState, Snake, action::Direction, state::Position};
    use burn::backend::NdArray;
    use burn::backend::ndarray::NdArrayDevice;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_observation_shape() {
        let device = NdArrayDevice::default();
        let snake = Snake::new(Position::new(5, 5), Direction::Right, 3);
        let state = GameState::new(snake, Position::new(10, 10), 20, 20);

        let obs = create_observation::<TestBackend>(&state, &device);
        let shape = obs.shape().dims;

        assert_eq!(shape, [4, 20, 20]);
    }

    #[test]
    fn test_head_channel() {
        let device = NdArrayDevice::default();
        let snake = Snake::new(Position::new(5, 5), Direction::Right, 3);
        let state = GameState::new(snake, Position::new(10, 10), 20, 20);

        let head_channel = create_head_channel::<TestBackend>(&state, &device);
        let data = head_channel.to_data();

        // Check head position has 1.0
        let head_idx = 5 * 20 + 5;
        assert_eq!(data.as_slice::<f32>().unwrap()[head_idx], 1.0);

        // Check sum equals 1.0 (only one position)
        let sum: f32 = data.as_slice::<f32>().unwrap().iter().sum();
        assert_eq!(sum, 1.0);
    }

    #[test]
    fn test_body_channel() {
        let device = NdArrayDevice::default();
        let snake = Snake::new(Position::new(5, 5), Direction::Right, 4);
        let state = GameState::new(snake, Position::new(10, 10), 20, 20);

        let body_channel = create_body_channel::<TestBackend>(&state, &device);
        let data = body_channel.to_data();

        // Snake length is 4, so body (excluding head) should be 3 segments
        let sum: f32 = data.as_slice::<f32>().unwrap().iter().sum();
        assert_eq!(sum, 3.0);

        // Head position should be 0.0 in body channel
        let head_idx = 5 * 20 + 5;
        assert_eq!(data.as_slice::<f32>().unwrap()[head_idx], 0.0);
    }

    #[test]
    fn test_food_channel() {
        let device = NdArrayDevice::default();
        let snake = Snake::new(Position::new(5, 5), Direction::Right, 3);
        let state = GameState::new(snake, Position::new(10, 12), 20, 20);

        let food_channel = create_food_channel::<TestBackend>(&state, &device);
        let data = food_channel.to_data();

        // Check food position has 1.0
        let food_idx = 12 * 20 + 10;
        assert_eq!(data.as_slice::<f32>().unwrap()[food_idx], 1.0);

        // Check sum equals 1.0 (only one position)
        let sum: f32 = data.as_slice::<f32>().unwrap().iter().sum();
        assert_eq!(sum, 1.0);
    }

    #[test]
    fn test_walls_channel() {
        let device = NdArrayDevice::default();
        let snake = Snake::new(Position::new(5, 5), Direction::Right, 3);
        let state = GameState::new(snake, Position::new(10, 10), 20, 20);

        let walls_channel = create_walls_channel::<TestBackend>(&state, &device);
        let data = walls_channel.to_data();

        // Check corners are 1.0
        assert_eq!(data.as_slice::<f32>().unwrap()[0], 1.0); // Top-left
        assert_eq!(data.as_slice::<f32>().unwrap()[19], 1.0); // Top-right
        assert_eq!(data.as_slice::<f32>().unwrap()[19 * 20], 1.0); // Bottom-left
        assert_eq!(data.as_slice::<f32>().unwrap()[19 * 20 + 19], 1.0); // Bottom-right

        // Check center is 0.0
        let center_idx = 10 * 20 + 10;
        assert_eq!(data.as_slice::<f32>().unwrap()[center_idx], 0.0);

        // Check sum: 2*width + 2*height - 4 (corners counted twice)
        let sum: f32 = data.as_slice::<f32>().unwrap().iter().sum();
        assert_eq!(sum, (2 * 20 + 2 * 20 - 4) as f32);
    }

    #[test]
    fn test_observation_with_different_grid_sizes() {
        let device = NdArrayDevice::default();

        // Test 10x10 grid
        let snake = Snake::new(Position::new(5, 5), Direction::Right, 3);
        let state = GameState::new(snake, Position::new(7, 7), 10, 10);
        let obs = create_observation::<TestBackend>(&state, &device);
        assert_eq!(obs.shape().dims, [4, 10, 10]);

        // Test 30x15 grid
        let snake = Snake::new(Position::new(10, 7), Direction::Right, 3);
        let state = GameState::new(snake, Position::new(20, 10), 30, 15);
        let obs = create_observation::<TestBackend>(&state, &device);
        assert_eq!(obs.shape().dims, [4, 15, 30]);
    }

    #[test]
    fn test_observation_values_in_range() {
        let device = NdArrayDevice::default();
        let snake = Snake::new(Position::new(5, 5), Direction::Right, 3);
        let state = GameState::new(snake, Position::new(10, 10), 20, 20);

        let obs = create_observation::<TestBackend>(&state, &device);
        let data = obs.to_data();

        // All values should be 0.0 or 1.0
        for &value in data.as_slice::<f32>().unwrap() {
            assert!(value == 0.0 || value == 1.0);
        }
    }
}
