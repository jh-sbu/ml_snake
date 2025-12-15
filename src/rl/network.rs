//! Actor-Critic neural network for Snake RL agent
//!
//! This module implements a convolutional neural network with two heads:
//! - **Actor head**: Outputs action logits for the policy (4 directional actions)
//! - **Critic head**: Outputs value estimate for state evaluation
//!
//! # Architecture
//!
//! ```text
//! Input: [batch, 4, H, W]
//!   ↓ Conv2d(4→32, k=3, p=1) + ReLU
//!   ↓ Conv2d(32→64, k=3, p=1) + ReLU
//!   ↓ Conv2d(64→64, k=3, p=1) + ReLU
//!   ↓ Flatten: [batch, 64*H*W]
//!   ↓ Linear(64*H*W → 512) + ReLU
//!   ↓ Split
//!   ├─→ Actor: Linear(512 → 4) → Action logits
//!   └─→ Critic: Linear(512 → 1) → Value estimate
//! ```
//!
//! The network processes 4-channel grid observations where each channel represents:
//! - Channel 0: Snake head position
//! - Channel 1: Snake body positions
//! - Channel 2: Food location
//! - Channel 3: Wall boundaries
//!
//! # Example
//!
//! ```rust
//! use ml_snake::rl::{ActorCriticConfig, ActorCriticNetwork};
//! use burn::backend::ndarray::NdArrayDevice;
//! use burn::backend::NdArray;
//! use burn::tensor::Tensor;
//!
//! type Backend = NdArray<f32>;
//!
//! // Create network for 20x20 grid
//! let device = NdArrayDevice::default();
//! let config = ActorCriticConfig::new(20, 20);
//! let network = config.init::<Backend>(&device);
//!
//! // Forward pass with batch of observations
//! let observation = Tensor::zeros([4, 4, 20, 20], &device);
//! let (action_logits, value) = network.forward(observation);
//!
//! assert_eq!(action_logits.dims(), [4, 4]); // [batch, num_actions]
//! assert_eq!(value.dims(), [4, 1]);         // [batch, 1]
//! ```

use burn::{
    module::Module,
    nn::{
        Linear, LinearConfig, PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig},
    },
    tensor::{Tensor, activation::relu, backend::Backend},
};

/// Configuration for the Actor-Critic network
///
/// This configuration struct defines the hyperparameters for the network architecture.
/// Use the `new()` constructor for typical use cases with default hyperparameters.
#[derive(Debug, Clone)]
pub struct ActorCriticConfig {
    /// Number of input channels (default: 4 for snake game)
    pub input_channels: usize,

    /// Number of actions the policy can output (default: 4 for Up/Down/Left/Right)
    pub num_actions: usize,

    /// Grid height in cells
    pub grid_height: usize,

    /// Grid width in cells
    pub grid_width: usize,

    /// Number of channels for each convolutional layer (default: [32, 64, 64])
    pub conv_channels: [usize; 3],

    /// Hidden dimension for the shared fully connected layer (default: 512)
    pub hidden_dim: usize,
}

impl ActorCriticConfig {
    /// Create a new configuration with default hyperparameters
    ///
    /// # Arguments
    ///
    /// * `grid_height` - Height of the game grid in cells
    /// * `grid_width` - Width of the game grid in cells
    ///
    /// # Example
    ///
    /// ```rust
    /// use ml_snake::rl::ActorCriticConfig;
    ///
    /// let config = ActorCriticConfig::new(20, 20);
    /// ```
    pub fn new(grid_height: usize, grid_width: usize) -> Self {
        Self {
            input_channels: 4,
            num_actions: 4,
            grid_height,
            grid_width,
            conv_channels: [32, 64, 64],
            hidden_dim: 512,
        }
    }

    /// Initialize the Actor-Critic network from this configuration
    ///
    /// # Arguments
    ///
    /// * `device` - The device to place the network on (CPU or GPU)
    ///
    /// # Returns
    ///
    /// A new `ActorCriticNetwork` instance initialized with the configuration parameters
    ///
    /// # Example
    ///
    /// ```rust
    /// use ml_snake::rl::ActorCriticConfig;
    /// use burn::backend::ndarray::NdArrayDevice;
    /// use burn::backend::NdArray;
    ///
    /// type Backend = NdArray<f32>;
    ///
    /// let device = NdArrayDevice::default();
    /// let config = ActorCriticConfig::new(20, 20);
    /// let network = config.init::<Backend>(&device);
    /// ```
    pub fn init<B: Backend>(&self, device: &B::Device) -> ActorCriticNetwork<B> {
        // Calculate flattened dimension after conv layers
        // Conv layers preserve spatial dimensions (padding=1, stride=1, kernel=3)
        let flattened_dim = self.conv_channels[2] * self.grid_height * self.grid_width;

        ActorCriticNetwork {
            conv1: Conv2dConfig::new([self.input_channels, self.conv_channels[0]], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
            conv2: Conv2dConfig::new([self.conv_channels[0], self.conv_channels[1]], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
            conv3: Conv2dConfig::new([self.conv_channels[1], self.conv_channels[2]], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
            fc_shared: LinearConfig::new(flattened_dim, self.hidden_dim).init(device),
            actor_head: LinearConfig::new(self.hidden_dim, self.num_actions).init(device),
            critic_head: LinearConfig::new(self.hidden_dim, 1).init(device),
        }
    }
}

impl Default for ActorCriticConfig {
    fn default() -> Self {
        Self::new(20, 20)
    }
}

/// Actor-Critic Convolutional Neural Network
///
/// This network processes grid-based observations through a shared convolutional
/// trunk and outputs both action logits (policy) and value estimates (critic).
///
/// The network is generic over the Backend, allowing it to run on different
/// hardware (CPU with NdArray, GPU with Wgpu) and support automatic differentiation
/// for training (Autodiff wrapper).
///
/// # Type Parameters
///
/// * `B` - The Burn backend to use (e.g., `NdArray<f32>`, `Wgpu`, `Autodiff<NdArray<f32>>`)
#[derive(Module, Debug)]
pub struct ActorCriticNetwork<B: Backend> {
    /// First convolutional layer: 4 → 32 channels
    conv1: Conv2d<B>,
    /// Second convolutional layer: 32 → 64 channels
    conv2: Conv2d<B>,
    /// Third convolutional layer: 64 → 64 channels
    conv3: Conv2d<B>,
    /// Shared fully connected layer after flattening
    fc_shared: Linear<B>,
    /// Actor head: outputs action logits
    actor_head: Linear<B>,
    /// Critic head: outputs value estimate
    critic_head: Linear<B>,
}

impl<B: Backend> ActorCriticNetwork<B> {
    /// Forward pass through the network
    ///
    /// Processes a batch of grid observations through the convolutional trunk,
    /// then splits into actor and critic heads to produce action logits and
    /// value estimates.
    ///
    /// # Arguments
    ///
    /// * `observation` - Tensor with shape `[batch, 4, height, width]` where:
    ///   - `batch` is the batch size
    ///   - `4` is the number of input channels (head, body, food, walls)
    ///   - `height` and `width` are the grid dimensions
    ///
    /// # Returns
    ///
    /// A tuple of:
    /// - `action_logits`: Tensor with shape `[batch, num_actions]` - unnormalized action probabilities
    /// - `value`: Tensor with shape `[batch, 1]` - value estimate for each state
    ///
    /// # Example
    ///
    /// ```rust
    /// use ml_snake::rl::{ActorCriticConfig, ActorCriticNetwork};
    /// use burn::backend::ndarray::NdArrayDevice;
    /// use burn::backend::NdArray;
    /// use burn::tensor::Tensor;
    ///
    /// type Backend = NdArray<f32>;
    ///
    /// let device = NdArrayDevice::default();
    /// let config = ActorCriticConfig::new(20, 20);
    /// let network = config.init::<Backend>(&device);
    ///
    /// // Create a batch of 8 observations
    /// let observation = Tensor::zeros([8, 4, 20, 20], &device);
    /// let (action_logits, value) = network.forward(observation);
    ///
    /// assert_eq!(action_logits.dims(), [8, 4]);
    /// assert_eq!(value.dims(), [8, 1]);
    /// ```
    pub fn forward(&self, observation: Tensor<B, 4>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        // Convolutional trunk with ReLU activations
        // Input: [batch, 4, H, W]
        let x = self.conv1.forward(observation);
        let x = relu(x);
        // After conv1: [batch, 32, H, W]

        let x = self.conv2.forward(x);
        let x = relu(x);
        // After conv2: [batch, 64, H, W]

        let x = self.conv3.forward(x);
        let x = relu(x);
        // After conv3: [batch, 64, H, W]

        // Flatten: [batch, 64, H, W] → [batch, 64*H*W]
        let [batch_size, channels, height, width] = x.dims();
        let flattened_dim = channels * height * width;
        let x = x.reshape([batch_size, flattened_dim]);

        // Shared fully connected layer with ReLU
        let x = self.fc_shared.forward(x);
        let x = relu(x);
        // After fc_shared: [batch, 512]

        // Split into actor and critic heads
        let action_logits = self.actor_head.forward(x.clone());
        // Actor output: [batch, 4]

        let value = self.critic_head.forward(x);
        // Critic output: [batch, 1]

        (action_logits, value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Autodiff;
    use burn::backend::ndarray::{NdArray, NdArrayDevice};
    use burn::tensor::{Distribution, TensorData};

    type TestBackend = NdArray<f32>;
    type TestAutodiffBackend = Autodiff<NdArray<f32>>;

    #[test]
    fn test_forward_pass_shapes() {
        let device = NdArrayDevice::default();
        let config = ActorCriticConfig::new(20, 20);
        let network = config.init::<TestBackend>(&device);

        // Create dummy observation [batch=2, channels=4, H=20, W=20]
        let observation = Tensor::zeros([2, 4, 20, 20], &device);

        let (action_logits, value) = network.forward(observation);

        // Verify shapes
        assert_eq!(action_logits.dims(), [2, 4]); // [batch, num_actions]
        assert_eq!(value.dims(), [2, 1]); // [batch, 1]
    }

    #[test]
    fn test_different_batch_sizes() {
        let device = NdArrayDevice::default();
        let config = ActorCriticConfig::new(20, 20);
        let network = config.init::<TestBackend>(&device);

        for batch_size in [1, 4, 16, 32] {
            let observation = Tensor::zeros([batch_size, 4, 20, 20], &device);

            let (action_logits, value) = network.forward(observation);

            assert_eq!(action_logits.dims(), [batch_size, 4]);
            assert_eq!(value.dims(), [batch_size, 1]);
        }
    }

    #[test]
    fn test_different_grid_sizes() {
        let device = NdArrayDevice::default();

        // Test 10x10 grid
        let config = ActorCriticConfig::new(10, 10);
        let network = config.init::<TestBackend>(&device);
        let obs = Tensor::zeros([1, 4, 10, 10], &device);
        let (logits, value) = network.forward(obs);
        assert_eq!(logits.dims(), [1, 4]);
        assert_eq!(value.dims(), [1, 1]);

        // Test 30x30 grid
        let config = ActorCriticConfig::new(30, 30);
        let network = config.init::<TestBackend>(&device);
        let obs = Tensor::zeros([1, 4, 30, 30], &device);
        let (logits, value) = network.forward(obs);
        assert_eq!(logits.dims(), [1, 4]);
        assert_eq!(value.dims(), [1, 1]);
    }

    #[test]
    fn test_gradient_flow() {
        let device = NdArrayDevice::default();
        let config = ActorCriticConfig::new(20, 20);
        let network = config.init::<TestAutodiffBackend>(&device);

        // Create observation with gradient tracking
        let observation = Tensor::ones([1, 4, 20, 20], &device).require_grad();

        // Forward pass
        let (action_logits, value) = network.forward(observation.clone());

        // Create dummy loss (sum of outputs)
        let loss = action_logits.sum() + value.sum();

        // Backward pass
        let gradients = loss.backward();

        // Check that input has gradients
        let obs_grad = observation.grad(&gradients);
        assert!(
            obs_grad.is_some(),
            "Gradients should flow back to input observation"
        );

        // Verify gradients are non-zero
        let grad_tensor = obs_grad.unwrap();
        let grad_data: TensorData = grad_tensor.into_data();
        let grad_slice = grad_data.as_slice::<f32>().unwrap();
        let grad_sum: f32 = grad_slice.iter().sum();
        assert!(
            grad_sum.abs() > 1e-6,
            "Gradients should be non-zero, got sum: {}",
            grad_sum
        );
    }

    #[test]
    fn test_separate_head_gradients() {
        let device = NdArrayDevice::default();
        let config = ActorCriticConfig::new(20, 20);
        let network = config.init::<TestAutodiffBackend>(&device);

        let observation = Tensor::ones([2, 4, 20, 20], &device).require_grad();

        // Test actor head gradient flow
        let (action_logits, _) = network.forward(observation.clone());
        let actor_loss = action_logits.sum();
        let actor_grads = actor_loss.backward();

        // Check that gradients exist for the input
        let obs_grad = observation.grad(&actor_grads);
        assert!(obs_grad.is_some(), "Actor head should produce gradients");

        // Test critic head gradient flow
        let observation2 = Tensor::ones([2, 4, 20, 20], &device).require_grad();
        let (_, value) = network.forward(observation2.clone());
        let critic_loss = value.sum();
        let critic_grads = critic_loss.backward();

        // Check that gradients exist for the input
        let obs_grad2 = observation2.grad(&critic_grads);
        assert!(obs_grad2.is_some(), "Critic head should produce gradients");
    }

    #[test]
    fn test_batch_consistency() {
        let device = NdArrayDevice::default();
        let config = ActorCriticConfig::new(20, 20);
        let network = config.init::<TestBackend>(&device);

        // Create same observation
        let single_obs = Tensor::ones([1, 4, 20, 20], &device);

        // Process as single
        let (logits_single, value_single) = network.forward(single_obs.clone());

        // Process as batch of 4 (same observation repeated)
        let obs_batch = Tensor::cat(
            vec![
                single_obs.clone(),
                single_obs.clone(),
                single_obs.clone(),
                single_obs,
            ],
            0,
        );
        let (logits_batch, value_batch) = network.forward(obs_batch);

        // Extract first element of batch and compare with single
        let logits_single_data: TensorData = logits_single.into_data();
        let logits_batch_data: TensorData = logits_batch.into_data();

        let single_vals = logits_single_data.as_slice::<f32>().unwrap();
        let batch_vals = logits_batch_data.as_slice::<f32>().unwrap();

        // Compare first batch element with single result
        for j in 0..4 {
            let diff = (single_vals[j] - batch_vals[j]).abs();
            assert!(
                diff < 1e-5,
                "Batch element 0 should match single at position {}, diff: {}",
                j,
                diff
            );
        }

        // Similarly for values
        let value_single_data: TensorData = value_single.into_data();
        let value_batch_data: TensorData = value_batch.into_data();

        let single_val = value_single_data.as_slice::<f32>().unwrap()[0];
        let batch_val = value_batch_data.as_slice::<f32>().unwrap()[0];
        let diff = (single_val - batch_val).abs();
        assert!(
            diff < 1e-5,
            "Value for batch element 0 should match single, diff: {}",
            diff
        );
    }

    #[test]
    fn test_output_finite() {
        let device = NdArrayDevice::default();
        let config = ActorCriticConfig::new(20, 20);
        let network = config.init::<TestBackend>(&device);

        // Random observation
        let observation = Tensor::random([8, 4, 20, 20], Distribution::Uniform(0.0, 1.0), &device);

        let (action_logits, value) = network.forward(observation);

        // Check all outputs are finite
        let logits_data: TensorData = action_logits.into_data();
        for &val in logits_data.as_slice::<f32>().unwrap() {
            assert!(val.is_finite(), "Logits should be finite, got: {}", val);
        }

        let value_data: TensorData = value.into_data();
        for &val in value_data.as_slice::<f32>().unwrap() {
            assert!(val.is_finite(), "Values should be finite, got: {}", val);
        }
    }

    #[test]
    fn test_with_real_observations() {
        use crate::game::GameConfig;
        use crate::rl::SnakeEnvironment;

        let device = NdArrayDevice::default();

        // Create environment and get real observations
        let game_config = GameConfig::new(20, 20);
        let mut env = SnakeEnvironment::<TestBackend>::new(game_config, device.clone());

        let obs = env.reset();

        // Create network
        let network_config = ActorCriticConfig::new(20, 20);
        let network = network_config.init::<TestBackend>(&device);

        // Process observation (add batch dimension)
        let obs_batch = obs.unsqueeze_dim(0); // [1, 4, 20, 20]
        let (action_logits, value) = network.forward(obs_batch);

        // Verify shapes
        assert_eq!(action_logits.dims(), [1, 4]);
        assert_eq!(value.dims(), [1, 1]);

        // Verify outputs are finite
        let logits_data: TensorData = action_logits.into_data();
        let value_data: TensorData = value.into_data();

        for &val in logits_data.as_slice::<f32>().unwrap() {
            assert!(val.is_finite());
        }
        for &val in value_data.as_slice::<f32>().unwrap() {
            assert!(val.is_finite());
        }
    }
}
