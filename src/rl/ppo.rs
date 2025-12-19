//! PPO (Proximal Policy Optimization) agent implementation
//!
//! This module implements the PPO algorithm for training the Snake RL agent.
//! It includes action selection, loss computation, and parameter updates.

use super::buffer::RolloutBuffer;
use super::config::PPOConfig;
use super::network::ActorCriticNetwork;
use burn::{
    module::AutodiffModule,
    optim::{Adam, AdamConfig, GradientsParams, Optimizer, adaptor::OptimizerAdaptor},
    tensor::{
        ElementConversion, Int, Tensor,
        activation::{log_softmax, softmax},
        backend::AutodiffBackend,
    },
};
use rand::Rng;

/// PPO agent for reinforcement learning
///
/// Combines an actor-critic neural network with the PPO training algorithm.
/// Manages experience collection, advantage estimation, and policy optimization.
///
/// # Type Parameters
///
/// * `B` - Autodiff backend for gradient computation
///
/// # Example
///
/// ```rust,ignore
/// use ml_snake::rl::{PPOAgent, PPOConfig, ActorCriticConfig};
/// use burn::backend::{Autodiff, ndarray::{NdArray, NdArrayDevice}};
///
/// type Backend = Autodiff<NdArray<f32>>;
///
/// let device = NdArrayDevice::default();
/// let network_config = ActorCriticConfig::new(20, 20);
/// let network = network_config.init::<Backend>(&device);
/// let ppo_config = PPOConfig::default();
///
/// let agent = PPOAgent::new(network, ppo_config, 20, 20, device);
/// ```
pub struct PPOAgent<B: AutodiffBackend> {
    /// Actor-Critic neural network
    network: ActorCriticNetwork<B>,

    /// Adam optimizer for network parameters
    optim: OptimizerAdaptor<Adam, ActorCriticNetwork<B>, B>,

    /// PPO hyperparameters
    config: PPOConfig,

    /// Experience buffer for rollout data
    buffer: RolloutBuffer<B::InnerBackend>,

    /// Training step counter
    training_step: usize,

    /// Episode counter
    episodes_trained: usize,

    /// Grid height (for model persistence)
    grid_height: usize,

    /// Grid width (for model persistence)
    grid_width: usize,

    /// Device for tensor operations
    device: B::Device,
}

impl<B: AutodiffBackend> PPOAgent<B> {
    /// Create a new PPO agent
    ///
    /// # Arguments
    ///
    /// * `network` - Actor-critic neural network
    /// * `config` - PPO hyperparameters
    /// * `device` - Device for computation
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use ml_snake::rl::{PPOAgent, PPOConfig, ActorCriticConfig};
    /// use burn::backend::{Autodiff, ndarray::{NdArray, NdArrayDevice}};
    ///
    /// type Backend = Autodiff<NdArray<f32>>;
    ///
    /// let device = NdArrayDevice::default();
    /// let network_config = ActorCriticConfig::new(20, 20);
    /// let network = network_config.init::<Backend>(&device);
    /// let ppo_config = PPOConfig::default();
    ///
    /// let agent = PPOAgent::new(network, ppo_config, 20, 20, device);
    /// ```
    pub fn new(
        network: ActorCriticNetwork<B>,
        config: PPOConfig,
        grid_height: usize,
        grid_width: usize,
        device: B::Device,
    ) -> Self {
        // Validate config
        config.validate().expect("Invalid PPO configuration");

        // Create optimizer
        let optim = AdamConfig::new().init();

        // Create buffer
        let buffer = RolloutBuffer::new(config.update_frequency, device.clone());

        Self {
            network,
            optim,
            config,
            buffer,
            training_step: 0,
            episodes_trained: 0,
            grid_height,
            grid_width,
            device,
        }
    }

    /// Select an action from an observation during rollout
    ///
    /// Samples an action from the policy distribution and returns the action index,
    /// log probability, and value estimate.
    ///
    /// # Arguments
    ///
    /// * `observation` - State observation tensor [4, H, W]
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// - `action` - Discrete action index
    /// - `log_prob` - Log probability of the selected action
    /// - `value` - Value estimate V(s)
    pub fn select_action(&self, observation: Tensor<B::InnerBackend, 3>) -> (usize, f32, f32) {
        let device = observation.device();

        // Add batch dimension
        let obs_batch = observation.unsqueeze_dim(0); // [1, 4, H, W]

        // Forward pass in valid (no-grad) mode
        let network = self.network.clone().valid();
        let (action_logits, value) = network.forward(obs_batch);

        // Sample action from categorical distribution
        let action_probs = softmax(action_logits.clone(), 1); // [1, num_actions]
        let action_idx = sample_categorical(&action_probs);

        // Compute log probability of selected action
        let log_probs = log_softmax(action_logits, 1);
        let action_tensor =
            Tensor::<B::InnerBackend, 1, Int>::from_ints([action_idx as i32], &device);
        let log_prob = log_probs
            .gather(1, action_tensor.unsqueeze_dim(1))
            .squeeze::<1>(1)
            .into_scalar()
            .elem::<f32>();

        // Extract value estimate
        let value_scalar = value.squeeze::<1>(1).into_scalar().elem::<f32>();

        (action_idx, log_prob, value_scalar)
    }

    /// Store a transition in the buffer
    ///
    /// # Arguments
    ///
    /// * `observation` - State observation
    /// * `action` - Action taken
    /// * `log_prob` - Log probability of the action
    /// * `reward` - Reward received
    /// * `value` - Value estimate
    /// * `done` - Whether episode terminated
    pub fn store_transition(
        &mut self,
        observation: Tensor<B::InnerBackend, 3>,
        action: usize,
        log_prob: f32,
        reward: f32,
        value: f32,
        done: bool,
    ) {
        self.buffer
            .push(observation, action, log_prob, reward, value, done);
    }

    /// Check if the buffer is full and ready for update
    ///
    /// # Returns
    ///
    /// `true` if buffer has reached update_frequency capacity
    pub fn should_update(&self) -> bool {
        self.buffer.is_full()
    }

    /// Perform a PPO update
    ///
    /// Computes advantages using GAE, then performs multiple epochs of
    /// minibatch updates using the clipped PPO objective.
    ///
    /// # Arguments
    ///
    /// * `last_value` - Value estimate for the last state (for bootstrapping)
    /// * `last_done` - Whether the last state was terminal
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// - `policy_loss` - Average policy loss
    /// - `value_loss` - Average value loss
    /// - `entropy` - Average policy entropy
    /// - `total_loss` - Average total loss
    pub fn update(&mut self, last_value: f32, last_done: bool) -> (f32, f32, f32, f32) {
        // Compute advantages using GAE
        self.buffer.compute_advantages(
            self.config.gamma,
            self.config.gae_lambda,
            last_value,
            last_done,
        );

        // Accumulate losses
        let mut total_policy_loss = 0.0;
        let mut total_value_loss = 0.0;
        let mut total_entropy = 0.0;
        let mut n_updates = 0;

        // Multiple epochs over the data
        for _epoch in 0..self.config.n_epochs {
            // Sample random batches
            let batch_indices = self.buffer.sample_indices(self.config.batch_size);

            for indices in batch_indices {
                // Get batch data from buffer (TensorData format)
                let (obs_data, actions_data, old_log_probs_data, advantages_data, returns_data) =
                    self.buffer.get_batch(&indices);

                // Construct tensors directly on autodiff backend
                let obs: Tensor<B, 4> = Tensor::from_data(obs_data, &self.device);
                let actions: Tensor<B, 1, Int> = Tensor::from_data(actions_data, &self.device);
                let old_log_probs: Tensor<B, 1> =
                    Tensor::from_data(old_log_probs_data, &self.device);
                let advantages: Tensor<B, 1> = Tensor::from_data(advantages_data, &self.device);
                let returns: Tensor<B, 1> = Tensor::from_data(returns_data, &self.device);

                // Forward pass
                let (action_logits, values) = self.network.forward(obs);

                // Compute losses
                let (policy_loss, entropy) =
                    self.compute_policy_loss(&action_logits, &actions, &old_log_probs, &advantages);

                let value_loss = self.compute_value_loss(&values, &returns);

                // Total loss: L_policy - c_entropy * H + c_value * L_value
                let total_loss = policy_loss.clone() - entropy.clone() * self.config.entropy_coef
                    + value_loss.clone() * self.config.value_coef;

                // Backward pass
                let grads = total_loss.backward();

                // Update network parameters
                let grads = GradientsParams::from_grads(grads, &self.network);
                self.network =
                    self.optim
                        .step(self.config.learning_rate, self.network.clone(), grads);

                // Accumulate losses (convert to scalars)
                total_policy_loss += policy_loss.into_scalar().elem::<f32>();
                total_value_loss += value_loss.into_scalar().elem::<f32>();
                total_entropy += entropy.into_scalar().elem::<f32>();
                n_updates += 1;
            }
        }

        // Clear buffer for next rollout
        self.buffer.clear();
        self.training_step += 1;

        // Return average losses
        let n = n_updates as f32;
        (
            total_policy_loss / n,
            total_value_loss / n,
            total_entropy / n,
            (total_policy_loss + total_value_loss) / n,
        )
    }

    /// Compute the clipped PPO policy loss
    ///
    /// Implements the clipped surrogate objective:
    /// L = -E[min(r * A, clip(r, 1-ε, 1+ε) * A)]
    /// where r = π_new / π_old
    ///
    /// # Arguments
    ///
    /// * `action_logits` - Action logits from network [batch, num_actions]
    /// * `actions` - Actions taken [batch]
    /// * `old_log_probs` - Log probabilities from old policy [batch]
    /// * `advantages` - Advantage estimates [batch]
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// - `policy_loss` - Clipped policy loss (scalar)
    /// - `entropy` - Policy entropy (scalar)
    fn compute_policy_loss(
        &self,
        action_logits: &Tensor<B, 2>,
        actions: &Tensor<B, 1, Int>,
        old_log_probs: &Tensor<B, 1>,
        advantages: &Tensor<B, 1>,
    ) -> (Tensor<B, 1>, Tensor<B, 1>) {
        // Compute new log probabilities
        let log_probs = log_softmax(action_logits.clone(), 1);
        let new_log_probs = log_probs
            .gather(1, actions.clone().unsqueeze_dim(1))
            .squeeze(1);

        // Compute probability ratio: r = exp(log π_new - log π_old)
        let ratio = (new_log_probs.clone() - old_log_probs.clone()).exp();

        // Clipped surrogate objective
        let surr1 = ratio.clone() * advantages.clone();
        let surr2 = ratio.clamp(
            1.0 - self.config.clip_epsilon,
            1.0 + self.config.clip_epsilon,
        ) * advantages.clone();

        // Policy loss: -E[min(surr1, surr2)]
        // We use mean instead of expected value
        let policy_loss = surr1.min_pair(surr2).neg().mean();

        // Entropy: -E[Σ π(a|s) * log π(a|s)]
        let probs = softmax(action_logits.clone(), 1);
        let log_probs_all = log_softmax(action_logits.clone(), 1);
        let entropy = (probs * log_probs_all).sum_dim(1).neg().mean();

        (policy_loss, entropy)
    }

    /// Compute the value function loss (MSE)
    ///
    /// L = E[(V(s) - R)²]
    ///
    /// # Arguments
    ///
    /// * `values` - Value estimates from network [batch, 1]
    /// * `returns` - Target returns [batch]
    ///
    /// # Returns
    ///
    /// Value loss (scalar)
    fn compute_value_loss(&self, values: &Tensor<B, 2>, returns: &Tensor<B, 1>) -> Tensor<B, 1> {
        let values = values.clone().squeeze(1); // [batch]
        let diff = values - returns.clone();
        (diff.clone() * diff).mean()
    }

    /// Get the current training step
    pub fn training_step(&self) -> usize {
        self.training_step
    }

    /// Get a reference to the neural network
    pub fn network(&self) -> &ActorCriticNetwork<B> {
        &self.network
    }

    /// Get a reference to the PPO configuration
    pub fn config(&self) -> &PPOConfig {
        &self.config
    }

    /// Get the grid height
    pub fn grid_height(&self) -> usize {
        self.grid_height
    }

    /// Get the grid width
    pub fn grid_width(&self) -> usize {
        self.grid_width
    }

    /// Get the number of episodes trained
    pub fn episodes_trained(&self) -> usize {
        self.episodes_trained
    }

    /// Increment the episode counter
    pub fn increment_episode(&mut self) {
        self.episodes_trained += 1;
    }
}

/// Sample an action from a categorical distribution
///
/// # Arguments
///
/// * `probs` - Action probabilities [1, num_actions]
///
/// # Returns
///
/// Sampled action index
fn sample_categorical<B: burn::tensor::backend::Backend>(probs: &Tensor<B, 2>) -> usize {
    let probs_data = probs.to_data();
    let probs_slice: Vec<f32> = probs_data.to_vec().expect("Failed to convert probs to vec");

    let mut rng = rand::thread_rng();
    let random_val: f32 = rng.sample(rand::distributions::Standard);
    let mut cumsum = 0.0;

    for (idx, &prob) in probs_slice.iter().enumerate() {
        cumsum += prob;
        if random_val < cumsum {
            return idx;
        }
    }

    // Fallback to last action
    probs_slice.len() - 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::GameConfig;
    use crate::rl::{ActorCriticConfig, SnakeEnvironment};
    use burn::backend::{
        Autodiff,
        ndarray::{NdArray, NdArrayDevice},
    };

    type TestBackend = Autodiff<NdArray<f32>>;
    type TestInferenceBackend = NdArray<f32>;

    fn create_test_agent() -> PPOAgent<TestBackend> {
        let device = NdArrayDevice::default();
        let network_config = ActorCriticConfig::new(10, 10);
        let network = network_config.init::<TestBackend>(&device);
        let mut ppo_config = PPOConfig::default();
        ppo_config.update_frequency = 128; // Smaller for tests
        ppo_config.batch_size = 32;

        PPOAgent::new(network, ppo_config, 10, 10, device)
    }

    fn create_test_observation() -> Tensor<TestInferenceBackend, 3> {
        let device = NdArrayDevice::default();
        Tensor::zeros([4, 10, 10], &device)
    }

    #[test]
    fn test_agent_creation() {
        let agent = create_test_agent();
        assert_eq!(agent.training_step(), 0);
        assert!(!agent.should_update());
    }

    #[test]
    fn test_select_action() {
        let agent = create_test_agent();
        let obs = create_test_observation();

        let (action, log_prob, value) = agent.select_action(obs);

        // Action should be in valid range
        assert!(action < 4);

        // Log probability should be negative
        assert!(log_prob < 0.0);

        // Value should be finite
        assert!(value.is_finite());
    }

    #[test]
    fn test_store_transition() {
        let mut agent = create_test_agent();
        let obs = create_test_observation();

        agent.store_transition(obs, 0, -1.0, 1.0, 0.5, false);

        assert!(!agent.should_update());
    }

    #[test]
    fn test_buffer_fills() {
        let mut agent = create_test_agent();
        let obs = create_test_observation();

        // Fill buffer to capacity (128)
        for _ in 0..128 {
            agent.store_transition(obs.clone(), 0, -1.0, 1.0, 0.5, false);
        }

        assert!(agent.should_update());
    }

    #[test]
    fn test_update_with_small_buffer() {
        let device = NdArrayDevice::default();
        let network_config = ActorCriticConfig::new(10, 10);
        let network = network_config.init::<TestBackend>(&device);
        let mut ppo_config = PPOConfig::default();
        ppo_config.update_frequency = 32; // Small buffer
        ppo_config.batch_size = 16;
        ppo_config.n_epochs = 2; // Fewer epochs for speed

        let mut agent = PPOAgent::new(network, ppo_config, 10, 10, device);

        // Fill buffer
        for _ in 0..32 {
            let obs = create_test_observation();
            agent.store_transition(obs, 0, -1.0, 1.0, 0.5, false);
        }

        assert!(agent.should_update());

        // Perform update
        let (policy_loss, value_loss, entropy, total_loss) = agent.update(0.5, false);

        // Check that losses are finite
        assert!(policy_loss.is_finite());
        assert!(value_loss.is_finite());
        assert!(entropy.is_finite());
        assert!(total_loss.is_finite());

        // Buffer should be cleared
        assert!(!agent.should_update());

        // Training step should increment
        assert_eq!(agent.training_step(), 1);
    }

    #[test]
    fn test_policy_loss_computation() {
        let agent = create_test_agent();
        let device = NdArrayDevice::default();

        // Create dummy data
        let action_logits = Tensor::from_floats([[1.0, 2.0, 3.0, 4.0]], &device);
        let actions = Tensor::from_ints([2], &device);
        let old_log_probs = Tensor::from_floats([-1.5], &device);
        let advantages = Tensor::from_floats([0.5], &device);

        let (policy_loss, entropy) =
            agent.compute_policy_loss(&action_logits, &actions, &old_log_probs, &advantages);

        // Loss should be a scalar (single element)
        assert_eq!(policy_loss.dims().len(), 1);
        assert_eq!(policy_loss.dims()[0], 1);

        // Entropy should be positive (negated in calculation)
        let entropy_val: f32 = entropy.into_scalar().elem();
        assert!(entropy_val > 0.0);
    }

    #[test]
    fn test_value_loss_computation() {
        let agent = create_test_agent();
        let device = NdArrayDevice::default();

        let values = Tensor::from_floats([[0.5], [0.8], [0.3]], &device);
        let returns = Tensor::from_floats([0.6, 0.7, 0.4], &device);

        let value_loss = agent.compute_value_loss(&values, &returns);

        // Loss should be a scalar
        assert_eq!(value_loss.dims().len(), 1);
        assert_eq!(value_loss.dims()[0], 1);

        // Loss should be non-negative (MSE)
        let loss_val: f32 = value_loss.into_scalar().elem();
        assert!(loss_val >= 0.0);
    }

    #[test]
    fn test_integration_with_environment() {
        let device = NdArrayDevice::default();

        // Create environment
        let game_config = GameConfig::new(10, 10);
        let mut env = SnakeEnvironment::<TestInferenceBackend>::new(game_config, device.clone());

        // Create agent
        let network_config = ActorCriticConfig::new(10, 10);
        let network = network_config.init::<TestBackend>(&device);
        let mut ppo_config = PPOConfig::default();
        ppo_config.update_frequency = 32;
        ppo_config.batch_size = 16;

        let mut agent = PPOAgent::new(network, ppo_config, 10, 10, device);

        // Collect some transitions
        let mut obs = env.reset();

        for _ in 0..32 {
            let (action, log_prob, value) = agent.select_action(obs.clone());
            let (next_obs, reward, done) = env.step(action);

            agent.store_transition(obs, action, log_prob, reward, value, done);

            if done {
                obs = env.reset();
            } else {
                obs = next_obs;
            }
        }

        // Should be ready for update
        assert!(agent.should_update());

        // Perform update
        let (_, _, last_value) = agent.select_action(obs);
        let (p_loss, v_loss, entropy, _) = agent.update(last_value, false);

        // Verify losses are reasonable
        assert!(p_loss.is_finite());
        assert!(v_loss.is_finite());
        assert!(entropy.is_finite());
    }
}
