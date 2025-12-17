//! PPO algorithm hyperparameter configuration

use serde::{Deserialize, Serialize};

/// Configuration for the PPO (Proximal Policy Optimization) algorithm
///
/// This struct contains all hyperparameters used by the PPO training algorithm.
/// Default values are based on common PPO implementations and tuned for the
/// Snake game environment.
///
/// # Example
///
/// ```rust
/// use ml_snake::rl::PPOConfig;
///
/// // Use default hyperparameters
/// let config = PPOConfig::default();
///
/// // Or customize specific parameters
/// let config = PPOConfig {
///     learning_rate: 1e-3,
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PPOConfig {
    /// Learning rate for the Adam optimizer
    ///
    /// Default: 3e-4
    pub learning_rate: f64,

    /// Discount factor for future rewards (gamma)
    ///
    /// Determines how much future rewards are valued relative to immediate rewards.
    /// Values closer to 1.0 make the agent more far-sighted.
    ///
    /// Default: 0.99
    pub gamma: f32,

    /// GAE (Generalized Advantage Estimation) lambda parameter
    ///
    /// Controls the bias-variance tradeoff in advantage estimation.
    /// Higher values (closer to 1.0) use more Monte Carlo estimates (higher variance, lower bias).
    /// Lower values use more TD estimates (lower variance, higher bias).
    ///
    /// Default: 0.95
    pub gae_lambda: f32,

    /// PPO clipping parameter (epsilon)
    ///
    /// Limits how much the policy can change in a single update.
    /// Prevents destructively large policy updates.
    ///
    /// Default: 0.2
    pub clip_epsilon: f32,

    /// Coefficient for the entropy bonus in the loss function
    ///
    /// Encourages exploration by adding entropy of the policy to the objective.
    /// Higher values lead to more exploration.
    ///
    /// Default: 0.01
    pub entropy_coef: f32,

    /// Coefficient for the value function loss
    ///
    /// Weights the importance of value function fitting relative to policy optimization.
    ///
    /// Default: 0.5
    pub value_coef: f32,

    /// Maximum gradient norm for gradient clipping
    ///
    /// Prevents exploding gradients by clipping the global gradient norm.
    ///
    /// Default: 0.5
    pub max_grad_norm: f32,

    /// Number of optimization epochs per PPO update
    ///
    /// How many times to iterate over the rollout buffer during each update.
    ///
    /// Default: 4
    pub n_epochs: usize,

    /// Minibatch size for training
    ///
    /// Number of samples to use in each gradient update step.
    ///
    /// Default: 64
    pub batch_size: usize,

    /// Number of environment steps to collect before performing a PPO update
    ///
    /// Also determines the buffer capacity.
    ///
    /// Default: 2048
    pub update_frequency: usize,
}

impl PPOConfig {
    /// Create a new configuration with default hyperparameters
    ///
    /// # Example
    ///
    /// ```rust
    /// use ml_snake::rl::PPOConfig;
    ///
    /// let config = PPOConfig::new();
    /// assert_eq!(config.learning_rate, 3e-4);
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    /// Validate configuration parameters
    ///
    /// Checks that all hyperparameters are in valid ranges.
    ///
    /// # Returns
    ///
    /// `Ok(())` if all parameters are valid, `Err(String)` with an error message otherwise.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ml_snake::rl::PPOConfig;
    ///
    /// let mut config = PPOConfig::default();
    /// assert!(config.validate().is_ok());
    ///
    /// config.learning_rate = -0.1;
    /// assert!(config.validate().is_err());
    /// ```
    pub fn validate(&self) -> Result<(), String> {
        if self.learning_rate <= 0.0 {
            return Err(format!(
                "learning_rate must be positive, got {}",
                self.learning_rate
            ));
        }

        if !(0.0..=1.0).contains(&self.gamma) {
            return Err(format!(
                "gamma must be in [0, 1], got {}",
                self.gamma
            ));
        }

        if !(0.0..=1.0).contains(&self.gae_lambda) {
            return Err(format!(
                "gae_lambda must be in [0, 1], got {}",
                self.gae_lambda
            ));
        }

        if self.clip_epsilon <= 0.0 || self.clip_epsilon > 1.0 {
            return Err(format!(
                "clip_epsilon must be in (0, 1], got {}",
                self.clip_epsilon
            ));
        }

        if self.entropy_coef < 0.0 {
            return Err(format!(
                "entropy_coef must be non-negative, got {}",
                self.entropy_coef
            ));
        }

        if self.value_coef < 0.0 {
            return Err(format!(
                "value_coef must be non-negative, got {}",
                self.value_coef
            ));
        }

        if self.max_grad_norm <= 0.0 {
            return Err(format!(
                "max_grad_norm must be positive, got {}",
                self.max_grad_norm
            ));
        }

        if self.n_epochs == 0 {
            return Err("n_epochs must be at least 1".to_string());
        }

        if self.batch_size == 0 {
            return Err("batch_size must be at least 1".to_string());
        }

        if self.update_frequency == 0 {
            return Err("update_frequency must be at least 1".to_string());
        }

        if self.batch_size > self.update_frequency {
            return Err(format!(
                "batch_size ({}) cannot exceed update_frequency ({})",
                self.batch_size, self.update_frequency
            ));
        }

        Ok(())
    }
}

impl Default for PPOConfig {
    fn default() -> Self {
        Self {
            learning_rate: 3e-4,
            gamma: 0.99,
            gae_lambda: 0.95,
            clip_epsilon: 0.2,
            entropy_coef: 0.01,
            value_coef: 0.5,
            max_grad_norm: 0.5,
            n_epochs: 4,
            batch_size: 64,
            update_frequency: 2048,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = PPOConfig::default();
        assert_eq!(config.learning_rate, 3e-4);
        assert_eq!(config.gamma, 0.99);
        assert_eq!(config.gae_lambda, 0.95);
        assert_eq!(config.clip_epsilon, 0.2);
        assert_eq!(config.entropy_coef, 0.01);
        assert_eq!(config.value_coef, 0.5);
        assert_eq!(config.max_grad_norm, 0.5);
        assert_eq!(config.n_epochs, 4);
        assert_eq!(config.batch_size, 64);
        assert_eq!(config.update_frequency, 2048);
    }

    #[test]
    fn test_new_creates_default() {
        let config = PPOConfig::new();
        let default = PPOConfig::default();
        assert_eq!(config.learning_rate, default.learning_rate);
        assert_eq!(config.gamma, default.gamma);
    }

    #[test]
    fn test_default_config_is_valid() {
        let config = PPOConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validation_negative_learning_rate() {
        let mut config = PPOConfig::default();
        config.learning_rate = -0.1;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validation_gamma_out_of_range() {
        let mut config = PPOConfig::default();
        config.gamma = 1.5;
        assert!(config.validate().is_err());

        config.gamma = -0.1;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validation_gae_lambda_out_of_range() {
        let mut config = PPOConfig::default();
        config.gae_lambda = 1.5;
        assert!(config.validate().is_err());

        config.gae_lambda = -0.1;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validation_clip_epsilon_invalid() {
        let mut config = PPOConfig::default();
        config.clip_epsilon = 0.0;
        assert!(config.validate().is_err());

        config.clip_epsilon = 1.5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validation_negative_coefficients() {
        let mut config = PPOConfig::default();
        config.entropy_coef = -0.1;
        assert!(config.validate().is_err());

        config.entropy_coef = 0.01;
        config.value_coef = -0.5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validation_zero_epochs() {
        let mut config = PPOConfig::default();
        config.n_epochs = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validation_zero_batch_size() {
        let mut config = PPOConfig::default();
        config.batch_size = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validation_zero_update_frequency() {
        let mut config = PPOConfig::default();
        config.update_frequency = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validation_batch_size_exceeds_update_frequency() {
        let mut config = PPOConfig::default();
        config.batch_size = 3000;
        config.update_frequency = 2048;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validation_batch_size_equals_update_frequency() {
        let mut config = PPOConfig::default();
        config.batch_size = 2048;
        config.update_frequency = 2048;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_custom_config() {
        let config = PPOConfig {
            learning_rate: 1e-3,
            gamma: 0.95,
            n_epochs: 10,
            ..Default::default()
        };
        assert_eq!(config.learning_rate, 1e-3);
        assert_eq!(config.gamma, 0.95);
        assert_eq!(config.n_epochs, 10);
        assert_eq!(config.clip_epsilon, 0.2); // From default
        assert!(config.validate().is_ok());
    }
}
