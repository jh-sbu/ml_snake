//! Backend type aliases and device management
//!
//! This module provides convenient type aliases for the Burn backends used in
//! training and inference, as well as helper functions for device management.
//!
//! # Backend Selection
//!
//! - **TrainingBackend**: Autodiff-enabled NdArray backend for training (CPU)
//! - **InferenceBackend**: Plain NdArray backend for inference (CPU)
//!
//! NdArray backend is sufficient for the Snake environment given its small state
//! space and network size. GPU support (via Wgpu backend) could be added later
//! if needed for larger-scale training.
//!
//! # Example
//!
//! ```rust
//! use ml_snake::rl::{TrainingBackend, InferenceBackend, default_device};
//!
//! // Create device for training
//! let device = default_device();
//!
//! // Use with training components
//! // let network = ActorCriticConfig::new(20, 20).init::<TrainingBackend>(&device);
//! ```

use burn::backend::{
    Autodiff,
    ndarray::{NdArray, NdArrayDevice},
};

/// Backend type for training (with autodiff)
///
/// This is the backend used for training the PPO agent. It includes automatic
/// differentiation support needed for gradient-based optimization.
pub type TrainingBackend = Autodiff<NdArray<f32>>;

/// Backend type for inference (without autodiff)
///
/// This is the backend used for running trained models. It's more efficient
/// than the training backend since it doesn't track gradients.
pub type InferenceBackend = NdArray<f32>;

/// Get the default device for computation
///
/// Returns the default NdArray device (CPU). This can be called multiple times
/// safely as it uses Burn's device management.
///
/// # Example
///
/// ```rust
/// use ml_snake::rl::default_device;
///
/// let device = default_device();
/// // Use device with Burn tensors and modules
/// ```
pub fn default_device() -> NdArrayDevice {
    NdArrayDevice::default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_device() {
        let device = default_device();
        // Should not panic and should be usable
        let _device_copy = device.clone();
    }

    #[test]
    fn test_multiple_device_calls() {
        let device1 = default_device();
        let device2 = default_device();
        // Should be able to create multiple device references
        assert_eq!(
            std::mem::discriminant(&device1),
            std::mem::discriminant(&device2)
        );
    }
}
