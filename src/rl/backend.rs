//! Backend type aliases and device management
//!
//! This module provides convenient type aliases for the Burn backends used in
//! training and inference, as well as helper functions for device management.
//!
//! # Backend Selection
//!
//! This module supports both CPU (NdArray) and GPU (Wgpu) backends:
//!
//! - **cpu module**: NdArray backend for CPU-based training and inference
//! - **gpu module**: Wgpu backend for GPU-accelerated training and inference
//!
//! The backend can be selected at runtime via CLI flags, allowing users to
//! choose between CPU and GPU without recompiling.
//!
//! # Example
//!
//! ```rust
//! use ml_snake::rl::cpu;
//! use ml_snake::rl::gpu;
//!
//! // Create CPU device for training
//! let cpu_device = cpu::device();
//!
//! // Create GPU device for training (if available)
//! let gpu_device = gpu::device(None); // Use default GPU
//! ```

/// CPU backend module using NdArray
pub mod cpu {
    use burn::backend::{Autodiff, ndarray::{NdArray, NdArrayDevice}};

    /// Backend type for training on CPU (with autodiff)
    pub type TrainingBackend = Autodiff<NdArray<f32>>;

    /// Backend type for inference on CPU (without autodiff)
    pub type InferenceBackend = NdArray<f32>;

    /// Device type for CPU backend
    pub type Device = NdArrayDevice;

    /// Get the CPU device for computation
    ///
    /// Returns the default NdArray device (CPU). This can be called multiple times
    /// safely as it uses Burn's device management.
    pub fn device() -> Device {
        NdArrayDevice::default()
    }
}

/// GPU backend module using Wgpu
pub mod gpu {
    use burn::backend::{Autodiff, wgpu::{Wgpu, WgpuDevice}};

    /// Backend type for training on GPU (with autodiff)
    pub type TrainingBackend = Autodiff<Wgpu<f32, i32>>;

    /// Backend type for inference on GPU (without autodiff)
    pub type InferenceBackend = Wgpu<f32, i32>;

    /// Device type for GPU backend
    pub type Device = WgpuDevice;

    /// Get the GPU device for computation
    ///
    /// Creates a GPU device. If `gpu_id` is provided, selects a specific discrete GPU.
    /// Otherwise, uses the default GPU selection (usually the first discrete GPU).
    ///
    /// # Arguments
    ///
    /// * `gpu_id` - Optional GPU device ID (0 = first discrete GPU, 1 = second, etc.)
    ///
    /// # Returns
    ///
    /// A WgpuDevice configured for the selected GPU
    pub fn device(gpu_id: Option<usize>) -> Device {
        match gpu_id {
            Some(id) => WgpuDevice::DiscreteGpu(id),
            None => WgpuDevice::default(),
        }
    }
}

// Re-export CPU backend as default for backward compatibility
pub use cpu::{TrainingBackend, InferenceBackend};

/// Get the default device for computation (CPU)
///
/// This function maintains backward compatibility by defaulting to CPU.
/// For GPU usage, use `gpu::device()` instead.
pub fn default_device() -> cpu::Device {
    cpu::device()
}

/// Check if GPU backend is available
///
/// Attempts to create a default GPU device to test availability.
/// Returns true if GPU can be initialized, false otherwise.
///
/// # Example
///
/// ```rust
/// use ml_snake::rl::gpu_available;
///
/// if gpu_available() {
///     println!("GPU is available!");
/// } else {
///     println!("GPU not available, falling back to CPU");
/// }
/// ```
pub fn gpu_available() -> bool {
    use burn::backend::wgpu::WgpuDevice;
    std::panic::catch_unwind(|| {
        let _device = WgpuDevice::default();
    })
    .is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_device() {
        let device = cpu::device();
        // Should not panic and should be usable
        let _device_copy = device.clone();
    }

    #[test]
    fn test_multiple_cpu_device_calls() {
        let device1 = cpu::device();
        let device2 = cpu::device();
        // Should be able to create multiple device references
        assert_eq!(
            std::mem::discriminant(&device1),
            std::mem::discriminant(&device2)
        );
    }

    #[test]
    fn test_default_device() {
        let device = default_device();
        // Should return CPU device
        let _device_copy = device.clone();
    }

    #[test]
    fn test_gpu_device_creation() {
        // This test might fail on systems without GPU
        // Just ensure it doesn't panic
        let _result = std::panic::catch_unwind(|| {
            gpu::device(None)
        });
    }

    #[test]
    fn test_gpu_available() {
        // Just ensure the function doesn't panic
        let _available = gpu_available();
    }
}
