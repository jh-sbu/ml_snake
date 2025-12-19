//! Model persistence for saving and loading trained agents
//!
//! This module provides functionality to save and load trained PPO agents,
//! including both the network weights and training metadata. It uses Burn's
//! Record system for serialization.

use super::{ActorCriticConfig, ActorCriticNetwork, PPOAgent, PPOConfig};
use anyhow::{Context, Result};
use burn::{
    module::Module,
    record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder},
    tensor::backend::AutodiffBackend,
};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Metadata saved with the model
///
/// Contains configuration and training information needed to properly
/// reconstruct and use the saved model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// PPO configuration used during training
    pub ppo_config: PPOConfig,

    /// Grid height in cells
    pub grid_height: usize,

    /// Grid width in cells
    pub grid_width: usize,

    /// Total training steps completed
    pub training_steps: usize,

    /// Number of episodes trained
    pub episodes_trained: usize,

    /// Version identifier for compatibility checking
    pub version: String,
}

impl ModelMetadata {
    /// Create new metadata
    pub fn new(
        ppo_config: PPOConfig,
        grid_height: usize,
        grid_width: usize,
        training_steps: usize,
        episodes_trained: usize,
    ) -> Self {
        Self {
            ppo_config,
            grid_height,
            grid_width,
            training_steps,
            episodes_trained,
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }
}

/// Save a trained PPO agent to a file
///
/// Serializes both the neural network weights and training metadata to the
/// specified path. Creates parent directories if they don't exist.
///
/// The model is saved in two files:
/// - `<path>` - Network weights (Burn record format)
/// - `<path>.meta.json` - Metadata as JSON
///
/// # Arguments
///
/// * `agent` - The trained PPO agent to save
/// * `path` - Path where the model should be saved
///
/// # Returns
///
/// `Ok(())` on success, or an error if saving fails
pub fn save_model<B: AutodiffBackend>(agent: &PPOAgent<B>, path: &Path) -> Result<()> {
    // Create parent directories if needed
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create directory: {:?}", parent))?;
    }

    // Extract network and convert to record
    let network = agent.network();
    let record = network.clone().into_record();

    // Save network weights using Burn's recorder
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    recorder
        .record(record, path.to_path_buf())
        .context("Failed to save network weights")?;

    // Create metadata
    let metadata = ModelMetadata::new(
        agent.config().clone(),
        agent.grid_height(),
        agent.grid_width(),
        agent.training_step(),
        agent.episodes_trained(),
    );

    // Save metadata as JSON
    let meta_path = path.with_extension("meta.json");
    let meta_json =
        serde_json::to_string_pretty(&metadata).context("Failed to serialize metadata")?;
    std::fs::write(&meta_path, meta_json)
        .with_context(|| format!("Failed to write metadata to {:?}", meta_path))?;

    Ok(())
}

/// Load a trained network from a file
///
/// Deserializes a previously saved model, returning both the network and its
/// associated metadata.
///
/// # Arguments
///
/// * `path` - Path to the saved model file (without .meta.json extension)
/// * `device` - Device to load the model onto
///
/// # Returns
///
/// A tuple containing the loaded network and its metadata
pub fn load_network<B: AutodiffBackend>(
    path: &Path,
    device: &B::Device,
) -> Result<(ActorCriticNetwork<B>, ModelMetadata)> {
    // Load metadata first
    let meta_path = path.with_extension("meta.json");
    let meta_json = std::fs::read_to_string(&meta_path)
        .with_context(|| format!("Failed to read metadata from {:?}", meta_path))?;
    let metadata: ModelMetadata =
        serde_json::from_str(&meta_json).context("Failed to deserialize metadata")?;

    // Reconstruct network from metadata
    let network_config = ActorCriticConfig::new(metadata.grid_height, metadata.grid_width);
    let mut network = network_config.init::<B>(device);

    // Load network weights using Burn's recorder
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let record = recorder
        .load(path.to_path_buf(), device)
        .with_context(|| format!("Failed to load network weights from {:?}", path))?;

    network = network.load_record(record);

    Ok((network, metadata))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rl::{TrainingBackend, default_device};
    use tempfile::TempDir;

    #[test]
    fn test_metadata_creation() {
        let ppo_config = PPOConfig::default();
        let metadata = ModelMetadata::new(ppo_config, 20, 20, 1000, 100);

        assert_eq!(metadata.grid_height, 20);
        assert_eq!(metadata.grid_width, 20);
        assert_eq!(metadata.training_steps, 1000);
        assert_eq!(metadata.episodes_trained, 100);
    }

    #[test]
    fn test_metadata_serialization() {
        let ppo_config = PPOConfig::default();
        let metadata = ModelMetadata::new(ppo_config, 20, 20, 1000, 100);

        // Serialize
        let json = serde_json::to_string(&metadata).unwrap();

        // Deserialize
        let deserialized: ModelMetadata = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.grid_height, 20);
        assert_eq!(deserialized.grid_width, 20);
        assert_eq!(deserialized.training_steps, 1000);
    }
}
