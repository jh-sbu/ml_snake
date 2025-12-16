//! Training statistics tracking for PPO
//!
//! This module provides utilities for tracking and monitoring training progress,
//! including episode rewards, lengths, scores, and loss values.

use std::collections::VecDeque;

/// Training statistics tracker with rolling averages
///
/// Tracks episode-level metrics (rewards, lengths, scores) and training-level
/// metrics (policy loss, value loss, entropy) using rolling windows for
/// smoothed statistics.
///
/// # Example
///
/// ```rust
/// use ml_snake::metrics::TrainingStats;
///
/// let mut stats = TrainingStats::new(100);
///
/// // Record an episode
/// stats.record_episode(15.5, 150, 5);
///
/// // Record a training update
/// stats.record_update(0.02, 0.05, 0.8);
///
/// // Get statistics
/// println!("Mean reward: {}", stats.mean_episode_reward());
/// println!("{}", stats.format_summary());
/// ```
#[derive(Debug, Clone)]
pub struct TrainingStats {
    /// Episode rewards (rolling window)
    episode_rewards: VecDeque<f32>,

    /// Episode lengths in steps (rolling window)
    episode_lengths: VecDeque<usize>,

    /// Episode scores (food eaten) (rolling window)
    episode_scores: VecDeque<u32>,

    /// Policy losses (rolling window)
    policy_losses: VecDeque<f32>,

    /// Value losses (rolling window)
    value_losses: VecDeque<f32>,

    /// Entropy values (rolling window)
    entropies: VecDeque<f32>,

    /// Total number of episodes completed
    total_episodes: usize,

    /// Total number of environment steps taken
    total_steps: usize,

    /// Window size for rolling averages
    window_size: usize,
}

impl TrainingStats {
    /// Create a new training statistics tracker
    ///
    /// # Arguments
    ///
    /// * `window_size` - Number of recent values to keep for rolling averages
    ///
    /// # Example
    ///
    /// ```rust
    /// use ml_snake::metrics::TrainingStats;
    ///
    /// // Track last 100 episodes
    /// let stats = TrainingStats::new(100);
    /// ```
    pub fn new(window_size: usize) -> Self {
        Self {
            episode_rewards: VecDeque::with_capacity(window_size),
            episode_lengths: VecDeque::with_capacity(window_size),
            episode_scores: VecDeque::with_capacity(window_size),
            policy_losses: VecDeque::with_capacity(window_size),
            value_losses: VecDeque::with_capacity(window_size),
            entropies: VecDeque::with_capacity(window_size),
            total_episodes: 0,
            total_steps: 0,
            window_size,
        }
    }

    /// Record the completion of an episode
    ///
    /// # Arguments
    ///
    /// * `reward` - Total reward accumulated during the episode
    /// * `length` - Number of steps taken in the episode
    /// * `score` - Final score (e.g., number of food items eaten)
    ///
    /// # Example
    ///
    /// ```rust
    /// use ml_snake::metrics::TrainingStats;
    ///
    /// let mut stats = TrainingStats::new(100);
    /// stats.record_episode(15.5, 150, 5);
    ///
    /// assert_eq!(stats.total_episodes(), 1);
    /// assert_eq!(stats.total_steps(), 150);
    /// ```
    pub fn record_episode(&mut self, reward: f32, length: usize, score: u32) {
        Self::push_deque(&mut self.episode_rewards, reward, self.window_size);
        Self::push_deque(&mut self.episode_lengths, length, self.window_size);
        Self::push_deque(&mut self.episode_scores, score, self.window_size);
        self.total_episodes += 1;
        self.total_steps += length;
    }

    /// Record a training update
    ///
    /// # Arguments
    ///
    /// * `policy_loss` - Policy loss value from the update
    /// * `value_loss` - Value function loss from the update
    /// * `entropy` - Policy entropy from the update
    ///
    /// # Example
    ///
    /// ```rust
    /// use ml_snake::metrics::TrainingStats;
    ///
    /// let mut stats = TrainingStats::new(100);
    /// stats.record_update(0.02, 0.05, 0.8);
    ///
    /// assert!((stats.mean_policy_loss() - 0.02).abs() < 1e-5);
    /// ```
    pub fn record_update(&mut self, policy_loss: f32, value_loss: f32, entropy: f32) {
        Self::push_deque(&mut self.policy_losses, policy_loss, self.window_size);
        Self::push_deque(&mut self.value_losses, value_loss, self.window_size);
        Self::push_deque(&mut self.entropies, entropy, self.window_size);
    }

    /// Get the mean episode reward over the rolling window
    ///
    /// # Returns
    ///
    /// The average reward, or 0.0 if no episodes have been recorded
    pub fn mean_episode_reward(&self) -> f32 {
        self.mean(&self.episode_rewards)
    }

    /// Get the mean episode length over the rolling window
    ///
    /// # Returns
    ///
    /// The average episode length in steps
    pub fn mean_episode_length(&self) -> f32 {
        let sum: usize = self.episode_lengths.iter().sum();
        if self.episode_lengths.is_empty() {
            0.0
        } else {
            sum as f32 / self.episode_lengths.len() as f32
        }
    }

    /// Get the mean episode score over the rolling window
    ///
    /// # Returns
    ///
    /// The average score (e.g., food eaten)
    pub fn mean_episode_score(&self) -> f32 {
        let sum: u32 = self.episode_scores.iter().sum();
        if self.episode_scores.is_empty() {
            0.0
        } else {
            sum as f32 / self.episode_scores.len() as f32
        }
    }

    /// Get the mean policy loss over the rolling window
    ///
    /// # Returns
    ///
    /// The average policy loss, or 0.0 if no updates have been recorded
    pub fn mean_policy_loss(&self) -> f32 {
        self.mean(&self.policy_losses)
    }

    /// Get the mean value loss over the rolling window
    ///
    /// # Returns
    ///
    /// The average value loss, or 0.0 if no updates have been recorded
    pub fn mean_value_loss(&self) -> f32 {
        self.mean(&self.value_losses)
    }

    /// Get the mean entropy over the rolling window
    ///
    /// # Returns
    ///
    /// The average entropy, or 0.0 if no updates have been recorded
    pub fn mean_entropy(&self) -> f32 {
        self.mean(&self.entropies)
    }

    /// Get the total number of episodes completed
    pub fn total_episodes(&self) -> usize {
        self.total_episodes
    }

    /// Get the total number of environment steps taken
    pub fn total_steps(&self) -> usize {
        self.total_steps
    }

    /// Get the window size for rolling averages
    pub fn window_size(&self) -> usize {
        self.window_size
    }

    /// Format a summary of the current statistics
    ///
    /// # Returns
    ///
    /// A formatted string with key metrics
    ///
    /// # Example
    ///
    /// ```rust
    /// use ml_snake::metrics::TrainingStats;
    ///
    /// let mut stats = TrainingStats::new(100);
    /// stats.record_episode(15.5, 150, 5);
    /// stats.record_update(0.02, 0.05, 0.8);
    ///
    /// println!("{}", stats.format_summary());
    /// // Output: Episodes: 1 | Steps: 150 | Reward: 15.50 | Score: 5.00 | Len: 150.0 | P_Loss: 0.0200 | V_Loss: 0.0500 | Entropy: 0.8000
    /// ```
    pub fn format_summary(&self) -> String {
        format!(
            "Episodes: {} | Steps: {} | Reward: {:.2} | Score: {:.2} | Len: {:.1} | P_Loss: {:.4} | V_Loss: {:.4} | Entropy: {:.4}",
            self.total_episodes,
            self.total_steps,
            self.mean_episode_reward(),
            self.mean_episode_score(),
            self.mean_episode_length(),
            self.mean_policy_loss(),
            self.mean_value_loss(),
            self.mean_entropy(),
        )
    }

    /// Helper function to compute mean of a VecDeque<f32>
    fn mean(&self, deque: &VecDeque<f32>) -> f32 {
        if deque.is_empty() {
            0.0
        } else {
            deque.iter().sum::<f32>() / deque.len() as f32
        }
    }

    /// Helper function to push to a deque with size limit
    fn push_deque<T>(deque: &mut VecDeque<T>, value: T, window_size: usize) {
        if deque.len() >= window_size {
            deque.pop_front();
        }
        deque.push_back(value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let stats = TrainingStats::new(100);
        assert_eq!(stats.window_size(), 100);
        assert_eq!(stats.total_episodes(), 0);
        assert_eq!(stats.total_steps(), 0);
    }

    #[test]
    fn test_record_episode() {
        let mut stats = TrainingStats::new(100);
        stats.record_episode(10.0, 50, 3);

        assert_eq!(stats.total_episodes(), 1);
        assert_eq!(stats.total_steps(), 50);
        assert!((stats.mean_episode_reward() - 10.0).abs() < 1e-5);
        assert!((stats.mean_episode_length() - 50.0).abs() < 1e-5);
        assert!((stats.mean_episode_score() - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_record_update() {
        let mut stats = TrainingStats::new(100);
        stats.record_update(0.02, 0.05, 0.8);

        assert!((stats.mean_policy_loss() - 0.02).abs() < 1e-5);
        assert!((stats.mean_value_loss() - 0.05).abs() < 1e-5);
        assert!((stats.mean_entropy() - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_rolling_average() {
        let mut stats = TrainingStats::new(3);

        // Add 3 episodes
        stats.record_episode(1.0, 10, 1);
        stats.record_episode(2.0, 20, 2);
        stats.record_episode(3.0, 30, 3);

        assert_eq!(stats.total_episodes(), 3);
        assert!((stats.mean_episode_reward() - 2.0).abs() < 1e-5);

        // Add a 4th episode - should evict the first
        stats.record_episode(4.0, 40, 4);

        assert_eq!(stats.total_episodes(), 4);
        // Mean should now be (2.0 + 3.0 + 4.0) / 3 = 3.0
        assert!((stats.mean_episode_reward() - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_rolling_window_update() {
        let mut stats = TrainingStats::new(2);

        stats.record_update(0.1, 0.2, 0.9);
        stats.record_update(0.2, 0.3, 0.8);

        assert!((stats.mean_policy_loss() - 0.15).abs() < 1e-5);

        // Add a 3rd update - should evict the first
        stats.record_update(0.3, 0.4, 0.7);

        // Mean should now be (0.2 + 0.3) / 2 = 0.25
        assert!((stats.mean_policy_loss() - 0.25).abs() < 1e-5);
    }

    #[test]
    fn test_total_steps_accumulate() {
        let mut stats = TrainingStats::new(10);

        stats.record_episode(1.0, 10, 1);
        stats.record_episode(2.0, 20, 2);
        stats.record_episode(3.0, 30, 3);

        assert_eq!(stats.total_steps(), 60);
    }

    #[test]
    fn test_format_summary() {
        let mut stats = TrainingStats::new(100);
        stats.record_episode(15.5, 150, 5);
        stats.record_update(0.02, 0.05, 0.8);

        let summary = stats.format_summary();
        assert!(summary.contains("Episodes: 1"));
        assert!(summary.contains("Steps: 150"));
        assert!(summary.contains("Reward: 15.50"));
        assert!(summary.contains("Score: 5.00"));
        assert!(summary.contains("Len: 150.0"));
        assert!(summary.contains("P_Loss: 0.0200"));
        assert!(summary.contains("V_Loss: 0.0500"));
        assert!(summary.contains("Entropy: 0.8000"));
    }

    #[test]
    fn test_empty_stats() {
        let stats = TrainingStats::new(100);

        assert_eq!(stats.mean_episode_reward(), 0.0);
        assert_eq!(stats.mean_episode_length(), 0.0);
        assert_eq!(stats.mean_episode_score(), 0.0);
        assert_eq!(stats.mean_policy_loss(), 0.0);
        assert_eq!(stats.mean_value_loss(), 0.0);
        assert_eq!(stats.mean_entropy(), 0.0);
    }

    #[test]
    fn test_multiple_episodes_and_updates() {
        let mut stats = TrainingStats::new(10);

        for i in 0..5 {
            stats.record_episode(i as f32, i * 10, i as u32);
            stats.record_update(i as f32 * 0.01, i as f32 * 0.02, 1.0 - i as f32 * 0.1);
        }

        assert_eq!(stats.total_episodes(), 5);
        assert_eq!(stats.total_steps(), 0 + 10 + 20 + 30 + 40); // 100

        // Mean reward: (0 + 1 + 2 + 3 + 4) / 5 = 2.0
        assert!((stats.mean_episode_reward() - 2.0).abs() < 1e-5);
    }
}
