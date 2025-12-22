//! Performance metrics tracking for bottleneck identification
//!
//! This module provides utilities for tracking timing information across
//! different parts of the training pipeline to identify performance bottlenecks.
//! Uses RAII-based scopes for automatic timing and rolling windows for statistics.

use anyhow::{Context, Result};
use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::io::Write as IoWrite;
use std::path::Path;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Keys for different timing categories
///
/// Divided into coarse-grained (always collected when enabled) and
/// fine-grained (only when fine_grained flag is true) categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TimingKey {
    // Coarse-grained metrics (always collected when perf enabled)
    /// Full episode execution time
    Episode,
    /// Complete PPO update cycle
    PpoUpdate,
    /// Model checkpoint save time
    ModelSave,
    /// Model load time
    ModelLoad,

    // Fine-grained metrics (only when perf_fine_grained enabled)
    /// Neural network forward pass
    NetworkForward,
    /// Neural network backward pass
    NetworkBackward,
    /// Observation tensor creation
    ObservationCreate,
    /// Batch data retrieval from buffer
    BatchRetrieval,
    /// GAE advantage computation
    GaeComputation,
}

impl TimingKey {
    /// Get a human-readable name for display
    pub fn name(&self) -> &'static str {
        match self {
            TimingKey::Episode => "Episode",
            TimingKey::PpoUpdate => "PPO Update",
            TimingKey::ModelSave => "Model Save",
            TimingKey::ModelLoad => "Model Load",
            TimingKey::NetworkForward => "Network Forward",
            TimingKey::NetworkBackward => "Network Backward",
            TimingKey::ObservationCreate => "Observation Create",
            TimingKey::BatchRetrieval => "Batch Retrieval",
            TimingKey::GaeComputation => "GAE Computation",
        }
    }

    /// Check if this is a fine-grained metric
    pub fn is_fine_grained(&self) -> bool {
        matches!(
            self,
            TimingKey::NetworkForward
                | TimingKey::NetworkBackward
                | TimingKey::ObservationCreate
                | TimingKey::BatchRetrieval
                | TimingKey::GaeComputation
        )
    }
}

/// Statistics for a single timing category
#[derive(Debug, Clone)]
pub struct TimingStats {
    /// Number of measurements
    count: usize,
    /// Total accumulated time
    total_time: Duration,
    /// Minimum time observed
    min_time: Duration,
    /// Maximum time observed
    max_time: Duration,
    /// Recent timings for rolling average (window of last N)
    recent_times: VecDeque<Duration>,
}

impl TimingStats {
    /// Create new empty timing stats
    pub(crate) fn new(window_size: usize) -> Self {
        Self {
            count: 0,
            total_time: Duration::ZERO,
            min_time: Duration::MAX,
            max_time: Duration::ZERO,
            recent_times: VecDeque::with_capacity(window_size),
        }
    }

    /// Record a new timing measurement
    pub(crate) fn record(&mut self, duration: Duration, window_size: usize) {
        self.count += 1;
        self.total_time += duration;
        self.min_time = self.min_time.min(duration);
        self.max_time = self.max_time.max(duration);

        // Add to rolling window
        if self.recent_times.len() >= window_size {
            self.recent_times.pop_front();
        }
        self.recent_times.push_back(duration);
    }

    /// Get average time from rolling window
    pub fn avg_time(&self) -> Duration {
        if self.recent_times.is_empty() {
            Duration::ZERO
        } else {
            let sum: Duration = self.recent_times.iter().sum();
            sum / self.recent_times.len() as u32
        }
    }

    /// Get overall average time (total / count)
    pub fn overall_avg_time(&self) -> Duration {
        if self.count == 0 {
            Duration::ZERO
        } else {
            self.total_time / self.count as u32
        }
    }

    /// Calculate throughput (operations per second)
    pub fn throughput(&self) -> f64 {
        let avg = self.avg_time();
        if avg.is_zero() {
            0.0
        } else {
            1_000_000.0 / avg.as_micros() as f64
        }
    }
}

/// Performance metrics tracker
///
/// Collects timing information for different operations using RAII scopes.
/// Maintains rolling windows of recent measurements for statistics.
///
/// # Example
///
/// ```rust,ignore
/// use ml_snake::metrics::{PerformanceMetrics, TimingKey};
///
/// let mut perf = PerformanceMetrics::new(true, false);
///
/// {
///     let _timer = perf.start_scope(TimingKey::Episode);
///     // Episode code here...
/// } // Timer automatically records on drop
///
/// println!("{}", perf.format_summary());
/// ```
pub struct PerformanceMetrics {
    /// Whether metrics collection is enabled
    enabled: bool,
    /// Whether fine-grained metrics are collected
    fine_grained: bool,
    /// Statistics per timing key
    stats: HashMap<TimingKey, TimingStats>,
    /// Rolling window size for averaging
    window_size: usize,
}

impl PerformanceMetrics {
    /// Create a new performance metrics tracker
    ///
    /// # Arguments
    ///
    /// * `enabled` - Whether to collect any metrics
    /// * `fine_grained` - Whether to collect fine-grained metrics (higher overhead)
    ///
    /// # Example
    ///
    /// ```rust
    /// use ml_snake::metrics::PerformanceMetrics;
    ///
    /// // Enable coarse-grained metrics only
    /// let perf = PerformanceMetrics::new(true, false);
    ///
    /// // Enable all metrics including fine-grained
    /// let perf_detailed = PerformanceMetrics::new(true, true);
    ///
    /// // Disabled (no overhead)
    /// let perf_disabled = PerformanceMetrics::new(false, false);
    /// ```
    pub fn new(enabled: bool, fine_grained: bool) -> Self {
        Self {
            enabled,
            fine_grained,
            stats: HashMap::new(),
            window_size: 100,
        }
    }

    /// Start a timing scope for the given key
    ///
    /// Returns a RAII guard that automatically records elapsed time on drop.
    /// Returns `None` if metrics are disabled or if fine-grained metrics are
    /// disabled but the key is fine-grained.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// {
    ///     let _timer = perf.start_scope(TimingKey::Episode);
    ///     // Code to time...
    /// } // Automatically records on drop
    /// ```
    pub fn start_scope(&mut self, key: TimingKey) -> Option<PerformanceScope> {
        if !self.enabled {
            return None;
        }

        // Check if fine-grained metric is disabled
        if key.is_fine_grained() && !self.fine_grained {
            return None;
        }

        Some(PerformanceScope {
            metrics: self as *mut PerformanceMetrics,
            key,
            start: Instant::now(),
        })
    }

    /// Record a timing measurement manually
    ///
    /// Normally you should use `start_scope()` instead for automatic timing.
    ///
    /// # Arguments
    ///
    /// * `key` - The timing category
    /// * `duration` - The measured duration
    pub fn record(&mut self, key: TimingKey, duration: Duration) {
        if !self.enabled {
            return;
        }

        if key.is_fine_grained() && !self.fine_grained {
            return;
        }

        let stats = self
            .stats
            .entry(key)
            .or_insert_with(|| TimingStats::new(self.window_size));
        stats.record(duration, self.window_size);
    }

    /// Get statistics for a specific timing key
    pub fn get_stats(&self, key: &TimingKey) -> Option<&TimingStats> {
        self.stats.get(key)
    }

    /// Format a compact summary suitable for periodic logging
    ///
    /// Shows rolling average, min, max, and throughput for each metric.
    ///
    /// # Returns
    ///
    /// Formatted string like:
    /// ```text
    /// Performance (last 100):
    ///   Episode:     avg=42.3ms  min=35.1ms  max=58.7ms  (23.6/s)
    ///   PPO Update:  avg=156.2ms min=142.8ms max=178.4ms (6.4/s)
    /// ```
    pub fn format_summary(&self) -> String {
        if self.stats.is_empty() {
            return String::from("Performance: No data collected yet");
        }

        let mut lines = vec![format!("Performance (last {}):", self.window_size)];

        // Sort keys for consistent output
        let mut keys: Vec<_> = self.stats.keys().collect();
        keys.sort_by_key(|k| format!("{:?}", k));

        for key in keys {
            if let Some(stats) = self.stats.get(key) {
                if stats.count > 0 {
                    lines.push(format!(
                        "  {:<18} avg={:>7}  min={:>7}  max={:>7}  ({:.1}/s)",
                        format!("{}:", key.name()),
                        Self::format_duration(stats.avg_time()),
                        Self::format_duration(stats.min_time),
                        Self::format_duration(stats.max_time),
                        stats.throughput()
                    ));
                }
            }
        }

        lines.join("\n")
    }

    /// Format a detailed summary with percentages and total time
    ///
    /// Shows comprehensive statistics including total time spent in each
    /// operation and percentage of overall time.
    ///
    /// # Returns
    ///
    /// Formatted string with:
    /// - Total counts and times per operation
    /// - Percentage breakdown
    /// - Throughput metrics
    pub fn format_detailed(&self) -> String {
        if self.stats.is_empty() {
            return String::from("No performance data collected");
        }

        let mut lines = Vec::new();

        // Calculate total time across all operations
        let total_time: Duration = self.stats.values().map(|s| s.total_time).sum();

        lines.push(String::from("Operation Breakdown:"));

        // Sort by total time (descending)
        let mut entries: Vec<_> = self.stats.iter().collect();
        entries.sort_by(|a, b| b.1.total_time.cmp(&a.1.total_time));

        for (key, stats) in entries {
            if stats.count > 0 {
                let percentage = if !total_time.is_zero() {
                    (stats.total_time.as_secs_f64() / total_time.as_secs_f64()) * 100.0
                } else {
                    0.0
                };

                lines.push(format!(
                    "  {:<18} avg={:>7}  count={:<8}  total={:>9}  ({:>5.1}%)",
                    format!("{}:", key.name()),
                    Self::format_duration(stats.overall_avg_time()),
                    stats.count,
                    Self::format_duration_long(stats.total_time),
                    percentage
                ));
            }
        }

        lines.push(String::new());
        lines.push(String::from("Throughput:"));

        // Sort keys for throughput display
        let mut keys: Vec<_> = self.stats.keys().collect();
        keys.sort_by_key(|k| format!("{:?}", k));

        for key in keys {
            if let Some(stats) = self.stats.get(key) {
                if stats.count > 0 {
                    lines.push(format!(
                        "  {:<18} {:.1} ops/sec",
                        format!("{}:", key.name()),
                        stats.throughput()
                    ));
                }
            }
        }

        lines.push(String::new());
        lines.push(format!("Total time measured: {}", Self::format_duration_long(total_time)));

        lines.join("\n")
    }

    /// Export metrics to CSV file
    ///
    /// # Arguments
    ///
    /// * `path` - File path to write CSV
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, error if file I/O fails
    ///
    /// # Format
    ///
    /// ```text
    /// timestamp,metric,count,total_ms,avg_ms,min_ms,max_ms,throughput
    /// 2024-12-22T12:34:56,Episode,10000,421340.5,42.1,35.1,58.7,23.7
    /// ```
    pub fn export_csv(&self, path: &Path) -> Result<()> {
        let mut file = File::create(path)
            .with_context(|| format!("Failed to create CSV file at {:?}", path))?;

        // Write header
        writeln!(
            file,
            "timestamp,metric,count,total_ms,avg_ms,min_ms,max_ms,throughput"
        )?;

        // Get current timestamp
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        let timestamp_str = format!(
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}",
            1970 + (timestamp.as_secs() / 31_557_600), // Approximate year
            ((timestamp.as_secs() / 2_629_800) % 12) + 1, // Approximate month
            ((timestamp.as_secs() / 86_400) % 30) + 1, // Approximate day
            (timestamp.as_secs() / 3600) % 24,
            (timestamp.as_secs() / 60) % 60,
            timestamp.as_secs() % 60
        );

        // Write data rows
        let mut keys: Vec<_> = self.stats.keys().collect();
        keys.sort_by_key(|k| format!("{:?}", k));

        for key in keys {
            if let Some(stats) = self.stats.get(key) {
                if stats.count > 0 {
                    writeln!(
                        file,
                        "{},{},{},{:.1},{:.1},{:.1},{:.1},{:.2}",
                        timestamp_str,
                        key.name(),
                        stats.count,
                        stats.total_time.as_secs_f64() * 1000.0,
                        stats.overall_avg_time().as_secs_f64() * 1000.0,
                        stats.min_time.as_secs_f64() * 1000.0,
                        stats.max_time.as_secs_f64() * 1000.0,
                        stats.throughput()
                    )?;
                }
            }
        }

        file.flush()?;
        Ok(())
    }

    /// Format duration in compact form (e.g., "42.3ms", "1.2s")
    fn format_duration(duration: Duration) -> String {
        let micros = duration.as_micros();
        if micros < 1000 {
            format!("{}µs", micros)
        } else if micros < 1_000_000 {
            format!("{:.1}ms", micros as f64 / 1000.0)
        } else {
            format!("{:.2}s", duration.as_secs_f64())
        }
    }

    /// Format duration in long form (e.g., "7m01s", "2h15m")
    fn format_duration_long(duration: Duration) -> String {
        let total_secs = duration.as_secs();
        if total_secs < 60 {
            format!("{:.1}s", duration.as_secs_f64())
        } else if total_secs < 3600 {
            let mins = total_secs / 60;
            let secs = total_secs % 60;
            format!("{}m{:02}s", mins, secs)
        } else {
            let hours = total_secs / 3600;
            let mins = (total_secs % 3600) / 60;
            format!("{}h{:02}m", hours, mins)
        }
    }
}

/// RAII guard for automatic timing
///
/// Automatically records elapsed time when dropped.
pub struct PerformanceScope {
    metrics: *mut PerformanceMetrics,
    key: TimingKey,
    start: Instant,
}

impl Drop for PerformanceScope {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed();
        // Safety: This pointer is only valid for the lifetime of the scope
        // which is guaranteed to be shorter than the PerformanceMetrics lifetime
        unsafe {
            (*self.metrics).record(self.key, elapsed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_new_metrics() {
        let metrics = PerformanceMetrics::new(true, false);
        assert!(metrics.enabled);
        assert!(!metrics.fine_grained);
        assert_eq!(metrics.window_size, 100);
    }

    #[test]
    fn test_disabled_metrics_no_scope() {
        let mut metrics = PerformanceMetrics::new(false, false);
        let scope = metrics.start_scope(TimingKey::Episode);
        assert!(scope.is_none());
    }

    #[test]
    fn test_fine_grained_disabled() {
        let mut metrics = PerformanceMetrics::new(true, false);

        // Coarse-grained should work
        let scope = metrics.start_scope(TimingKey::Episode);
        assert!(scope.is_some());

        // Fine-grained should not work
        let scope = metrics.start_scope(TimingKey::NetworkForward);
        assert!(scope.is_none());
    }

    #[test]
    fn test_fine_grained_enabled() {
        let mut metrics = PerformanceMetrics::new(true, true);

        // Both should work
        let scope1 = metrics.start_scope(TimingKey::Episode);
        assert!(scope1.is_some());

        let scope2 = metrics.start_scope(TimingKey::NetworkForward);
        assert!(scope2.is_some());
    }

    #[test]
    fn test_timing_recorded() {
        let mut metrics = PerformanceMetrics::new(true, false);

        {
            let _scope = metrics.start_scope(TimingKey::Episode);
            thread::sleep(Duration::from_millis(10));
        }

        let stats = metrics.get_stats(&TimingKey::Episode);
        assert!(stats.is_some());

        let stats = stats.unwrap();
        assert_eq!(stats.count, 1);
        assert!(stats.min_time >= Duration::from_millis(10));
        assert!(stats.avg_time() >= Duration::from_millis(10));
    }

    #[test]
    fn test_multiple_timings() {
        let mut metrics = PerformanceMetrics::new(true, false);

        for _ in 0..5 {
            let _scope = metrics.start_scope(TimingKey::Episode);
            thread::sleep(Duration::from_millis(5));
        }

        let stats = metrics.get_stats(&TimingKey::Episode).unwrap();
        assert_eq!(stats.count, 5);
        assert_eq!(stats.recent_times.len(), 5);
    }

    #[test]
    fn test_rolling_window() {
        let mut metrics = PerformanceMetrics::new(true, false);
        metrics.window_size = 3;

        for i in 0..5 {
            metrics.record(TimingKey::Episode, Duration::from_millis(i * 10));
        }

        let stats = metrics.get_stats(&TimingKey::Episode).unwrap();
        assert_eq!(stats.count, 5); // Total count
        assert_eq!(stats.recent_times.len(), 3); // Window size
    }

    #[test]
    fn test_format_summary() {
        let mut metrics = PerformanceMetrics::new(true, false);

        metrics.record(TimingKey::Episode, Duration::from_millis(42));
        metrics.record(TimingKey::PpoUpdate, Duration::from_millis(156));

        let summary = metrics.format_summary();
        assert!(summary.contains("Episode"));
        assert!(summary.contains("PPO Update"));
        assert!(summary.contains("42"));
        assert!(summary.contains("156"));
    }

    #[test]
    fn test_format_detailed() {
        let mut metrics = PerformanceMetrics::new(true, false);

        metrics.record(TimingKey::Episode, Duration::from_millis(100));
        metrics.record(TimingKey::PpoUpdate, Duration::from_millis(200));

        let detailed = metrics.format_detailed();
        assert!(detailed.contains("Operation Breakdown"));
        assert!(detailed.contains("Throughput"));
        assert!(detailed.contains("%"));
    }

    #[test]
    fn test_timing_key_name() {
        assert_eq!(TimingKey::Episode.name(), "Episode");
        assert_eq!(TimingKey::PpoUpdate.name(), "PPO Update");
        assert_eq!(TimingKey::NetworkForward.name(), "Network Forward");
    }

    #[test]
    fn test_timing_key_is_fine_grained() {
        assert!(!TimingKey::Episode.is_fine_grained());
        assert!(!TimingKey::PpoUpdate.is_fine_grained());
        assert!(TimingKey::NetworkForward.is_fine_grained());
        assert!(TimingKey::BatchRetrieval.is_fine_grained());
    }

    #[test]
    fn test_throughput_calculation() {
        let mut stats = TimingStats::new(10);
        stats.record(Duration::from_millis(100), 10); // 100ms = 10 ops/sec

        let throughput = stats.throughput();
        assert!((throughput - 10.0).abs() < 0.1);
    }
}
