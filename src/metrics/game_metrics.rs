use std::time::{Duration, Instant};

pub struct GameMetrics {
    pub start_time: Instant,
    pub elapsed_time: Duration,
    pub high_score: u32,
    pub games_played: u32,
}

impl GameMetrics {
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            elapsed_time: Duration::ZERO,
            high_score: 0,
            games_played: 0,
        }
    }

    pub fn update(&mut self) {
        self.elapsed_time = self.start_time.elapsed();
    }

    pub fn on_game_start(&mut self) {
        self.start_time = Instant::now();
        self.elapsed_time = Duration::ZERO;
    }

    pub fn on_game_over(&mut self, final_score: u32) {
        self.games_played += 1;
        if final_score > self.high_score {
            self.high_score = final_score;
        }
    }

    pub fn format_time(&self) -> String {
        let total_secs = self.elapsed_time.as_secs();
        let minutes = total_secs / 60;
        let seconds = total_secs % 60;
        format!("{:02}:{:02}", minutes, seconds)
    }
}

impl Default for GameMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_formatting() {
        let mut metrics = GameMetrics::new();
        metrics.elapsed_time = Duration::from_secs(125);
        assert_eq!(metrics.format_time(), "02:05");

        metrics.elapsed_time = Duration::from_secs(0);
        assert_eq!(metrics.format_time(), "00:00");

        metrics.elapsed_time = Duration::from_secs(3661);
        assert_eq!(metrics.format_time(), "61:01");
    }

    #[test]
    fn test_high_score_tracking() {
        let mut metrics = GameMetrics::new();

        metrics.on_game_over(10);
        assert_eq!(metrics.high_score, 10);
        assert_eq!(metrics.games_played, 1);

        metrics.on_game_over(5);
        assert_eq!(metrics.high_score, 10); // Should not decrease
        assert_eq!(metrics.games_played, 2);

        metrics.on_game_over(15);
        assert_eq!(metrics.high_score, 15); // Should update
        assert_eq!(metrics.games_played, 3);
    }

    #[test]
    fn test_game_start_resets_time() {
        let mut metrics = GameMetrics::new();
        std::thread::sleep(Duration::from_millis(50));
        metrics.update();

        assert!(metrics.elapsed_time.as_millis() >= 50);

        metrics.on_game_start();
        metrics.update();
        assert!(metrics.elapsed_time.as_millis() < 50);
    }
}
