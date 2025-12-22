# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Snake game implementation using Ratatui (Rust TUI library) with machine learning capabilities. The project is in early development stages.

## Build and Run Commands

```bash
# Build the project
cargo build

# Run the project
cargo run

# Build with optimizations (release mode)
cargo build --release

# Run with release optimizations
cargo run --release

# Check code without building
cargo check

# Run tests
cargo test

# Run a specific test
cargo test <test_name>

# Run tests with output
cargo test -- --nocapture

# Format code
cargo fmt

# Check formatting without modifying
cargo fmt -- --check

# Run clippy linter
cargo clippy

# Run clippy with all warnings
cargo clippy -- -W clippy::all
```

## Architecture

The project uses Rust 2024 edition and follows a modular architecture with clean separation of concerns:

### Module Structure
- **`game/`**: Pure game logic (no I/O) - state, actions, engine, collision detection
- **`render/`**: Ratatui-based TUI rendering
- **`input/`**: Keyboard input handling
- **`rl/`**: Reinforcement learning components (environment, PPO agent, neural network, training)
- **`metrics/`**: Game and training statistics tracking
- **`modes/`**: Three main execution modes (human, train, visualize)

### Key Design Principles
1. **Pure Game Engine**: Game logic is independent of rendering and I/O
2. **Clean Separation**: Game, rendering, and RL are decoupled
3. **Backend Generic**: Uses Burn's Backend trait for CPU/GPU flexibility
4. **Mode Isolation**: Each execution mode is self-contained

### RL Implementation
- **Algorithm**: PPO (Proximal Policy Optimization) with GAE
- **Network**: Actor-Critic CNN processing 4-channel grid observations
- **Input**: Grid vision (snake body/head, food, walls as separate channels)
- **Framework**: Burn ML library (version 0.16)

### Three Operating Modes

```bash
# Human play mode - play snake with keyboard
cargo run -- --mode human

# Training mode - train RL agent with PPO
cargo run -- --mode train --episodes 10000 --save-path models/snake.bin

# Visualization mode - watch trained agent play
cargo run -- --mode visualize --model-path models/snake.bin
```

## GPU Support

The project supports both CPU (NdArray) and GPU (Wgpu) backends for accelerated training and inference.

### Backend Selection

```bash
# Auto-detect best available backend (default)
cargo run --release -- --mode train --episodes 10000

# Explicitly use CPU backend
cargo run --release -- --mode train --backend cpu --episodes 10000

# Explicitly use GPU backend (requires compatible GPU and drivers)
cargo run --release -- --mode train --backend gpu --episodes 10000

# Use specific GPU device (multi-GPU systems)
cargo run --release -- --mode train --backend gpu --gpu-device 0
```

### Backend Features

**CPU Backend (NdArray)**
- Works on all systems
- No additional dependencies
- Optimized batch size: 32
- Update frequency: 2048 steps

**GPU Backend (Wgpu)**
- Cross-platform GPU support (NVIDIA, AMD, Intel)
- Requires up-to-date graphics drivers
- Requires Vulkan (Linux/Windows), Metal (macOS), or DirectX 12 (Windows)
- Optimized batch size: 256 (8x larger for better GPU utilization)
- Update frequency: 4096 steps (2x larger)
- Expected speedup: 3-6x faster training

### Cross-Backend Compatibility

Models trained on one backend can be loaded on any other backend:
```bash
# Train on CPU
cargo run --release -- --mode train --backend cpu --save-path models/cpu_model.bin --episodes 1000

# Visualize on GPU
cargo run --release -- --mode visualize --backend gpu --model-path models/cpu_model.bin

# Train on GPU
cargo run --release -- --mode train --backend gpu --save-path models/gpu_model.bin --episodes 1000

# Visualize on CPU
cargo run --release -- --mode visualize --backend cpu --model-path models/gpu_model.bin
```

### GPU Requirements

To use the GPU backend, ensure you have:
- **NVIDIA GPUs**: Latest NVIDIA drivers with Vulkan support
- **AMD GPUs**: Latest AMD drivers with Vulkan support (amdgpu/amdvlk)
- **Intel GPUs**: Latest Intel drivers with Vulkan support
- **macOS**: Metal support (built-in on modern macOS)
- **Windows**: DirectX 12 or Vulkan support

If GPU is not available or initialization fails, the auto-detection will gracefully fall back to CPU.

## Performance Metrics

The project includes comprehensive performance metrics for identifying bottlenecks during development and optimization.

### Enabling Performance Metrics

```bash
# Enable basic performance metrics (coarse-grained)
cargo run --release -- --mode train --episodes 1000 --perf

# Enable detailed metrics with fine-grained timing (higher overhead)
cargo run --release -- --mode train --episodes 1000 --perf --perf-fine-grained

# Export metrics to CSV for analysis
cargo run --release -- --mode train --episodes 1000 --perf --perf-output metrics.csv

# Combine all options
cargo run --release -- --mode train --episodes 1000 --perf --perf-fine-grained --perf-output metrics.csv
```

### Metrics Collected

**Coarse-grained metrics** (always collected when `--perf` enabled):
- **Episode Loop**: Total time per episode including all operations
- **PPO Update**: Full PPO update cycle (GAE computation + gradient updates)
- **Model Save**: Checkpoint and final model save operations

**Fine-grained metrics** (only with `--perf-fine-grained`):
- **Network Forward**: Individual neural network forward passes
- **Network Backward**: Gradient computation (backward pass)
- **Observation Create**: Observation tensor creation time
- **Batch Retrieval**: Retrieving batch data from replay buffer
- **GAE Computation**: Advantage estimation calculations

### Performance Output

**During Training** (every log_frequency episodes):
```
[12:34:56] [Elapsed: 00:15:23] [Episode 1000/10000]
  Episodes: 1000 | Steps: 150432 | Reward: 15.50 | Score: 5.00 | ...

  Performance (last 100):
    Episode:           avg=42.3ms  min=35.1ms  max=58.7ms  (23.6/s)
    PPO Update:        avg=156.2ms min=142.8ms max=178.4ms (6.4/s)
    Network Forward:   avg=8.4ms   min=7.2ms   max=12.1ms  (119.0/s)
```

**Final Summary** (at training completion):
```
======================================================================
Performance Metrics:
======================================================================
Operation Breakdown:
  Episode:           avg=42.1ms   count=10000  total=7m01s   (31.1%)
  PPO Update:        avg=158.3ms  count=2048   total=5m24s   (23.9%)
  Network Forward:   avg=8.3ms    count=50000  total=6m55s   (30.6%)
  Model Save:        avg=245.8ms  count=10     total=2.5s    (0.2%)

Throughput:
  Episode:           23.7 ops/sec
  PPO Update:        6.3 ops/sec
  Network Forward:   120.5 ops/sec
  Model Save:        4.1 ops/sec

Total time measured: 22m38s
======================================================================
```

**CSV Export Format**:
```csv
timestamp,metric,count,total_ms,avg_ms,min_ms,max_ms,throughput
2025-12-22T12:34:56,Episode,10000,421340.5,42.1,35.1,58.7,23.7
2025-12-22T12:34:56,PpoUpdate,2048,324198.2,158.3,142.8,178.4,6.3
2025-12-22T12:34:56,NetworkForward,50000,415000.0,8.3,7.2,12.1,120.5
```

### Performance Overhead

- **Disabled** (default): 0% overhead
- **Coarse-grained** (`--perf`): <1% overhead (~10-15 timings per episode)
- **Fine-grained** (`--perf --perf-fine-grained`): 2-5% overhead (hundreds of timings per episode)

Fine-grained metrics are recommended only for development and bottleneck analysis, not for production training runs.

### Use Cases

1. **Identify Bottlenecks**: Find which operations consume the most time
2. **Backend Comparison**: Compare CPU vs GPU performance with concrete metrics
3. **Optimization Validation**: Measure impact of code optimizations
4. **Batch Size Tuning**: Understand relationship between batch size and throughput
5. **Development Profiling**: Track performance during feature development
