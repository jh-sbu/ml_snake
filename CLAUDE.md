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
