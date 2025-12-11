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
