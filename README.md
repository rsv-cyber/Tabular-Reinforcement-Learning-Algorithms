# Tabular-Reinforcement-Learning-Algorithms

A comprehensive implementation of **tabular Reinforcement Learning algorithms** applied to the FrozenLake environment. All algorithms are implemented from scratch using only NumPy and Gymnasium.

## 📋 Table of Contents

- [Algorithms Implemented](#-algorithms-implemented)
- [Results](#-results)
- [Installation](#-installation)
- [Usage](#-usage)
- [File Structure](#-file-structure)
- [Algorithm Comparison](#-algorithm-comparison)
- [License](#-license)

## Algorithms Implemented

| Algorithm | Type | Key Feature |
|-----------|------|-------------|
| **Value Iteration** | Dynamic Programming | Bellman optimality update |
| **Policy Iteration** | Dynamic Programming | Policy evaluation + improvement |
| **First-Visit Monte Carlo** | Model-Free | Episodic learning with ε-greedy |
| **SARSA** | On-Policy TD | Learns Q-values from current policy |
| **Q-Learning** | Off-Policy TD | Learns optimal policy independent of behavior |

## Environment: FrozenLake

- **Description**: Agent navigates a 4x4 frozen lake from start (S) to goal (G) avoiding holes (H)
- **State Space**: 16 discrete states
- **Action Space**: 4 actions (← ↑ → ↓)
- **Reward**: +1 for reaching goal, 0 otherwise
- **Version**: Deterministic (`is_slippery=False`)

### Performance Comparison

| Algorithm | Training Episodes | Success Rate | Convergence |
|-----------|------------------|--------------|-------------|
| Value Iteration | N/A (offline) | 100% | 7 iterations |
| Policy Iteration | N/A (offline) | 100% | 2 iterations |
| First-Visit MC | 8,000 | ~95% | ~6,000 episodes |
| SARSA | 5,000 | 100% | ~1,000 episodes |
| Q-Learning | 5,000 | 100% | ~500 episodes |


## Installation

```bash
# Clone repository
git clone https://github.com/rsv-cyber/Tabular-Reinforcement-Learning-Algorithms.git
cd Tabular-Reinforcement-Learning-Algorithms

# Install dependencies
pip install numpy gymnasium
