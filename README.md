# Tabular-Reinforcement-Learning-Algorithms

A comprehensive implementation of **tabular Reinforcement Learning algorithms** applied to the FrozenLake environment. All algorithms are implemented from scratch using only NumPy and Gymnasium.

## 📋 Table of Contents

- [Algorithms Implemented](#-algorithms-implemented)
- [Environment: FrozenLake](#-environment-:-frozenlake)
- [Performance Comparison](#-performance-comparison)
- [Installation](#-installation)
- [Usage](#-usage)
- [File Structure](#-file-structure)


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
```

## Usage

### Option 1: Run All Algorithms at Once

```bash
python rl_algorithms.py
```

### Option 2: Run Individual Algorithms

```bash
# Example: Run Q-Learning only
from rl_algorithms import QLearningAgent
import gymnasium as gym

env = gym.make("FrozenLake-v1", is_slippery=False)
agent = QLearningAgent(env, alpha=0.8, gamma=0.95, epsilon=0.8)
policy, q_values = agent.train(num_episodes=5000)

# Visualize the policy
from rl_algorithms import print_policy_grid
print_policy_grid(policy, "Q-Learning Optimal Policy")
```

### Option 3: Interactive Jupyter Notebook

```bash
jupyter notebook Tabular_RL_Algorithms.ipynb
```

### Option 4: Custom Hyperparameter Tuning

```bash
# Modify parameters in rl_algorithms.py or pass them directly
vi_agent = ValueIteration(env, discount_factor=0.99, theta=1e-8)
mc_agent = FirstVisitMC(
    env, 
    discount_factor=0.95, 
    epsilon=0.7,          # Higher exploration
    epsilon_decay=0.999,  # Slower decay
    min_epsilon=0.02      # Lower minimum
)
```

## File Structure

Tabular-Reinforcement-Learning-Algorithms/
│
├── README.md                          # Project documentation
│
├── rl_algorithms.py                   # ALL algorithms in one file
│   ├── class ValueIteration
│   ├── class PolicyIteration  
│   ├── class FirstVisitMC
│   ├── class SARSA
│   └── class QLearningAgent
│
├── Tabular_RL_Algorithms.ipynb        # Jupyter notebook (original)
│
└── examples/
    └── demo.py                        # Demonstration script
