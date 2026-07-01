"""
Tabular Reinforcement Learning Algorithms
All algorithms implemented from scratch using NumPy and Gymnasium
"""

import numpy as np
import gymnasium as gym
from collections import defaultdict


# ============================================
# 1. VALUE ITERATION
# ============================================

class ValueIteration:
    """Value Iteration algorithm using Bellman optimality updates"""
    
    def __init__(self, env, discount_factor, theta=1e-8):
        self.env = env.unwrapped
        self.discount_factor = discount_factor
        self.theta = theta
        self.n_action = env.action_space.n
        self.n_states = env.observation_space.n
        self.v_values = np.zeros(self.n_states)
        self.q_values = np.zeros((self.n_states, self.n_action))

    def value_estimate(self):
        delta = np.inf
        iteration = 0

        while delta > self.theta:
            delta = 0
            old_values = self.v_values.copy()

            for s in range(self.n_states):
                if s == self.n_states - 1:  # Goal state
                    continue

                for a in range(self.n_action):
                    action_value = 0
                    for prob, next_state, reward, done in self.env.P[s][a]:
                        action_value += prob * (reward + self.discount_factor * old_values[next_state])
                    self.q_values[s, a] = action_value

                self.v_values[s] = np.max(self.q_values[s, :])
                delta = max(delta, abs(old_values[s] - self.v_values[s]))

            iteration += 1

        print(f"Value iteration converged after {iteration} iterations")
        return self.v_values, self.q_values

    def optimal_policy(self):
        optimal_policy = np.zeros(self.n_states, dtype=int)
        for s in range(self.n_states):
            if s == self.n_states - 1:
                optimal_policy[s] = -1
            else:
                optimal_policy[s] = np.argmax(self.q_values[s, :])
        return optimal_policy


# ============================================
# 2. POLICY ITERATION
# ============================================

class PolicyIteration:
    """Policy Iteration algorithm with policy evaluation and improvement"""
    
    def __init__(self, env, discount_factor, theta=1e-8):
        self.env = env.unwrapped
        self.discount_factor = discount_factor
        self.theta = theta
        self.n_actions = env.action_space.n
        self.n_states = env.observation_space.n
        self.v_values = np.zeros(self.n_states)
        self.policy = np.ones((self.n_states, self.n_actions)) / self.n_actions

    def policy_evaluation(self):
        iteration = 0
        delta = np.inf

        while delta > self.theta:
            delta = 0
            for s in range(self.n_states):
                if s == self.n_states - 1:
                    continue
                old_values = self.v_values[s]
                new_value = 0
                for a in range(self.n_actions):
                    action_prob = self.policy[s, a]
                    action_value = 0
                    for prob, next_state, reward, done in self.env.P[s][a]:
                        action_value += prob * (reward + self.discount_factor * self.v_values[next_state])
                    new_value += action_prob * action_value
                self.v_values[s] = new_value
                delta = max(delta, abs(old_values - new_value))
            iteration += 1

        print(f"Policy evaluation converged after {iteration} iterations")

    def policy_improvement(self):
        policy_stable = True
        for s in range(self.n_states):
            if s == self.n_states - 1:
                continue
            old_action = np.argmax(self.policy[s])
            action_values = np.zeros(self.n_actions)
            for a in range(self.n_actions):
                for prob, next_state, reward, done in self.env.P[s][a]:
                    action_values[a] += prob * (reward + self.discount_factor * self.v_values[next_state])
            best_action = np.argmax(action_values)
            self.policy[s] = np.eye(self.n_actions)[best_action]
            if old_action != best_action:
                policy_stable = False
        return policy_stable

    def policy_iteration(self):
        iteration = 0
        while True:
            iteration += 1
            print(f"\n--- Policy Iteration {iteration} ---")
            print("Policy Evaluation:")
            self.policy_evaluation()
            print("Policy Improvement:")
            policy_stable = self.policy_improvement()
            if policy_stable:
                print(f"\n✓ Policy converged after {iteration} iterations!")
                break
        optimal_policy = np.argmax(self.policy, axis=1)
        return optimal_policy, self.v_values


# ============================================
# 3. FIRST-VISIT MONTE CARLO
# ============================================

class FirstVisitMC:
    """First-Visit Monte Carlo control with ε-greedy exploration"""
    
    def __init__(self, env, discount_factor, epsilon, epsilon_decay, min_epsilon):
        self.env = env.unwrapped
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n

        self.q_values = np.ones((self.n_states, self.n_actions)) / self.n_actions * 10
        self.returns = defaultdict(list)

    def generate_episode(self):
        state, info = self.env.reset()
        state = int(state)
        done = False
        episode = []

        while not done:
            if np.random.random() < self.epsilon:
                action = np.random.randint(self.n_actions)
            else:
                action = np.argmax(self.q_values[state])

            next_state, reward, terminated, truncated, info = self.env.step(action)
            next_state = int(next_state)
            done = terminated or truncated
            episode.append((state, action, reward))
            state = next_state

        return episode

    def train(self, num_episodes=5000):
        success_history = []

        for episode_num in range(num_episodes):
            G = 0
            visited = set()
            episode = self.generate_episode()

            for t in range(len(episode) - 1, -1, -1):
                state, action, reward = episode[t]
                G = reward + self.discount_factor * G

                if (state, action) not in visited:
                    self.returns[(state, action)].append(G)
                    visited.add((state, action))
                    self.q_values[state, action] = np.mean(self.returns[(state, action)])

            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            if episode[-1][2] > 0:
                success_history.append(1)
            else:
                success_history.append(0)

            if (episode_num + 1) % 500 == 0:
                recent_success = np.mean(success_history[-100:]) if success_history else 0
                print(f"  Episode {episode_num + 1}/{num_episodes} completed")
                print(f"    ε={self.epsilon:.4f}, Recent success rate: {recent_success*100:.1f}%")

        greedy_policy = np.argmax(self.q_values, axis=1)
        return greedy_policy, self.q_values


# ============================================
# 4. SARSA
# ============================================

class SARSA:
    """SARSA (State-Action-Reward-State-Action) on-policy TD control"""
    
    def __init__(self, env, alpha, discount_factor, epsilon, epsilon_decay=0.999, min_epsilon=0.01):
        self.env = env.unwrapped
        self.alpha = alpha
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.n_actions = env.action_space.n
        self.n_states = env.observation_space.n
        self.q_values = np.zeros((self.n_states, self.n_actions))

    def epsilon_greedy_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.q_values[state])

    def train(self, num_episodes=5000, max_steps_per_episode=100):
        episode_rewards = []
        success_history = []

        for episode_num in range(num_episodes):
            state, info = self.env.reset()
            state = int(state)
            action = self.epsilon_greedy_action(state)

            steps = 0
            episode_reward = 0

            while steps < max_steps_per_episode:
                next_state, reward, terminated, truncated, info = self.env.step(action)
                next_state = int(next_state)
                done = terminated or truncated

                next_action = self.epsilon_greedy_action(next_state)

                if done:
                    td_target = reward
                else:
                    td_target = reward + self.discount_factor * self.q_values[next_state, next_action]

                td_error = td_target - self.q_values[state, action]
                self.q_values[state, action] += self.alpha * td_error

                state = next_state
                action = next_action

                steps += 1
                episode_reward += reward

                if done:
                    break

            episode_rewards.append(episode_reward)
            success_history.append(1 if episode_reward > 0 else 0)

            if (episode_num + 1) % 500 == 0:
                recent_success = np.mean(success_history[-100:]) if success_history else 0
                print(f"  Episode {episode_num + 1}/{num_episodes} - "
                      f"ε={self.epsilon:.4f} - "
                      f"Success rate: {recent_success*100:.1f}% - "
                      f"Avg reward: {np.mean(episode_rewards[-100:]):.2f}")

        greedy_policy = np.argmax(self.q_values, axis=1)
        return greedy_policy, self.q_values, episode_rewards, success_history


# ============================================
# 5. Q-LEARNING
# ============================================

class QLearningAgent:
    """Q-Learning off-policy TD control"""
    
    def __init__(self, env, alpha, discount_factor, epsilon):
        self.env = env.unwrapped
        self.alpha = alpha
        self.discount_factor = discount_factor
        self.epsilon = epsilon

        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.q_values = np.zeros((self.n_states, self.n_actions))

    def train(self, num_episodes=5000):
        success_per_500 = []
        epsilon = self.epsilon

        for episode in range(num_episodes):
            state, _ = self.env.reset()
            state = int(state)
            done = False
            total_reward = 0

            while not done:
                # Epsilon-greedy action
                if np.random.random() < epsilon:
                    action = np.random.randint(self.n_actions)
                else:
                    action = np.argmax(self.q_values[state])

                # Take action
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                next_state = int(next_state)
                done = terminated or truncated
                total_reward += reward

                # Q-learning update
                if done:
                    self.q_values[state, action] += self.alpha * (reward - self.q_values[state, action])
                else:
                    self.q_values[state, action] += self.alpha * (
                        reward + self.discount_factor * np.max(self.q_values[next_state]) - self.q_values[state, action]
                    )

                state = next_state

            # Decay epsilon
            epsilon = max(0.05, epsilon * 0.995)

            # Track success
            if (episode + 1) % 500 == 0:
                test_successes = 0
                for _ in range(100):
                    s, _ = self.env.reset()
                    s = int(s)
                    s_done = False
                    while not s_done:
                        a = np.argmax(self.q_values[s])
                        ns, r, t, tr, _ = self.env.step(a)
                        s = int(ns)
                        s_done = t or tr
                        if r > 0:
                            test_successes += 1
                            break
                success_rate = test_successes
                success_per_500.append(success_rate)
                print(f"Episode {episode+1}: ε={epsilon:.3f}, Test Success Rate: {success_rate}%")

        policy = np.argmax(self.q_values, axis=1)
        return policy, self.q_values


# ============================================
# UTILITY FUNCTIONS
# ============================================

def print_policy_grid(policy, title="Optimal Policy"):
    """Print policy as a 4x4 grid with symbols"""
    action_symbols = {0: '←', 1: '↓', 2: '→', 3: '↑'}

    print(f"\n{title}:")
    print("   0  1  2  3")
    print("  ┌─────────┐")

    for i in range(4):
        row = f"{i} │"
        for j in range(4):
            state = i * 4 + j
            if state == 15:  # Goal
                row += " G "
            elif state in [5, 7, 11, 12]:  # Holes
                row += " H "
            else:
                row += f" {action_symbols[policy[state]]} "
        print(row)
    print("  └─────────┘")


def print_state_values(state_values):
    """Print state values as a 4x4 grid"""
    print("\nState Values:")
    for i in range(4):
        row = []
        for j in range(4):
            state = i * 4 + j
            row.append(f"{state_values[state]:7.4f}")
        print(" ".join(row))


def test_policy(env, policy, num_tests=100):
    """Test a policy and return success rate"""
    successes = 0
    for _ in range(num_tests):
        state, _ = env.reset()
        state = int(state)
        done = False
        while not done:
            action = policy[state]
            next_state, reward, terminated, truncated, _ = env.step(action)
            state = int(next_state)
            done = terminated or truncated
            if reward > 0:
                successes += 1
                break
    return successes / num_tests * 100


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    print("="*60)
    print("TABULAR REINFORCEMENT LEARNING ALGORITHMS")
    print("="*60)

    # Create environment
    env = gym.make("FrozenLake-v1", is_slippery=False)

    # ============================================
    # 1. Value Iteration
    # ============================================
    print("\n" + "="*60)
    print("1. VALUE ITERATION")
    print("="*60)

    vi_agent = ValueIteration(env, discount_factor=0.95, theta=1e-6)
    state_values, q_values = vi_agent.value_estimate()
    vi_policy = vi_agent.optimal_policy()
    print_policy_grid(vi_policy, "Value Iteration Optimal Policy")
    print_state_values(state_values)

    # ============================================
    # 2. Policy Iteration
    # ============================================
    print("\n" + "="*60)
    print("2. POLICY ITERATION")
    print("="*60)

    pi_agent = PolicyIteration(env, discount_factor=0.95, theta=1e-6)
    pi_policy, pi_values = pi_agent.policy_iteration()
    print_policy_grid(pi_policy, "Policy Iteration Optimal Policy")
    print_state_values(pi_values)

    # ============================================
    # 3. First-Visit Monte Carlo
    # ============================================
    print("\n" + "="*60)
    print("3. FIRST-VISIT MONTE CARLO")
    print("="*60)

    mc_agent = FirstVisitMC(
        env,
        discount_factor=0.95,
        epsilon=0.5,
        epsilon_decay=0.9995,
        min_epsilon=0.05
    )
    mc_policy, mc_q_values = mc_agent.train(num_episodes=8000)
    print_policy_grid(mc_policy, "First-Visit MC Optimal Policy")
    mc_success = test_policy(env, mc_policy)
    print(f"\nMC Success Rate: {mc_success:.1f}%")

    # ============================================
    # 4. SARSA
    # ============================================
    print("\n" + "="*60)
    print("4. SARSA")
    print("="*60)

    sarsa_agent = SARSA(
        env,
        discount_factor=0.95,
        alpha=0.1,
        epsilon=0.5
    )
    sarsa_policy, sarsa_q_values, _, _ = sarsa_agent.train(num_episodes=5000)
    print_policy_grid(sarsa_policy, "SARSA Optimal Policy")
    sarsa_success = test_policy(env, sarsa_policy)
    print(f"\nSARSA Success Rate: {sarsa_success:.1f}%")

    # ============================================
    # 5. Q-Learning
    # ============================================
    print("\n" + "="*60)
    print("5. Q-LEARNING")
    print("="*60)

    ql_agent = QLearningAgent(
        env,
        alpha=0.8,
        discount_factor=0.95,
        epsilon=0.8
    )
    ql_policy, ql_q_values = ql_agent.train(num_episodes=5000)
    print_policy_grid(ql_policy, "Q-Learning Optimal Policy")
    ql_success = test_policy(env, ql_policy)
    print(f"\nQ-Learning Success Rate: {ql_success:.1f}%")

    # ============================================
    # Summary
    # ============================================
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Value Iteration:     ✓ Converged")
    print(f"Policy Iteration:    ✓ Converged")
    print(f"Monte Carlo:         ✓ {mc_success:.1f}% success rate")
    print(f"SARSA:               ✓ {sarsa_success:.1f}% success rate")
    print(f"Q-Learning:          ✓ {ql_success:.1f}% success rate")

    env.close()
