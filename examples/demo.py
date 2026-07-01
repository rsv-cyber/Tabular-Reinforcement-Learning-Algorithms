"""
Demo script for Tabular Reinforcement Learning Algorithms
Quickly test all algorithms on FrozenLake
"""

import gymnasium as gym
from rl_algorithms import (
    ValueIteration,
    PolicyIteration,
    FirstVisitMC,
    SARSA,
    QLearningAgent,
    print_policy_grid,
    test_policy
)


def main():
    print("="*60)
    print("TABULAR RL DEMO ON FROZENLAKE")
    print("="*60)

    # Create environment
    env = gym.make("FrozenLake-v1", is_slippery=False)

    # Choose algorithm to test
    print("\nAvailable algorithms:")
    print("  1. Value Iteration")
    print("  2. Policy Iteration")
    print("  3. First-Visit Monte Carlo")
    print("  4. SARSA")
    print("  5. Q-Learning")
    print("  6. Run All")

    try:
        choice = int(input("\nSelect algorithm (1-6): "))
    except ValueError:
        choice = 6

    if choice == 1:
        # Value Iteration
        print("\nRunning Value Iteration...")
        agent = ValueIteration(env, discount_factor=0.95, theta=1e-6)
        state_values, q_values = agent.value_estimate()
        policy = agent.optimal_policy()
        print_policy_grid(policy, "Value Iteration")
        success = test_policy(env, policy)
        print(f"Success Rate: {success:.1f}%")

    elif choice == 2:
        # Policy Iteration
        print("\nRunning Policy Iteration...")
        agent = PolicyIteration(env, discount_factor=0.95, theta=1e-6)
        policy, state_values = agent.policy_iteration()
        print_policy_grid(policy, "Policy Iteration")
        success = test_policy(env, policy)
        print(f"Success Rate: {success:.1f}%")

    elif choice == 3:
        # Monte Carlo
        print("\nRunning First-Visit Monte Carlo...")
        agent = FirstVisitMC(
            env,
            discount_factor=0.95,
            epsilon=0.5,
            epsilon_decay=0.9995,
            min_epsilon=0.05
        )
        policy, q_values = agent.train(num_episodes=3000)
        print_policy_grid(policy, "Monte Carlo")
        success = test_policy(env, policy)
        print(f"Success Rate: {success:.1f}%")

    elif choice == 4:
        # SARSA
        print("\nRunning SARSA...")
        agent = SARSA(
            env,
            discount_factor=0.95,
            alpha=0.1,
            epsilon=0.5
        )
        policy, q_values, _, _ = agent.train(num_episodes=3000)
        print_policy_grid(policy, "SARSA")
        success = test_policy(env, policy)
        print(f"Success Rate: {success:.1f}%")

    elif choice == 5:
        # Q-Learning
        print("\nRunning Q-Learning...")
        agent = QLearningAgent(
            env,
            alpha=0.8,
            discount_factor=0.95,
            epsilon=0.8
        )
        policy, q_values = agent.train(num_episodes=3000)
        print_policy_grid(policy, "Q-Learning")
        success = test_policy(env, policy)
        print(f"Success Rate: {success:.1f}%")

    else:
        # Run All
        print("\nRunning all algorithms...")
        algorithms = [
            ("Value Iteration", ValueIteration(env, discount_factor=0.95, theta=1e-6), False),
            ("Policy Iteration", PolicyIteration(env, discount_factor=0.95, theta=1e-6), False),
            ("Monte Carlo", FirstVisitMC(env, discount_factor=0.95, epsilon=0.5, epsilon_decay=0.9995, min_epsilon=0.05), True),
            ("SARSA", SARSA(env, discount_factor=0.95, alpha=0.1, epsilon=0.5), True),
            ("Q-Learning", QLearningAgent(env, alpha=0.8, discount_factor=0.95, epsilon=0.8), True)
        ]

        for name, agent, train in algorithms:
            print(f"\n--- {name} ---")
            if name == "Value Iteration":
                state_values, q_values = agent.value_estimate()
                policy = agent.optimal_policy()
            elif name == "Policy Iteration":
                policy, state_values = agent.policy_iteration()
            elif train:
                policy, q_values = agent.train(num_episodes=2000)
            print_policy_grid(policy, name)
            success = test_policy(env, policy)
            print(f"Success Rate: {success:.1f}%")

    env.close()


if __name__ == "__main__":
    main()
