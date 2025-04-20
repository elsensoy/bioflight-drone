import numpy as np
import matplotlib.pyplot as plt
import json
import os
from utils import load_test_cases
# Step 1: Define the behavior patterns
# Each behavior is a binary pattern (-1, 1)

'''We stored 5 binary behavior patterns: glide, flap, turn_left, turn_right, soar.

train the Hopfield network using Hebbian learning.

test the network with a noisy version of "glide": [1, -1, -1, 1, -1] (notice the 4th bit is flipped).

The network iterated and converged to a final state: [-1, -1, -1, -1, -1].'''

patterns = {
    "glide":      np.array([1, -1, -1, -1, -1]),
    "flap":       np.array([-1, 1, -1, -1, -1]),
    "turn_left":  np.array([-1, -1, 1, -1, -1]),
    "turn_right": np.array([-1, -1, -1, 1, -1]),
    "soar":       np.array([-1, -1, -1, -1, 1])
}

def train_hopfield(patterns):
    size = len(next(iter(patterns.values())))
    W = np.zeros((size, size))
    for pattern in patterns.values():
        W += np.outer(pattern, pattern)
    np.fill_diagonal(W, 0)
    return W

def energy(state, W):
    return -0.5 * state @ W @ state.T

def run_hopfield(W, input_pattern, max_iter=10):
    state = input_pattern.copy()
    energy_trace = [energy(state, W)]
    for _ in range(max_iter):
        for i in range(len(state)):
            raw = np.dot(W[i], state)
            state[i] = 1 if raw >= 0 else -1
        current_energy = energy(state, W)
        energy_trace.append(current_energy)
        if energy_trace[-1] == energy_trace[-2]:
            break
    return state, energy_trace

def identify_pattern(state, patterns):
    for name, pattern in patterns.items():
        if np.array_equal(state, pattern):
            return name
    return "Unknown"

def main():
    # Define behavior patterns (can be modular later)
    patterns = {
        "glide":      np.array([1, -1, -1, -1, -1]),
        "flap":       np.array([-1, 1, -1, -1, -1]),
        "turn_left":  np.array([-1, -1, 1, -1, -1]),
        "turn_right": np.array([-1, -1, -1, 1, -1]),
        "soar":       np.array([-1, -1, -1, -1, 1])
    }

    # Train the Hopfield network
    W = train_hopfield(patterns)

    # Load test cases from JSON
    test_cases = load_test_cases("test_inputs.json")

    # Run each test case through the network
    for test in test_cases:
        label = test["label"]
        test_input = np.array(test["input"])

        final_state, energy_trace = run_hopfield(W, test_input)
        matched_behavior = identify_pattern(final_state, patterns)

        print(f"\nTest Case: {label}")
        print("Input:", test_input)
        print("Final State:", final_state)
        print("Predicted Behavior:", matched_behavior)

        # Plot energy convergence
        plt.plot(energy_trace)
        plt.xlabel("Iteration")
        plt.ylabel("Energy")
        plt.title(f"Energy Convergence: {label}")
        plt.grid(True)
        plt.show()

# Python entry point
if __name__ == "__main__":
    main()
