# models/hopfield.py
import numpy as np

class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.W = np.zeros((size, size))
        self.patterns = {} # Store patterns for identification

    def train(self, patterns_dict):
        """ Trains the network using Hebbian learning on a dictionary of patterns. """
        self.patterns = patterns_dict
        patterns_list = list(patterns_dict.values())
        if not patterns_list:
            print("Warning: No patterns provided for training.")
            return

        if len(patterns_list[0]) != self.size:
             raise ValueError(f"Pattern size {len(patterns_list[0])} does not match network size {self.size}")

        # Hebbian learning
        for pattern in patterns_list:
            self.W += np.outer(pattern, pattern)
        np.fill_diagonal(self.W, 0) # No self-connections

    def calculate_energy(self, state):
        """ Calculates the energy of a given state. """
        return -0.5 * state @ self.W @ state.T

    def recall(self, input_pattern, max_iter=20, verbose=False):
        """ Recalls a pattern from a potentially noisy input using asynchronous updates. """
        if len(input_pattern) != self.size:
             raise ValueError(f"Input pattern size {len(input_pattern)} does not match network size {self.size}")

        state = input_pattern.copy()
        energy_trace = [self.calculate_energy(state)]
        stable = False

        for iteration in range(max_iter):
            prev_state = state.copy()
            # Asynchronous update (update one neuron at a time, randomly or sequentially)
            for i in np.random.permutation(self.size):
            # for i in range(self.size): # Sequential update alternative
                activation = self.W[i] @ state
                state[i] = 1 if activation >= 0 else -1

            current_energy = self.calculate_energy(state)
            energy_trace.append(current_energy)

            if verbose:
                 print(f"Iter {iteration + 1}: State={state}, Energy={current_energy:.2f}")

            # Check for convergence (state unchanged or energy minimum reached)
            if np.array_equal(state, prev_state) or energy_trace[-1] == energy_trace[-2]:
                 stable = True
                 if verbose: print(f"Converged after {iteration + 1} iterations.")
                 break

        if not stable and verbose:
            print("Warning: Did not converge within max iterations.")

        return state, energy_trace

    def identify_recalled_pattern(self, final_state):
        """ Identifies which stored pattern the final state matches. """
        for name, pattern in self.patterns.items():
            if np.array_equal(final_state, pattern):
                return name
        return "Unknown"

    def load_patterns_from_dict(self, patterns_dict):
         """ Utility to load patterns if not done during training """
         self.patterns = patterns_dict