from models.hopfield import HopfieldNetwork
from agents.mpc_controller import BioFlightMPC 
import numpy as np
import json
import os
 
class BioFlightAgent:
    def __init__(self, config):
        self.config = config
        
        # 1. Legacy Memory Setup
        self.patterns = self._load_patterns(config['patterns_path'])
        self.memory = HopfieldNetwork(size=len(next(iter(self.patterns.values()))))
        # Pass the whole dictionary so the Hopfield net can see the names and patterns
        self.memory.train(self.patterns)
        
        # 2. Modern Control Setup
        self.mpc = BioFlightMPC(horizon=15, dt=0.1)

    def _map_observation_to_pattern(self, obs):
        """Refined: Map continuous velocity/height to a binary pattern"""
        # exp: Is velocity positive? Is height above 1m?
        pattern = np.where(obs > 0, 1, -1) 
        # Pad or trim to match Hopfield size
        return pattern[:self.memory.size]

    def compute_action(self, current_state):
        # A. Use Hopfield to recall the intended behavior pattern
        current_pat = self._map_observation_to_pattern(current_state)
        # The * captures any extra info (like energy or steps) into a list
        recalled_pat, *extras = self.memory.recall(current_pat) 
        
        # B. convert recalled pattern to a Geometric Reference (The 'Geodesic')
        # In a real setup, this would query your CUSP manifold model
        ref_trajectory = self._generate_geodesic_from_memory(recalled_pat)
        
        # C. Use MPC to find the smooth force needed to follow that memory
        # This integrates the EBM weights (alpha, beta, gamma)
        weights = {'alpha': 1.0, 'beta': 0.1, 'gamma': 0.05}
        optimal_force = self.mpc.solve(current_state, ref_trajectory, weights)
        
        return optimal_force

    def _generate_geodesic_from_memory(self, pattern):
        """Placeholder for Riemannian trajectory generation"""
        # If pattern is 'glide', return a sequence of points moving forward
        return [np.array([i*0.1, 0, 1.0]) for i in range(16)]



    def _load_patterns(self, path):
        """Loads and validates behavioral attractors from a JSON file."""
        if not os.path.exists(path):
            print(f"Warning: Pattern file not found at {path}. Loading default glide.")
            # Fallback pattern if file is missing
            return {"glide": np.array([1, -1, 1, -1, 1])}

        with open(path, 'r') as f:
            data = json.load(f)

        # Convert lists to NumPy arrays
        patterns = {name: np.array(vec) for name, vec in data.items()}

        # Validation: Ensure all patterns are the same length for the Hopfield Net
        lengths = [len(v) for v in patterns.values()]
        if len(set(lengths)) > 1:
            raise ValueError(f"Pattern mismatch! All vectors in {path} must be the same length. Found: {lengths}")

        print(f"Successfully loaded {len(patterns)} behaviors from {path}")
        return patterns