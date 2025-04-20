from models.hopfield import HopfieldNetwork
import numpy as np

class DroneAgent:
    def __init__(self, config):
        # Load patterns (perhaps from config or a dedicated file)
        self.patterns = self._load_patterns(config['patterns_path'])
        pattern_size = len(next(iter(self.patterns.values())))

        # Initialize Hopfield network
        self.memory = HopfieldNetwork(size=pattern_size)
        self.memory.train(list(self.patterns.values()))

        # Store config if needed
        self.config = config

    def _load_patterns(self, path):
        # Implement loading logic (e.g., from JSON/YAML)
        # Example (matching current format):
        patterns_data = { # Load this from file instead
            "glide": [1, -1, -1, -1, -1], # ... etc
        }
        return {name: np.array(p) for name, p in patterns_data.items()}

    def choose_action(self, observation):
        # 1. Preprocess observation to get a pattern-like input
        #    (This is a CRUCIAL step - how does drone state map to a binary pattern?)
        current_state_pattern = self._map_observation_to_pattern(observation)

        # 2. Use Hopfield network to recall the closest stable behavior
        recalled_state, _ = self.memory.recall(current_state_pattern)
        recalled_behavior_name = self.identify_pattern(recalled_state)

        # 3. Select low-level action based on recalled behavior
        #    (e.g., if "glide", set motors to low thrust; if "flap", oscillate motors)
        action = self._map_behavior_to_action(recalled_behavior_name)

        print(f"Observation mapped to: {current_state_pattern}")
        print(f"Memory recalled: {recalled_state} ({recalled_behavior_name})")
        print(f"Selected Action: {action}")

        return action

    def _map_observation_to_pattern(self, observation):
        # --- Placeholder ---
        # How do continuous drone states (pos, vel, angles)
        # become a binary pattern [-1, 1]^N that matches  Hopfield input?
        # Maybe discretize key state variables? Use thresholds?
        # For now, we return a dummy noisy pattern for testing structure
        print(f"Received observation: {observation}") # Log the actual observation
        noisy_glide = np.array([1, -1, -1, 1, -1]) # Example noisy input
        return noisy_glide

    def _map_behavior_to_action(self, behavior_name):
        # --- Placeholder ---
        # Map the high-level behavior name to low-level drone commands
        # (e.g., target RPMs for 4 motors expected by gym-pybullet-drones)
        # This depends heavily on the simulation environment's action space.
        # Example for gym-pybullet-drones (4 motors):
        if behavior_name == "glide":
            return np.array([0.1, 0.1, 0.1, 0.1]) * 4000 # Low thrust
        elif behavior_name == "flap": # Flapping isn't really a quadrotor thing... maybe 'hover'?
             return np.array([0.5, 0.5, 0.5, 0.5]) * 4000 # Hover thrust (adjust)
        elif behavior_name == "turn_left":
             return np.array([0.6, 0.4, 0.6, 0.4]) * 4000 # Differential thrust for yaw
        else:
             return np.array([0.0, 0.0, 0.0, 0.0]) # Default: Do nothing / Land

    def identify_pattern(self, state):
        # (Reuse or adapt the logic from the original main.py)
        for name, pattern in self.patterns.items():
            if np.array_equal(state, pattern):
                return name
        return "Unknown"