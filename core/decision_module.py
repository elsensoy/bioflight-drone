# core/decision_module.py
import json
import torch
import os
import sys
import random
from typing import Dict, List, Any, Tuple

# Configuration 
# Use absolute paths for robustness
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path) # e.g., /path/to/bioflight-drone/core
project_root = os.path.dirname(script_dir) # e.g., /path/to/bioflight-drone

sys.path.append(project_root)
PATTERNS_FILE = os.path.join(project_root, "data", "test_inputs.json")

# Hopfield Network Configuration
MAX_RECALL_STEPS = 50
RECALL_UPDATE_RULE = 'async' # 'async' or 'sync'
#import hopfield
try:
    from models.hopfield_pytorch import HopfieldNetworkPyTorch as HopfieldNetwork
    print("Using PyTorch implementation.")
except ImportError:
    print(f"Error: Could not import HopfieldNetworkPyTorch.")
    print(f"Attempted to import from: {os.path.join(project_root, 'models', 'hopfield_pytorch.py')}")
    sys.exit(1)

# Check for cuda availability
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

#  Helper Functions 

def load_patterns(filepath: str) -> Dict[str, List[int]]:
    print(f"Attempting to load patterns from: {filepath}")
    try:
        with open(filepath, 'r') as f:
            patterns = json.load(f)
        print(f"Successfully loaded patterns from {filepath}")
        return patterns
    except FileNotFoundError:
        print(f"Error: Patterns file not found at {filepath}")
        print("Please ensure the 'data' directory exists in the project root and contains 'test_inputs.json'.")
        print("Project root detected as:", project_root)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}")
        sys.exit(1)

def initialize_and_train_network(clean_patterns: Dict[str, List[int]]) -> HopfieldNetwork:
    """Initializes and trains the Hopfield network."""
    if not clean_patterns:
        raise ValueError("Cannot train network with empty patterns dictionary.")

    pattern_size = len(next(iter(clean_patterns.values())))
    print(f"Initializing network with size: {pattern_size}")
    network = HopfieldNetwork(size=pattern_size, device=DEVICE)

    print("\n Training Network ")
    network.train(clean_patterns)
    return network

def simulate_input_cue(all_patterns: Dict[str, List[int]]) -> Tuple[str, List[int]]:
    """Simulates receiving an input cue (selects a random noisy pattern)."""
    noisy_keys = [k for k in all_patterns if k.endswith("_noisy")]
    if not noisy_keys:
        raise ValueError("No noisy patterns found in the loaded data to use as cues.")

    selected_key = random.choice(noisy_keys)
    cue_pattern = all_patterns[selected_key]
    print(f"\n--- Simulating Input Cue ---")
    print(f"Selected cue: '{selected_key}'")
    # print(f"Cue pattern: {cue_pattern}") # Optional: print the pattern itself
    return selected_key, cue_pattern

def recognize_behavior(network: HopfieldNetwork, cue_pattern: List[int]) -> str:
    """Uses the Hopfield network to recognize the behavior from the cue."""
    print("\n--- Attempting Recognition ---")
    # Prepare input pattern (convert to tensor)
    input_pattern_tensor = torch.tensor(cue_pattern, dtype=torch.float32, device=DEVICE)

    # Perform recall (verbose=False for cleaner output)
    final_state, energy_trace, converged = network.recall(
        input_pattern_tensor,
        max_steps=MAX_RECALL_STEPS,
        update_rule=RECALL_UPDATE_RULE,
        verbose=False 
    )

    if not converged:
        print("Warning: Recall did not converge.")
        # TODO: how to handle non-convergence (e.g., return 'Unknown' or raise error)

    # Identify the recalled pattern
    identified_pattern_name = network.identify_recalled_pattern(final_state)
    print(f"Recall process converged in {len(energy_trace)-1} steps.")
    print(f"Identified pattern name: '{identified_pattern_name}'")
    return identified_pattern_name

def execute_behavior(recognized_behavior: str):
    """Placeholder function to simulate executing the recognized behavior."""
    print("\n--- Executing Decision ---")
    if recognized_behavior != "Unknown":
        print(f"Decision: Initiate '{recognized_behavior}' behavior.")
        # In a real system, this would trigger motor commands, state changes, etc.
        # based on the 'recognized_behavior' string.
    else:
        print("Decision: Input cue not recognized. Maintain current state or default behavior.")
        # TODO: Handle unrecognized input

# --- Main Simulation Loop ---

if __name__ == "__main__":
    # 1. Load all patterns
    all_patterns = load_patterns(PATTERNS_FILE)

    # 2. Separate clean patterns for training
    clean_patterns = {k: v for k, v in all_patterns.items() if not k.endswith("_noisy")}

    # 3. Initialize and train the network
    try:
        hopfield_memory = initialize_and_train_network(clean_patterns)
    except ValueError as e:
        print(f"Error initializing network: {e}")
        sys.exit(1)

    # 4. Simulation Loop (run a few times for demonstration)
    num_simulations = 3
    for i in range(num_simulations):
        print(f"\n===== Simulation Cycle {i+1} =====")
        # 5. Simulate getting an input cue
        try:
            cue_name, cue_pattern = simulate_input_cue(all_patterns)
        except ValueError as e:
            print(f"Error simulating cue: {e}")
            break # Stop simulation if no noisy patterns

        # 6. Use the network to recognize the behavior
        recognized_behavior = recognize_behavior(hopfield_memory, cue_pattern)

        # 7. Execute the decided behavior (placeholder)
        execute_behavior(recognized_behavior)

    print("\nSimulation finished.")

