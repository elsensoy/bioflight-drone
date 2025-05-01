# training/train_memory.py
import json
import torch
import matplotlib.pyplot as plt
import os
import sys
from typing import Dict, List, Tuple

# --- Configuration ---
 
script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

# Choose which implementation to use ('pytorch' or 'numpy')
IMPLEMENTATION = 'pytorch' # or 'numpy'

if IMPLEMENTATION == 'pytorch':
    try:
        from models.hopfield_pytorch import HopfieldNetworkPyTorch as HopfieldNetwork
        print("Using PyTorch implementation.")
    except ImportError:
        print("Error: Could not import HopfieldNetworkPyTorch. Make sure models/hopfield_pytorch.py exists.")
        sys.exit(1)
    # Check for CUDA availability
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")
elif IMPLEMENTATION == 'numpy':
    try:
        # Assuming the NumPy version is named HopfieldNetwork in models/hopfield.py
        from models.hopfield import HopfieldNetwork
        print("Using NumPy implementation.")
        DEVICE = None # NumPy runs on CPU
    except ImportError:
        print("Error: Could not import HopfieldNetwork from models/hopfield.py.")
        sys.exit(1)
else:
    print(f"Error: Unknown implementation '{IMPLEMENTATION}'. Choose 'pytorch' or 'numpy'.")
    sys.exit(1)


PATTERNS_FILE = os.path.join(parent_dir, "models/data", "test_inputs.json")
MAX_RECALL_STEPS = 50
RECALL_UPDATE_RULE = 'async' # 'async' or 'sync'
VERBOSE_RECALL = False # Set to True for detailed step-by-step recall output

# --- Helper Functions ---

def load_patterns(filepath: str) -> Dict[str, List[int]]:
    """Loads patterns from a JSON file."""
    try:
        with open(filepath, 'r') as f:
            patterns = json.load(f)
        print(f"Successfully loaded patterns from {filepath}")
        return patterns
    except FileNotFoundError:
        print(f"Error: Patterns file not found at {filepath}")
        print("Please run the script to generate 'test_inputs.json' first.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}")
        sys.exit(1)

def plot_energy_trace(energy_trace: List[float], title: str):
    """Plots the energy evolution during recall."""
    plt.figure(figsize=(8, 4))
    plt.plot(energy_trace, marker='o', linestyle='-')
    plt.title(title)
    plt.xlabel("Recall Step")
    plt.ylabel("Network Energy")
    plt.grid(True)
    plt.xticks(range(len(energy_trace))) # Ensure integer ticks for steps
    plt.tight_layout()
    plt.show()

# --- Main Training and Testing ---

if __name__ == "__main__":
    # 1. Load patterns
    all_patterns = load_patterns(PATTERNS_FILE)
    if not all_patterns:
        sys.exit(1)

    # Separate clean and noisy patterns
    clean_patterns = {k: v for k, v in all_patterns.items() if not k.endswith("_noisy")}
    noisy_patterns = {k: v for k, v in all_patterns.items() if k.endswith("_noisy")}

    if not clean_patterns:
        print("Error: No clean patterns found in the JSON file.")
        sys.exit(1)

    # Determine network size from the first pattern
    pattern_size = len(next(iter(clean_patterns.values())))
    print(f"Detected pattern size: {pattern_size}")

    # 2. Initialize and Train the Network
    if IMPLEMENTATION == 'pytorch':
        network = HopfieldNetwork(size=pattern_size, device=DEVICE)
    else: # NumPy
        network = HopfieldNetwork(size=pattern_size)

    print("\n--- Training Network ---")
    network.train(clean_patterns) # Train only on clean base patterns

    # 3. Test Recall with Noisy Patterns
    print("\n--- Testing Recall ---")
    successful_recalls = 0
    total_tests = 0
    first_plot_done = False # Flag to plot only the first recall process

    for noisy_key, noisy_pattern_list in noisy_patterns.items():
        total_tests += 1
        original_key = noisy_key.replace("_noisy", "")
        print(f"\nTesting recall for: '{noisy_key}' (should recall '{original_key}')")

        # Prepare input pattern (convert to tensor if using PyTorch)
        if IMPLEMENTATION == 'pytorch':
            input_pattern_tensor = torch.tensor(noisy_pattern_list, dtype=torch.float32, device=DEVICE)
            input_arg = input_pattern_tensor
        else:
            input_arg = noisy_pattern_list # Pass list or np.array directly

        # Perform recall
        final_state, energy_trace, converged = network.recall(
            input_arg,
            max_steps=MAX_RECALL_STEPS,
            update_rule=RECALL_UPDATE_RULE,
            verbose=VERBOSE_RECALL
        )

        # Identify the recalled pattern
        if IMPLEMENTATION == 'pytorch':
            # Ensure final_state is used for identification
            identified_pattern_name = network.identify_recalled_pattern(final_state)
            final_state_np = final_state.cpu().numpy().astype(int) # For printing
        else:
            #   NumPy version returns np.ndarray
            identified_pattern_name = network.identify_recalled_pattern(final_state)
            final_state_np = final_state.astype(int) # For printing


        print(f"  Converged: {converged} in {len(energy_trace)-1} steps.")
        print(f"  Final State: {final_state_np}")
        print(f"  Identified as: '{identified_pattern_name}'")

        # Check if recall was successful
        if identified_pattern_name == original_key:
            print("  Result: SUCCESS")
            successful_recalls += 1
        else:
            print(f"  Result: FAILED (Expected '{original_key}')")

        # 4. Visualize Energy (only for the first test case)
        if not first_plot_done and energy_trace:
            plot_title = f"Energy Evolution for '{noisy_key}' Recall ({RECALL_UPDATE_RULE.capitalize()} Update)"
            plot_energy_trace(energy_trace, plot_title)
            first_plot_done = True

    # --- Report Summary ---
    print("\n--- Recall Test Summary ---")
    if total_tests > 0:
        success_rate = (successful_recalls / total_tests) * 100
        print(f"Total noisy patterns tested: {total_tests}")
        print(f"Successful recalls: {successful_recalls}")
        print(f"Success rate: {success_rate:.2f}%")
    else:
        print("No noisy patterns were found to test.")

