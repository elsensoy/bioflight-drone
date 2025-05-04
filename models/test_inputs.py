import json
import numpy as np
import os
from typing import List 
# --- Configuration ---
PATTERN_SIZE = 24  # Increased pattern size
NUM_PATTERNS = 4   # We still have 4 base patterns
NOISE_LEVEL = 0.15 # Slightly reduced noise level 
PATTERN_NAMES = ["glide", "flap", "hover", "turn"]

# --- Functions ---

def generate_random_bipolar_pattern(size: int) -> List[int]:
    """Generates a random pattern of -1s and 1s."""
    # Generate random 0s and 1s, then map to -1s and 1s
    pattern = np.random.randint(0, 2, size=size) * 2 - 1
    return pattern.tolist()

def add_noise(pattern: List[int], noise_level: float) -> List[int]:
    """Adds noise by flipping bits with a certain probability."""
    pattern_arr = np.array(pattern)
    # Create a mask where True indicates a bit to flip
    flip_mask = np.random.rand(len(pattern_arr)) < noise_level
    noisy_arr = np.copy(pattern_arr)
    # Flip the bits at the masked locations
    noisy_arr[flip_mask] *= -1
    return noisy_arr.tolist()

# --- Main Script ---

if __name__ == "__main__":
    print(f"Generating {NUM_PATTERNS} patterns of size {PATTERN_SIZE}...")

    # Generate new random base patterns
    binary_patterns = {}
    for name in PATTERN_NAMES:
        binary_patterns[name] = generate_random_bipolar_pattern(PATTERN_SIZE)
        print(f"Generated '{name}': {binary_patterns[name]}")

    # Add noisy versions for testing
    noisy_patterns = {}
    for k, v in binary_patterns.items():
        noisy_patterns[f"{k}_noisy"] = add_noise(v, NOISE_LEVEL)

    # Merge clean and noisy patterns
    all_patterns = {**binary_patterns, **noisy_patterns}

    # Ensure the 'data' directory exists
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, "test_inputs.json")

    # Save patterns to JSON file
    try:
        with open(filepath, "w") as f:
            json.dump(all_patterns, f, indent=4)
        print(f"\nSuccessfully saved {len(all_patterns)} patterns (clean and noisy) to {filepath}")
    except IOError as e:
        print(f"\nError saving patterns to {filepath}: {e}")

