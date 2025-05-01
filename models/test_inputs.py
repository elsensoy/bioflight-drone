import json
import numpy as np
import os
# Define clean behavioral patterns (as binary vectors)
patterns = {
    "glide":     [ 1,  1,  0,  0, -1, -1,  1,  0],
    "flap":      [-1, -1, 1,  1,  1,  1, -1,  1],
    "hover":     [ 0,  1,  0,  1,  0,  1,  0,  1],
    "turn":      [ 1, -1,  1, -1,  1, -1,  1, -1],
}

# Optional: normalize to -1 / +1
def normalize(x):
    return [1 if v > 0 else -1 for v in x]

binary_patterns = {k: normalize(v) for k, v in patterns.items()}

# Add noisy versions for testing
def add_noise(pattern, noise_level=0.2):
    pattern = np.array(pattern)
    flip_mask = np.random.rand(len(pattern)) < noise_level
    noisy = np.copy(pattern)
    noisy[flip_mask] *= -1
    return noisy.tolist()

noisy_patterns = {
    f"{k}_noisy": add_noise(v, 0.3)  # 30% noise
    for k, v in binary_patterns.items()
}

# Merge and save
all_patterns = {**binary_patterns, **noisy_patterns}

os.makedirs("data", exist_ok=True)

with open("data/test_inputs.json", "w") as f:
    json.dump(all_patterns, f, indent=4)

print("Saved patterns to data/test_inputs.json")


# The generate_test_inputs.py script creates a small dataset of simple flight behavior patterns—like "glide", "flap", "hover", and "turn"—represented as binary vectors that mimic drone maneuvers. It then makes slightly "noisy" versions of each pattern by randomly flipping some values to simulate imperfect or distorted inputs. All these patterns are saved as a JSON file called test_inputs.json in a data/ folder, which is used to train and test a Hopfield network’s ability to recall and recognize stable behavior patterns from noisy inputs.