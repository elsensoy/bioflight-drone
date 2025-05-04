# core/pybullet_simulation.py
import json
# import torch # Commented out
import os
import sys
import random
import time
import pybullet as p
import pybullet_data
from typing import Dict, List, Tuple

# --- Configuration ---
# Use absolute paths for robustness
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path) # e.g., /path/to/bioflight-drone/core
project_root = os.path.dirname(script_dir) # e.g., /path/to/bioflight-drone

# Add project root to Python path to allow importing 'models'
sys.path.append(project_root)

# Construct the absolute path to the patterns file
PATTERNS_FILE = os.path.join(project_root, "data", "test_inputs.json")

# Hopfield Network Configuration
MAX_RECALL_STEPS = 50
RECALL_UPDATE_RULE = 'async' # 'async' or 'sync'

# PyBullet Configuration
SIMULATION_DURATION_SEC = 20 # How long to run the simulation
SIMULATION_TIME_STEP = 1./240. # Simulation step frequency
# Define which URDF to test loading
# TEST_URDF = "quadrotor.urdf"
TEST_URDF = "r2d2.urdf" # <--- Changed to samurai.urdf for testing
MODEL_START_POS = [0, 0, 1] # Adjusted start position if needed
MODEL_START_ORN = p.getQuaternionFromEuler([0, 0, 0])
CUE_TRIGGER_INTERVAL_SEC = 3 # How often to simulate a new cue

# --- Import Hopfield Network ---
# NOTE: Commenting out Hopfield import for debugging PyBullet crash
# try:
#     from models.hopfield_pytorch import HopfieldNetworkPyTorch as HopfieldNetwork
#     print("Using PyTorch implementation.")
# except ImportError:
#     print(f"Error: Could not import HopfieldNetworkPyTorch.")
#     print(f"Attempted to import from: {os.path.join(project_root, 'models', 'hopfield_pytorch.py')}")
#     sys.exit(1)

# Check for CUDA availability
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(f"Using device: {DEVICE}")

# --- Helper Functions (Hopfield related - commented out for debugging) ---
# ...(Hopfield functions remain commented out)...

# --- PyBullet Action Placeholders (Still needed for execute_behavior if uncommented) ---
# ...(Action functions remain commented out for debugging)...
def glide_action(drone_id): pass
def flap_action(drone_id): pass
def hover_action(drone_id): pass
def turn_action(drone_id): pass
def unknown_action(drone_id): pass
ACTION_MAP = { "glide": glide_action, "flap": flap_action, "hover": hover_action, "turn": turn_action, "Unknown": unknown_action }
# def execute_behavior(recognized_behavior: str, drone_id: int): pass


# --- Main Simulation Setup and Loop ---

if __name__ == "__main__":
    # # 1. Load patterns (Commented out)
    # # 2. Initialize and train the Hopfield network (Commented out)

    # 3. Initialize PyBullet
    print("Initializing PyBullet...")
    physicsClient = p.connect(p.GUI) # Connect with GUI
    # physicsClient = p.connect(p.DIRECT) # Try this if GUI crashes persist
    print("Setting search path...")
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    print("Setting physics parameters...")
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(SIMULATION_TIME_STEP)

    # Load ground plane
    print("Loading ground plane...")
    planeId = p.loadURDF("plane.urdf")
    print(f"Ground plane loaded with ID: {planeId}")

    # Construct full path to the test URDF
    pybullet_data_path = pybullet_data.getDataPath()
    model_urdf_full_path = os.path.join(pybullet_data_path, TEST_URDF) # Use TEST_URDF variable
    print(f"Attempting to load model model from: {model_urdf_full_path}")

    # Load model using the full path
    try:
        modelId = p.loadURDF(model_urdf_full_path, MODEL_START_POS, MODEL_START_ORN)
        print(f"Model '{TEST_URDF}' loaded with ID: {modelId}")
    except p.error as e:
        print(f"\n--- Error loading URDF: {TEST_URDF} ---")
        print(f"PyBullet error: {e}")
        print(f"Attempted path: {model_urdf_full_path}")
        p.disconnect()
        sys.exit(1)

    # 4. Simulation Loop (Simplified)
    start_time = time.time()
    print("\n--- Starting Simplified Simulation Loop ---")
    step_count = 0
    try:
        while (time.time() - start_time) < SIMULATION_DURATION_SEC:
            # --- Hopfield Decision Logic (Commented out) ---
            # --- Execute Behavior in PyBullet (Commented out) ---

            # --- Step Simulation ---
            p.stepSimulation()
            step_count += 1

            # Optional: Print progress periodically
            if step_count % 240 == 0: # Print every second (assuming 240Hz)
                 print(f"Simulation step {step_count}...")

            time.sleep(SIMULATION_TIME_STEP) # Standard PyBullet practice

    except Exception as e:
        # Catch potential exceptions during the loop
        print(f"\n--- Exception during simulation loop ---")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback

    finally:
        # 5. Cleanup
        print(f"\n--- Simulation Finished (Ran {step_count} steps) ---")
        print("Disconnecting PyBullet...")
        try:
            p.disconnect()
            print("PyBullet disconnected successfully.")
        except Exception as e:
            print(f"Exception during PyBullet disconnect: {e}")

