# core/pybullet_simulation.py
import json
import torch  
import os
import sys
import random
import time
import pybullet as p
import pybullet_data
from typing import Dict, List, Tuple

# Configuration 
# Use absolute paths for robustness
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path) # /path/to/bioflight-drone/core
project_root = os.path.dirname(script_dir) #  /path/to/bioflight-drone
trajectory_log = []

# Add project root to Python path to allow importing 'models'
sys.path.append(project_root)
PATTERNS_FILE = os.path.join(project_root, "data", "test_inputs.json")

# Hopfield Network Configuration
MAX_RECALL_STEPS = 50
RECALL_UPDATE_RULE = 'async'  

# PyBullet Configuration
SIMULATION_DURATION_SEC = 15 # Run for 15 seconds
SIMULATION_TIME_STEP = 1./240. # Simulation step frequency

# --- Path to Local URDF ---
# Define the path to the URDF relative to the project root
LOCAL_URDF_DIR = os.path.join(project_root, "urdfs")
TEST_URDF_FILENAME = "quadrotor.urdf" # The file manually downloaded
MODEL_URDF_FULL_PATH = os.path.join(LOCAL_URDF_DIR, TEST_URDF_FILENAME)

MODEL_START_POS = [0, 0, 1] # Reset start height for quadrotor
MODEL_START_ORN = p.getQuaternionFromEuler([0, 0, 0])
CUE_TRIGGER_INTERVAL_SEC = 3 # How often to simulate a new cue

# Import Hopfield Network
try:
    from models.hopfield_pytorch import HopfieldNetworkPyTorch as HopfieldNetwork
    print("Using PyTorch implementation.")
except ImportError:
    print(f"Error: Could not import HopfieldNetworkPyTorch.")
    print(f"Attempted to import from: {os.path.join(project_root, 'models', 'hopfield_pytorch.py')}")
    sys.exit(1)

# Check for CUDA  
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# Helper Functions (Hopfield related) 

def load_patterns(filepath: str) -> Dict[str, List[int]]:
    """Loads patterns from a JSON file."""
    print(f"Attempting to load patterns from: {filepath}")
    try:
        with open(filepath, 'r') as f:
            patterns = json.load(f)
        print(f"Successfully loaded patterns from {filepath}")
        return patterns
    except FileNotFoundError:
        print(f"Error: Patterns file not found at {filepath}")
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
    print("\n--- Training Network ---")
    network.train(clean_patterns)
    return network

def simulate_input_cue(all_patterns: Dict[str, List[int]]) -> Tuple[str, List[int]]:
    """Simulates receiving an input cue (selects a random noisy pattern)."""
    noisy_keys = [k for k in all_patterns if k.endswith("_noisy")]
    if not noisy_keys:
        raise ValueError("No noisy patterns found in the loaded data to use as cues.")
    selected_key = random.choice(noisy_keys)
    cue_pattern = all_patterns[selected_key]
    print(f"\n Simulating Input Cue")
    print(f"Selected cue: '{selected_key}'")
    return selected_key, cue_pattern

def recognize_behavior(network: HopfieldNetwork, cue_pattern: List[int]) -> str:
    """Uses the Hopfield network to recognize the behavior from the cue."""
    print("\n--- Attempting Recognition ---")
    input_pattern_tensor = torch.tensor(cue_pattern, dtype=torch.float32, device=DEVICE)
    final_state, energy_trace, converged = network.recall(
        input_pattern_tensor,
        max_steps=MAX_RECALL_STEPS,
        update_rule=RECALL_UPDATE_RULE,
        verbose=False 
    )
    if not converged: print("Warning: Recall did not converge.")
    identified_pattern_name = network.identify_recalled_pattern(final_state)
    print(f"Recall process converged in {len(energy_trace)-1} steps.")
    print(f"Identified pattern name: '{identified_pattern_name}'")
    return identified_pattern_name

# TODO: PyBullet Action Placeholders 

def set_velocity(drone_id, linear: Tuple[float, float, float], angular: Tuple[float, float, float] = (0, 0, 0)):
    """Sets the base linear and angular velocity of the drone."""
    p.resetBaseVelocity(drone_id, linearVelocity=linear, angularVelocity=angular)

def glide_action(drone_id):
    """Move forward gently while descending slightly."""
    set_velocity(drone_id, linear=(0.2, 0, 0.3))

def flap_action(drone_id):
    """Flap: simulate sharp vertical lift with minor oscillation."""
    t = time.time()
    vertical_lift = 0.5 + 0.4 * (random.random() - 0.5) 
    set_velocity(drone_id, linear=(0, 0, vertical_lift))

def hover_action(drone_id):
    """Maintain position with gentle upward correction."""
    set_velocity(drone_id, linear=(0, 0, 0.3))  # counteract gravity gently

def turn_action(drone_id):
    """Rotate in place while maintaining altitude."""
    set_velocity(drone_id, linear=(0, 0, 0.2), angular=(0, 0, 2.0))

def unknown_action(drone_id):
    """Failsafe: try to stabilize if the behavior is unrecognized."""
    set_velocity(drone_id, linear=(0, 0, 0))

ACTION_MAP = {
    "glide": glide_action,
    "flap": flap_action,
    "hover": hover_action,
    "turn": turn_action,
    "Unknown": unknown_action
}

def execute_behavior(recognized_behavior: str, drone_id: int):
    """Calls the appropriate PyBullet action function (placeholders)."""
    action_function = ACTION_MAP.get(recognized_behavior, unknown_action)
    action_function(drone_id) # Call the placeholder function

def log_drone_state(drone_id, timestamp):
    """Logs the current state of the drone."""
    pos, orn = p.getBasePositionAndOrientation(drone_id)
    lin_vel, ang_vel = p.getBaseVelocity(drone_id)
    trajectory_log.append({
        "timestamp": timestamp,
        "position": pos,
        "orientation": orn,
        "linear_velocity": lin_vel,
        "angular_velocity": ang_vel
    })

# Main Simulation Setup and Loop 

if __name__ == "__main__":
    # 1. Load patterns
    all_patterns = load_patterns(PATTERNS_FILE)
    clean_patterns = {k: v for k, v in all_patterns.items() if not k.endswith("_noisy")}

    # 2. Initialize and train the Hopfield network
    try:
        hopfield_memory = initialize_and_train_network(clean_patterns)
    except ValueError as e:
        print(f"Error initializing network: {e}")
        sys.exit(1)

    # 3. Initialize PyBullet
    print("Initializing PyBullet in GUI mode...")
    physicsClient = p.connect(p.GUI) # Switched back to GUI mode
    # physicsClient = p.connect(p.DIRECT)
    print("Setting search path (for plane.urdf)...")
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    print("Setting physics parameters...")
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(SIMULATION_TIME_STEP)

    # Load ground plane (from pybullet_data)
    print("Loading ground plane...")
    try:
        planeId = p.loadURDF("plane.urdf")
        print(f"Ground plane loaded with ID: {planeId}")
    except p.error as e:
        print(f"Error loading plane.urdf: {e}")
        p.disconnect()
        sys.exit(1)


    # Load model using the *local* full path
    print(f"Attempting to load model model from local path: {MODEL_URDF_FULL_PATH}")
    if not os.path.exists(MODEL_URDF_FULL_PATH):
         print(f"\n--- Error: Local URDF file not found! ---")
         print(f"Checked path: {MODEL_URDF_FULL_PATH}")
         p.disconnect()
         sys.exit(1)

    try:
        # Load the URDF. Ensure mesh files (like .obj) are in the same directory (urdfs/). 
        #If not, manually add them https://github.com/bulletphysics/bullet3/tree/master/data/Quadrotor
        modelId = p.loadURDF(MODEL_URDF_FULL_PATH, MODEL_START_POS, MODEL_START_ORN)
        print(f"Model '{TEST_URDF_FILENAME}' loaded with ID: {modelId}")
    except p.error as e:
        print(f"\n--- Error loading URDF: {TEST_URDF_FILENAME} ---")
        print(f"PyBullet error: {e}")
        p.disconnect()
        sys.exit(1)

    # 4. Simulation Loop
    start_time = time.time()
    last_cue_time = start_time - CUE_TRIGGER_INTERVAL_SEC # Trigger first cue immediately
    current_behavior = "Unknown" # Track the last recognized behavior

    print("\n Starting Simulation Loop (GUI mode) ")
    step_count = 0
    try:
        while (time.time() - start_time) < SIMULATION_DURATION_SEC:
            current_time = time.time()

            # Hopfield Decision Logic (Trigger periodically) 
            if current_time - last_cue_time >= CUE_TRIGGER_INTERVAL_SEC:
                last_cue_time = current_time
                try:
                    cue_name, cue_pattern = simulate_input_cue(all_patterns)
                    current_behavior = recognize_behavior(hopfield_memory, cue_pattern)
                    # Print the action only when a new decision is made
                    print(f"Decision: Applying '{current_behavior}' action.")
                except ValueError as e:
                    print(f"Error simulating cue: {e}")
                    current_behavior = "Unknown" # Default if error
                    print(f"Decision: Applying '{current_behavior}' action.")

            #  Execute Behavior in PyBullet (Placeholders) 
            # empty for now
            execute_behavior(current_behavior, modelId)

            # --- Step Simulation 
            p.stepSimulation()
            log_drone_state(modelId, time.time() - start_time)

            step_count += 1

            # Print progress periodically (less frequent in DIRECT mode)
            # if step_count % (240 * 5) == 0: # Print every 5 seconds
            #      print(f"Simulation step {step_count}...")

            # Add sleep back for GUI mode to prevent unresponsiveness
            time.sleep(SIMULATION_TIME_STEP)


    except Exception as e:
        # Catch potential exceptions during the loop
        print(f"\n Exception during simulation loop ")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc() 

    finally:
        # 5. Cleanup
        print(f"\n Simulation Finished (Ran {step_count} steps)")
        print("Disconnecting PyBullet...")
        try:
            p.disconnect()
            print("PyBullet disconnected successfully.")
        except Exception as e:
            print(f"Exception during PyBullet disconnect: {e}")

        output_path = os.path.join(project_root, "data", "trajectory_log.json")
        try:
            with open(output_path, 'w') as f:
                json.dump(trajectory_log, f, indent=2)
            print(f"Trajectory log saved to: {output_path}")
        except Exception as e:
            print(f"Failed to save trajectory log: {e}")
