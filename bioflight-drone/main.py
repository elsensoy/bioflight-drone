# main.py (or rename to run_simulation.py)
import gym
import numpy as np
import time
import yaml # For config loading

# Import simulation environment (ensure it's installed)
try:
    # Using TakeoffAviary as an example environment
    from gym_pybullet_drones.envs.single_agent_rl import TakeoffAviary
    from gym_pybullet_drones.utils.enums import DroneModel, Physics
except ImportError as e:
    print("*"*80)
    print(f"Error importing gym_pybullet_drones: {e}")
    print("Please ensure 'gym-pybullet-drones' is installed correctly.")
    print("Try: pip install -e external/gym-pybullet-drones")
    print("*"*80)
    exit()

# Import agent
from agents.drone_agent import DroneAgent

# --- Configuration Loading ---
def load_config(config_path="config/config.yaml"):
    """Loads configuration from a YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        # Return a default config or exit
        return None # Or raise error
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return None # Or raise error

# --- Main Simulation Runner ---
def main():
    # Load Config
    config = load_config()
    if config is None:
        print("Exiting due to configuration error.")
        return

    sim_config = config.get('simulation', {})
    agent_config = config.get('agent', {})
    agent_config['patterns_path'] = config.get('patterns_path', 'data/behavior_patterns.json') # Add pattern path to config

    # Initialize Simulation Environment
    print("Initializing simulation environment...")
    try:
        # Map string config values to Enums if necessary
        drone_model_enum = DroneModel(sim_config.get('drone_model', 'cf2x'))
        physics_enum = Physics(sim_config.get('physics', 'pyb'))

        env = TakeoffAviary(
            drone_model=drone_model_enum,
            initial_xyzs=np.array([sim_config.get('initial_pos', [0, 0, 0.1])]),
            physics=physics_enum,
            freq=sim_config.get('sim_freq', 240),
            aggregate_phy_steps=sim_config.get('ctrl_freq_divisor', 5), # ctrl_freq = sim_freq / divisor
            gui=sim_config.get('gui', True),
            record=sim_config.get('record', False)
        )
        print(f"Control Frequency: {env.CTRL_FREQ} Hz")
    except Exception as e:
        print(f"Error initializing environment: {e}")
        return

    # Initialize Agent
    print("Initializing drone agent...")
    agent = DroneAgent(agent_config)

    # Simulation Loop
    print("Starting simulation loop...")
    obs = env.reset()
    start_time = time.time()
    total_reward = 0
    step = 0
    done = False

    try:
        while not done:
            # 1. Agent chooses action based on observation
            action = agent.choose_action(obs)

            # 2. Step the environment with the agent's action
            obs, reward, done, info = env.step(action)
            total_reward += reward

            # 3. Render (optional) & Logging
            if sim_config.get('render', True): # Add render option to config
                 env.render()

            print(f"Step: {step}, Reward: {reward:.3f}, Total Reward: {total_reward:.3f}, Done: {done}")
            step += 1

            # Sync to simulation frequency if GUI is enabled
            if sim_config.get('gui', True):
                sync_freq = env.CTRL_FREQ # Sync to control frequency
                time.sleep(max(0, 1./sync_freq - (time.time() - start_time) % (1./sync_freq) ))

            # Optional: Check max duration
            duration_sec = sim_config.get('duration_sec', None)
            if duration_sec and (time.time() - start_time > duration_sec):
                print("Max duration reached.")
                done = True

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    finally:
        # Clean up
        env.close()
        print("-" * 50)
        print(f"Simulation Finished. Total steps: {step}, Total reward: {total_reward:.3f}")
        print("-" * 50)


if __name__ == "__main__":
    main()