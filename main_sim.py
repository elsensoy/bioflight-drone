import numpy as np
import matplotlib.pyplot as plt
from core.robot import DroneRobot
from agents.bio_agent import BioFlightAgent

def run_simulation():
    # 1. Setup
    config = {'patterns_path': 'data/behavior_patterns.json'}
    robot = DroneRobot(mass=0.5, dt=0.1)
    agent = BioFlightAgent(config)
    
    history = []
    current_state = robot.get_observation()

    print(" Starting BioFlight MPC Simulation...")
    
    # 2. Simulation Loop (100 steps)
    for t in range(100):
        
        # A. Agent decides on Force based on Hopfield Memory + MPC
        force = agent.compute_action(current_state)
        
        # B. Physics engine applies the force
        current_state = robot.step(force)
        
        # C. Record for visualization
        history.append(current_state.copy())

    # 3. Visualization
    history = np.array(history)
    plt.figure(figsize=(10, 5))
    plt.plot(history[:, 0], history[:, 1], label='MPC Trajectory', color='blue')
    plt.scatter(0, 0, color='green', label='Start')
    plt.title("Drone Trajectory: Double Integrator with Bio-MPC")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    run_simulation()