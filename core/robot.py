import numpy as np

class DroneRobot:
    def __init__(self, mass=0.5, dt=0.1):
        self.mass = mass
        self.dt = dt
        # State: [x, y, z, vx, vy, vz]
        self.state = np.zeros(6)
        self.gravity = np.array([0, 0, -9.81])

    def step(self, force):
        """Standard Double Integrator physics: F = ma"""
        pos = self.state[:3]
        vel = self.state[3:]
        
        # Calculate acceleration (a = F/m + g)
        accel = (force / self.mass) + self.gravity
        
        # Update velocity and position (Euler integration)
        new_vel = vel + accel * self.dt
        new_pos = pos + new_vel * self.dt
        
        self.state = np.concatenate([new_pos, new_vel])
        return self.state

    def get_observation(self):
        return self.state