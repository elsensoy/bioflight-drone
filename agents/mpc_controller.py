import casadi as ca
import numpy as np

class BioFlightMPC:
    def __init__(self, horizon=10, dt=0.1):
        self.N = horizon # How many steps to look ahead
        self.dt = dt     # Time per step
        
        # 1. Define States (Position x, y, z and Velocity vx, vy, vz)
        self.x = ca.MX.sym('x', 6) 
        # 2. Define Controls (Thrust vector fx, fy, fz)
        self.u = ca.MX.sym('u', 3)
        
        # 3. Drone Dynamics (F = ma)
        mass = 0.5 # kg
        g = 9.81   # Gravity

        # Using the standard physics-based model
        self.x_dot = ca.vertcat(
            self.x[3:], 
            (self.u[0] / mass),
            (self.u[1] / mass),
            (self.u[2] / mass) - g
        )
        
        # Create a CasADi function for the dynamics
        self.f = ca.Function('f', [self.x, self.u], [self.x_dot])

    def solve(self, current_state, reference_path, weights):
        opti = ca.Opti()
        
        # Variables to optimize
        X = opti.variable(6, self.N + 1)
        U = opti.variable(3, self.N)
        
        # Initial State Constraint
        opti.subject_to(X[:, 0] == current_state)
        
        total_cost = 0
        for k in range(self.N):
            #  OBSTACLE AVOIDANCE (CBF-lite)  
            # Corrected indentation here
            obstacle_pos = ca.vertcat(2.0, 2.0, 1.0)
            dist_obs_sq = ca.sumsqr(X[:3, k] - obstacle_pos)
            
            # Safety Constraint: h(x) >= 0
            margin = 0.5**2
            opti.subject_to(dist_obs_sq >= margin)
                
            # COST FUNCTION 
            # E_behavior: Riemannian distance to the reference path
            dist_ref_sq = ca.sumsqr(X[:3, k] - reference_path[k])
            
            # E_hardware: Cost of "Muscle" effort (thrust)
            effort = ca.sumsqr(U[:, k])
            
            # E_total calculation
            step_cost = (weights['alpha'] * dist_ref_sq) + (weights['beta'] * effort)
            total_cost += step_cost
            
            # Dynamics Constraint: x(k+1) = x(k) + f(x, u) * dt
            next_state = X[:, k] + self.f(X[:, k], U[:, k]) * self.dt
            opti.subject_to(X[:, k+1] == next_state)

        opti.minimize(total_cost)
        
        # Solver setup
        opti.solver('ipopt', {'print_time': 0}, {'tol': 1e-3, 'print_level': 0})
        
        try:
            sol = opti.solve()
            return sol.value(U[:, 0])
        except:
            # If the solver fails (usually because the obstacle is unavoidable), hover
            return np.array([0, 0, 0.5 * 9.81])