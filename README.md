

# BioFlight-Drone

This is an in-progress bio-inspired machine learning project that mimics the energy-efficient flight behavior of birds using **Energy-Based Models (EBMs)**, **Hopfield Networks**, **Riemannian Geometry**, and **Model Predictive Control (MPC)**.

The system enables drones to learn, remember, and execute efficient behaviors by minimizing a composite energy landscape, bridging the gap between biological "muscle memory" and rigorous physics-informed control.

-----

##  Core Concept: The "Thinking" Drone

Unlike standard drones that blindly follow GPS waypoints, **BioFlight** uses a three-layer "brain":

1.  **The Memory (Hopfield):** Stores "ideal" flight archetypes (glide, hover, bank) as stable energy attractors.
2.  **The Geometry (Riemannian):** Understands the "shape" of flight on curved manifolds rather than flat 3D space.
3.  **The Executive (MPC + CBF):** A pro-active controller that looks into the future, minimizing energy while guaranteeing safety via Control Barrier Functions.

-----

##  Project Goals

  - **Bio-Mimicry:** Encode avian flight patterns into high-dimensional associative memory.
  - **Energy Optimization:** Minimize battery drain by "sliding" down learned energy gradients.
  - **Geometric Intelligence:** Use product manifolds ($\mathbb{H} \times \mathbb{S} \times \mathbb{R}$) to represent trajectories.
  - **Provable Safety:** Integrate Lyapunov Stability and Control Barrier Functions (CBFs) to prevent crashes.

-----

## Mathematical Foundation

### 1\. Behavioral Memory (Hopfield Attractors)

We model distinct flight maneuvers as stable attractor states in an energy landscape:
$$E(x) = -\frac{1}{2} \sum_{i,j} w_{ij} x_i x_j$$
When the drone is "confused" by wind or noise, the network dynamics pull the state back to the nearest stored biological pattern:
$$x_i(t+1) = \text{sign}\left( \sum_j w_{ij} x_j(t) \right)$$

### 2\. Predictive Control (MPC)

To move from "thought" to "action," the agent solves a real-time optimization problem over a finite horizon $T$. It treats the drone as a **Double Integrator** system ($\ddot{p} = u$):

$$\min_{u} \sum_{k=0}^{T} \underbrace{E_{\text{total}}(x_k, u_k)}_{\text{Energy Cost}} + \underbrace{\text{dist}_{\mathcal{M}}(x_k, x_{ref})}_{\text{Riemannian Distance}}$$

### 3\. Stability & Safety (Lyapunov + CBF)

To ensure the drone doesn't just "learn" to crash efficiently, we enforce:

1.  **Lyapunov Stability:** $\dot{V}(x) < 0$, ensuring convergence to the goal.
2.  **Control Barrier Functions (CBF):** $h(x) \geq 0$, a mathematical "force field" that overrides the AI to prevent collisions with obstacles.

-----

##  System Architecture

| Module | Purpose | Tech |
| :--- | :--- | :--- |
| `core/robot.py` | Physical "Body" (Double Integrator physics) | NumPy |
| `agents/bio_agent.py` | The "Brain" (High-level decision making) | Python |
| `agents/mpc_controller.py` | The "Executive" (Trajectory optimization) | CasADi / IPOPT |
| `models/hopfield.py` | "Muscle Memory" (Pattern recall) | PyTorch / NumPy |
| `models/riemannian_net.py` | "Shape Perception" (Trajectory Manifolds) | CUSP / PyTorch |

-----

##  Quick Start

### 1\. Environment Setup

```bash
# Create environment
conda create -n bioflight python=3.10 -y
conda activate bioflight

# Install core dependencies
pip install casadi numpy matplotlib torch pybullet
```

### 2\. Initialize Geometry Engine (CUSP)

```bash
git submodule update --init --recursive
cd external/cusp && pip install -r requirements.txt && cd ../..
```

### 3\. Run the Simulation

Launch the MPC-guided drone simulation:

```bash
python main_sim.py
```

-----

## Roadmap & Experiments

  - [x] **Double Integrator Baseline:** Stabilizing a point-mass drone using EBMs.
  - [ ] **Riemannian Trajectory Tracking:** Using CUSP to follow geodesics on a sphere.
  - [ ] **Obstacle Avoidance:** Implementing CBFs within the CasADi solver loop.
  - [ ] **Hardware-in-the-Loop:** Exporting learned energy weights to real drone firmware (Beta).

-----

##  Citation & Credits

**Project Author:** Elida Sensoy ([elsensoy@umich.edu](mailto:elsensoy@umich.edu))

If you use this work, please cite:

```bibtex
@misc{Sensoy_bioflight_2025,
  title={BioFlight-Drone: Bio-Inspired Energy-Efficient Drone Control},
  author={Elida Sensoy},
  year={2025},
  publisher={GitHub},
  note={In Development}
}
```

 