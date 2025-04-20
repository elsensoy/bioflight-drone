bioflight-drone/
│
├── README.md
├── requirements.txt
├── main.py                     # Entry point: where training/testing is controlled
│
├── config/
│   └── config.yaml             # Parameters for model, training, energy weights
│
├── data/
│   ├── test_inputs.json        # Test cases for Hopfield net
│   └── bird_behavior_dataset/  # Folder for future behavior sequences
│
├── models/
│   ├── hopfield.py             # Hopfield memory network (lucidrains version or custom)
│   ├── energy_model.py         # EBM + CNN (like your example above)
│   └── riemannian_net.py       # Optional: PyG + cusp model
│
├── agents/
│   └── drone_agent.py          # The drone behavior selector using energy + memory
│
├── simulation/
│   ├── airsim_interface.py     # Interface for AirSim control (later)
│   ├── pybullet_wrapper.py     # Gym-PyBullet-Drone simulation controller
│   └── mock_env.py             # Minimal mocked environment for local testing
│
├── training/
│   ├── train_energy_model.py   # Training loop for CNN + EBM
│   ├── train_memory.py         # Hopfield training (you already did!)
│   └── evaluate.py             # Run predictions + energy metrics
│
├── utils/
│   ├── data_loader.py          # Loads json, preprocesses patterns
│   └── visualizer.py           # Plots energy, convergence, trajectories
│
└── notebooks/
    └── experiment_logs.ipynb   # Jupyter space for trying ideas


CUSP (Spectro-Riemannian Graph Neural Networks) is a neural network model that:

Learns on graphs

Uses Riemannian manifolds to represent complex relationships (like curvature of space, or non-Euclidean behavior spaces)

Includes graph curvature and geometric structure to improve generalization

It’s designed for tasks like:

Node classification

Graph classification

Learning embeddings on complex manifolds (like 
𝐻
𝑛
H 
n
 , 
𝑆
𝑛
S 
n
 , etc.)

We are interested in:
Modeling drone behavior as a dynamic system

Capturing evolutionary, shape-aware, low-energy motion trajectories

Comparing learned vs. true biological paths (e.g. bird vs. drone)

