# models/hopfield_pytorch.py
import torch
import numpy as np # Still useful for numpy arrays in dicts initially if needed
from typing import List, Dict, Tuple, Optional, Union

class HopfieldNetworkPyTorch:
    """
    Implements a discrete Hopfield network using PyTorch tensors.
    (Other methods like __init__, train, energy, identify_recalled_pattern, load_patterns_from_dict remain the same)
    """
    def __init__(self, size: int, device: Optional[Union[str, torch.device]] = None):
        """
        Initializes the Hopfield network using PyTorch.

        Args:
            size (int): The number of neurons (dimension of the patterns).
            device (Optional[Union[str, torch.device]]): The device to run on ('cpu', 'cuda', etc.).
                                                        Defaults to 'cpu'.
        """
        if not isinstance(size, int) or size <= 0:
            raise ValueError("Network size must be a positive integer.")
        self.size: int = size

        # Determine the device
        if device is None:
            self.device = torch.device('cpu')
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        # Initialize weight tensor with zeros on the specified device
        self.W: torch.Tensor = torch.zeros((size, size), dtype=torch.float32, device=self.device)
        # Dictionary to store learned patterns {name: pattern_tensor}
        self.patterns: Dict[str, torch.Tensor] = {}
        print(f"Initialized HopfieldNetworkPyTorch on device: {self.device}")

    def train(self, patterns_dict: Dict[str, List[int]]):
        """
        Trains the network using the Hebbian learning rule on a dictionary of patterns.

        Assumes patterns are represented using bipolar values (-1, 1).

        Args:
            patterns_dict (Dict[str, List[int]]): A dictionary where keys are pattern
                                                  names and values are the corresponding
                                                  bipolar pattern vectors (lists of -1s and 1s).

        Raises:
            ValueError: If patterns are not provided, sizes mismatch, or values are not bipolar.
        """
        if not patterns_dict:
            print("Warning: No patterns provided for training.")
            return

        self.patterns = {} # Reset stored patterns
        self.W = torch.zeros((self.size, self.size), dtype=torch.float32, device=self.device) # Reset weights

        num_patterns = len(patterns_dict)
        if num_patterns == 0:
             print("Warning: Empty patterns dictionary provided.")
             return

        # Hebbian learning: W = sum over patterns p (p * p^T)
        for name, pattern_list in patterns_dict.items():
            # Convert list to a PyTorch tensor and move to the correct device
            pattern = torch.tensor(pattern_list, dtype=torch.float32, device=self.device)

            if pattern.shape != (self.size,):
                raise ValueError(
                    f"Pattern '{name}' size {pattern.shape[0]} does not match "
                    f"network size {self.size}"
                )
            # Ensure pattern is bipolar (-1, 1)
            if not torch.all(torch.isin(pattern, torch.tensor([-1.0, 1.0], device=self.device))):
                 raise ValueError(
                    f"Pattern '{name}' must contain only bipolar values (-1, 1)."
                 )

            # Add contribution using torch.outer
            self.W += torch.outer(pattern, pattern)
            self.patterns[name] = pattern # Store the pattern tensor

        # Ensure no self-connections (zero diagonal)
        # Use torch.diag_embed(torch.diag(self.W)) to create a diagonal matrix
        # and subtract it, or simply fill the diagonal.
        self.W.fill_diagonal_(0)

        print(f"Training complete. Stored {len(self.patterns)} patterns.")

    def energy(self, state: torch.Tensor) -> float:
        """
        Calculates the Hopfield energy of a given state tensor.

        Energy E(x) = -0.5 * x^T * W * x

        Args:
            state (torch.Tensor): A bipolar state tensor (-1, 1) of size `self.size`
                                  on the network's device.

        Returns:
            float: The energy of the state.

        Raises:
            ValueError: If the state tensor size or device does not match.
        """
        if state.shape != (self.size,):
            raise ValueError(
                f"State tensor size {state.shape[0]} does not match network size {self.size}"
            )
        if state.device != self.device:
             raise ValueError(f"State tensor is on device {state.device}, but network is on {self.device}")

        # Energy calculation using tensor matrix multiplication
        # state @ W @ state assumes state is treated as 1D row vector then column vector
        energy_val = -0.5 * torch.dot(state, torch.matmul(self.W, state))
        return energy_val.item() # Return scalar Python float

    def recall(self, input_pattern: Union[List[int], np.ndarray, torch.Tensor],
               max_steps: int = 100,
               update_rule: str = 'async',
               verbose: bool = False) -> Tuple[torch.Tensor, List[float], bool]:
        """
        Recalls a stored pattern starting from an initial state using PyTorch.

        Args:
            input_pattern (Union[List[int], np.ndarray, torch.Tensor]): Initial state vector (-1, 1).
            max_steps (int): Maximum update iterations.
            update_rule (str): 'async' or 'sync'.
            verbose (bool): Print detailed updates if True.

        Returns:
            Tuple[torch.Tensor, List[float], bool]: Final state tensor, energy trace, convergence status.

        Raises:
            ValueError: For size/device mismatch or invalid update rule.
        """
        # Convert input to a tensor on the correct device
        if isinstance(input_pattern, torch.Tensor):
            state = input_pattern.to(dtype=torch.float32, device=self.device)
        else: # Handles list or numpy array
            state = torch.tensor(input_pattern, dtype=torch.float32, device=self.device)

        if state.shape != (self.size,):
            raise ValueError(
                f"Input pattern size {state.shape[0]} does not match network size {self.size}"
            )
        # Clamp values to -1 or 1 if necessary
        if not torch.all(torch.isin(state, torch.tensor([-1.0, 1.0], device=self.device))):
            print("Warning: Input pattern contains values other than -1 or 1. Clamping...")
            # Use torch.sign which maps 0 to 0, then correct 0s to 1s (or -1s if preferred)
            state = torch.sign(state)
            state[state == 0] = 1.0 # Map 0 to 1

        energy_trace = [self.energy(state)]
        converged = False

        if verbose:
            print(f"Recall initiated. Start Energy: {energy_trace[0]:.2f}")
            print(f"Initial State: {state.cpu().numpy().astype(int)}") # Convert for printing
        else:
             print(f"Recall initiated (max_steps={max_steps})...") # Indicate start

        for step in range(max_steps):
            prev_state = state.clone() # Important: clone the tensor

            if update_rule == 'async':
                # Asynchronous update: Iterate through neurons in random order
                indices = torch.randperm(self.size, device=self.device)
                for i in indices:
                    # activation = W[i, :] @ state (dot product of i-th row of W and state)
                    activation = torch.dot(self.W[i, :], state)
                    # Update state[i] based on sign
                    state[i] = 1.0 if activation >= 0 else -1.0
            elif update_rule == 'sync':
                # Synchronous update: Calculate all activations based on prev_state
                # activations = W @ state (matrix-vector product)
                activations = torch.matmul(self.W, state)
                # Update all neurons simultaneously using torch.where
                state = torch.where(activations >= 0,
                                    torch.tensor(1.0, device=self.device),
                                    torch.tensor(-1.0, device=self.device))
            else:
                raise ValueError(f"Unknown update rule: '{update_rule}'. Choose 'async' or 'sync'.")

            current_energy = self.energy(state)
            energy_trace.append(current_energy)

            # --- Debug Print Added ---
            # Print progress every 10 steps if not verbose
            if not verbose and (step + 1) % 10 == 0:
                print(f"  Recall step {step + 1}/{max_steps} reached...")
            # --- End Debug Print ---

            if verbose:
                 print(f"Step {step + 1}: Energy={current_energy:.2f}, State={state.cpu().numpy().astype(int)}")

            # Check for convergence: state hasn't changed
            if torch.equal(state, prev_state):
                 converged = True
                 if verbose:
                     print(f"\nConverged: State stable after {step + 1} steps.")
                 else:
                     print(f"Converged after {step + 1} steps.") # Print convergence even if not verbose
                 break

        if not converged:
            # This message will print if the loop finishes without convergence
            print(f"\nWarning: Did not converge within {max_steps} steps.")

        return state, energy_trace, converged

    def identify_recalled_pattern(self, final_state: torch.Tensor) -> str:
        """
        Compares the final state tensor to the stored prototype patterns.

        Args:
            final_state (torch.Tensor): The state tensor returned by recall (on network's device).

        Returns:
            str: The name of the matching stored pattern, or 'Unknown'.
        """
        if final_state.device != self.device:
             print(f"Warning: Final state is on {final_state.device}, comparing with patterns on {self.device}. Moving state.")
             final_state = final_state.to(self.device)

        if not self.patterns:
             print("Warning: No patterns stored in the network to identify against.")
             return "Unknown (No patterns stored)"

        for name, pattern_tensor in self.patterns.items():
            if torch.equal(final_state, pattern_tensor):
                return name
        return "Unknown"

    def load_patterns_from_dict(self, patterns_dict: Dict[str, List[int]]):
         """
         Utility to load pattern vectors into `self.patterns` as tensors
         without retraining. Assumes bipolar (-1, 1).

         Args:
            patterns_dict (Dict[str, List[int]]): Dictionary of pattern names and vectors.
         """
         self.patterns = {}
         for name, pattern_list in patterns_dict.items():
             pattern = torch.tensor(pattern_list, dtype=torch.float32, device=self.device)
             if pattern.shape != (self.size,):
                 print(f"Warning: Pattern '{name}' size mismatch. Skipping loading.")
                 continue
             if not torch.all(torch.isin(pattern, torch.tensor([-1.0, 1.0], device=self.device))):
                  print(f"Warning: Pattern '{name}' not bipolar. Skipping loading.")
                  continue
             self.patterns[name] = pattern
         print(f"Loaded {len(self.patterns)} patterns into memory for identification.")


# --- Example Usage (PyTorch version) ---keep or comment it
if __name__ == '__main__':
    # Check if CUDA is available, otherwise use CPU
    selected_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Define patterns (as lists)
    pattern_size = 6
    patterns_to_store = {
        'glide': [-1, 1, -1, 1, -1, 1],
        'flap':  [ 1, 1, -1, -1, 1, 1],
        'hover': [-1, -1, 1, 1, -1, -1]
    }

    # Create and train the network on the selected device
    network = HopfieldNetworkPyTorch(size=pattern_size, device=selected_device)
    network.train(patterns_to_store)

    print("\nNetwork Weights (W) on device:", network.W.device)
    # print(network.W.cpu().numpy()) # Print weights (move to CPU if needed)

    # --- Test Recall ---
    print("\n--- Recall Tests (PyTorch) ---")

    # Test 1: Recall 'glide' (should converge immediately)
    print("\nTest 1: Recall 'glide'")
    test_pattern_glide = torch.tensor(patterns_to_store['glide'], dtype=torch.float32, device=selected_device)
    final_state_glide, energy_glide, converged_glide = network.recall(test_pattern_glide, max_steps=10, verbose=True)
    identified_glide = network.identify_recalled_pattern(final_state_glide)
    print(f"Result: Converged={converged_glide}, Identified='{identified_glide}'")
    print(f"Energy Trace: {energy_glide}")

    # Test 2: Recall noisy 'flap' (async)
    print("\nTest 2: Recall noisy 'flap' (async)")
    noisy_flap_list = patterns_to_store['flap'].copy()
    noisy_flap_list[0] = -1
    noisy_flap_list[3] = 1
    noisy_flap_tensor = torch.tensor(noisy_flap_list, dtype=torch.float32, device=selected_device)
    print(f"Noisy Input: {noisy_flap_tensor.cpu().numpy().astype(int)}")
    final_state_flap, energy_flap, converged_flap = network.recall(noisy_flap_tensor, max_steps=20, update_rule='async', verbose=True)
    identified_flap = network.identify_recalled_pattern(final_state_flap)
    print(f"Result: Converged={converged_flap}, Identified='{identified_flap}'")
    print(f"Energy Trace: {energy_flap}")
    if identified_flap == 'flap': print("Recall successful!")
    else: print("Recall failed or converged to a different state.")

    # Test 3: Recall noisy 'hover' (sync)
    print("\nTest 3: Recall noisy 'hover' (sync)")
    noisy_hover_list = patterns_to_store['hover'].copy()
    noisy_hover_list[1] = 1
    noisy_hover_list[4] = 1
    noisy_hover_tensor = torch.tensor(noisy_hover_list, dtype=torch.float32, device=selected_device)
    print(f"Noisy Input: {noisy_hover_tensor.cpu().numpy().astype(int)}")
    final_state_hover, energy_hover, converged_hover = network.recall(noisy_hover_tensor, max_steps=20, update_rule='sync', verbose=True)
    identified_hover = network.identify_recalled_pattern(final_state_hover)
    print(f"Result: Converged={converged_hover}, Identified='{identified_hover}'")
    print(f"Energy Trace: {energy_hover}")
    if identified_hover == 'hover': print("Recall successful!")
    else: print("Recall failed or converged to a different state.")

    # Test 4: Energy calculation
    print("\nTest 4: Energy Calculation")
    energy_val = network.energy(torch.tensor(patterns_to_store['glide'], dtype=torch.float32, device=selected_device))
    print(f"Energy of 'glide' pattern: {energy_val:.2f}")

