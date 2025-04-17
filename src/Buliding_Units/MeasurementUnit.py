from abc import ABC, abstractmethod
import os
import torch
import copy
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.pyplot as plt


class MeasurementUnit(ABC):
    """
    Abstract class for modules used in a training frame that measure some property of the models.
    
    The `measure` method should be implemented by subclasses to compute a numerical metric
    given a list of models. Additionally, this class initializes a reporter that logs measurement
    values to a file located in the 'measurements' folder.
    """
    
    def __init__(self, measure_name):
        # Ensure that the "measurements" folder exists
        measurements_folder = "measurements"
        os.makedirs(measurements_folder, exist_ok=True)
        
        # Set the file path for the log file using measure_name
        file_name = f"{measure_name.replace(' ', '_').lower()}_log.txt"
        self.file_path = os.path.join(measurements_folder, file_name)
        
        # Initialize the reporter by opening the file in append mode
        self.reporter = open(self.file_path, "a")
        
        # Write initial information to the log file
        pid = os.getpid()
        self.reporter.write(f"Process ID: {pid}\n")
        self.reporter.write(f"Measurement Description: {measure_name}\n")
        self.reporter.write("\n")
        self.reporter.flush()

    @abstractmethod
    def measure(self, frame) -> float:
        """
        Compute and return a measurement given a list of model instances.
        
        Args:
            models (list): A list of model objects.
        
        Returns:
            float: The measurement value.
        """
        pass
    
    def log_measurement(self, measurement: float):
        """
        Log the measured value to the reporter file.
        
        Args:
            measurement (float): The measurement to be logged.
        """
        self.reporter.write(f"{measurement}\n")
        self.reporter.flush()
    
    def close(self):
        """
        Close the reporter file handle if it is open.
        """
        if not self.reporter.closed:
            self.reporter.close()
    
    def __del__(self):
        self.close()



class Working_memory_usage(MeasurementUnit):
    """
    Measurement unit that computes the memory usage of the model and optimizer.
    It calculates the total memory used by the model parameters and optimizer state,
    and returns the cumulative memory usage in megabytes (MB).
    """
    
    def __init__(self):
        super().__init__("Memory Usage Measurement")
    
    def measure(self, frame) -> float:
        """
        Compute the memory usage for each client.
        
        For distributed frames, it computes the memory usage for each client (i.e., each
        entry in 'distributed_models') by summing the memory used by its model and optimizer.
        For non-distributed frames, it computes the memory usage for the central model (frame.model)
        and for each optimizer in frame.optimizers.
        
        Returns:
            float: Total memory usage in megabytes (MB).
        """
        total_memory_bytes = 0
        
        # Check if the frame uses distributed models (i.e., multiple clients)
        if hasattr(frame, "distributed_models") and frame.distributed_models:
            for client, (model, optimizer) in frame.distributed_models.items():
                total_memory_bytes += self._get_model_memory(model)
                total_memory_bytes += self._get_optimizer_memory(optimizer)

            total_memory_bytes = total_memory_bytes/len(frame.distributed_models)

        else:
            # Non-distributed case: use frame.model (if available) and
            # iterate through the list of optimizers.
            if hasattr(frame, "model"):
                total_memory_bytes += self._get_model_memory(frame.model)
            if hasattr(frame, "optimizers"):
                for optimizer in frame.optimizers:
                    total_memory_bytes += self._get_optimizer_memory(optimizer)
        
        # Convert the total memory from bytes to megabytes.
        total_memory_mb = total_memory_bytes / (1024 * 1024)
        return total_memory_mb
    
    def _get_model_memory(self, model) -> int:
        """
        Calculate the memory used by the model's parameters.
        
        Returns:
            int: Memory in bytes.
        """
        memory = 0
        for param in model.parameters():
            memory += param.nelement() * param.element_size()
        return memory
    
    def _get_optimizer_memory(self, optimizer) -> int:
        """
        Calculate the memory used by the optimizer's state.
        
        Returns:
            int: Memory in bytes.
        """
        memory = 0
        for state in optimizer.state.values():
            memory += self._get_memory_of_obj(state)
        return memory



class Hessian_measurement(MeasurementUnit):
    """
    Measurement unit that approximates the full Hessian of the loss function for the model using 
    Hessian-vector products (HVPs) with finite differences. A single batch from the training loader 
    is used to perform the approximation. The full Hessian matrix is written to the log with the appropriate 
    begin/end markers.
    """
    def __init__(self):
        super().__init__("Hessian Measurement")
    
    
    def measure(self, frame) -> float:
        model = frame.center_model
        device = next(model.parameters()).device
        data_iter = iter(frame.train_loader)
        try:
            inputs, targets = next(data_iter)
        except StopIteration:
            return 0.0
        inputs, targets = inputs.to(device), targets.to(device)
        
        criterion = getattr(frame, "criterion", torch.nn.MSELoss())

        # --------------------------------------------------------------
        # Fast spectrum estimation (block‑aware, no full Hessian build)
        # --------------------------------------------------------------
        min_eig_diag, max_eig_offdiag = fast_hessian_spectrum(
            model, inputs, targets, criterion, max_iter=30
        )

        # Optionally, you might still want to draw the heat‑map as before.
        # Because that requires the *full* Hessian we skip it by default to
        # keep the routine lightweight.  Uncomment the following block if the
        # visualisation is important *and* the model is small enough.
        # ------------------------------------------------------------------
        # hessian = _compute_hessian(model, inputs, targets, criterion)
        # ls = _get_block_parameter_counts(model)
        # hessian_blocks = _decompose_matrix_to_blocks(hessian, ls)
        # block_norms = _compute_block_operator_norms(hessian_blocks)
        # [plotting code…]

        # Return the requested scalars as a tuple.
        return float(min_eig_diag), float(max_eig_offdiag)


def _compute_hessian(model, inputs, targets, criterion, epsilon=1e-4):
    """
    Computes the Hessian matrix of the loss with respect to the model parameters using finite differences.
    
    Inputs:
      model:        The neural network model (a torch.nn.Module).
      inputs:       Tensor of input data to the model.
      targets:      Tensor of target outputs corresponding to the inputs.
      criterion:    Loss function used to compute the loss between model outputs and targets.
      epsilon:      Small perturbation value for finite difference approximation (default is 1e-4).
    """
    device = next(model.parameters()).device
    
    # Save original parameters as a flattened vector
    original_vector = torch.nn.utils.parameters_to_vector(model.parameters()).detach()
    n_params = original_vector.numel()
    
    # Compute baseline loss and its gradient
    model.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    grad = torch.autograd.grad(loss, model.parameters(), create_graph=False)
    grad_vec = torch.nn.utils.parameters_to_vector(grad).detach()
    
    # Initialize Hessian matrix
    H = torch.zeros(n_params, n_params, device=device)
    for i in range(n_params):
        perturb = torch.zeros_like(original_vector)
        perturb[i] = epsilon
        perturbed_vector = original_vector + perturb
        # Update model parameters with the perturbed vector
        torch.nn.utils.vector_to_parameters(perturbed_vector, model.parameters())
        
        model.zero_grad()
        outputs_perturbed = model(inputs)
        loss_perturbed = criterion(outputs_perturbed, targets)
        grad_perturbed = torch.autograd.grad(loss_perturbed, model.parameters(), create_graph=False)
        grad_perturbed_vec = torch.nn.utils.parameters_to_vector(grad_perturbed).detach()
        
        hvp = (grad_perturbed_vec - grad_vec) / epsilon
        H[:, i] = hvp
    
    # Restore original parameters
    torch.nn.utils.vector_to_parameters(original_vector, model.parameters())
    return H

def _decompose_matrix_to_blocks(H, l):
    """
    Decomposes a square PyTorch tensor H into block tensors based on a list of integers l.

    Args:
        H (torch.Tensor): The n*n square PyTorch tensor to decompose.
                          Requires H.dim() == 2.
        l (list of int): A list of positive integers [l1, l2, ...]
                         representing the dimensions of the square diagonal blocks.
                         The sum of elements in l must equal n.

    Returns:
        list of lists of torch.Tensor:
            A nested list representing the block tensor H'.
            H'[i][j] is the block (a torch.Tensor) at the i-th block row
            and j-th block column.
            Returns None if the input is invalid.

    Raises:
        ValueError: If inputs are invalid (e.g., H is not a 2D tensor, H is not square,
                    sum of l != n, l contains non-positive integers).
        TypeError: If H is not a PyTorch tensor.
    """
    # --- Input Validation ---
    if not isinstance(H, torch.Tensor):
      raise TypeError(f"Input H must be a PyTorch tensor (got {type(H)}).")

    if H.dim() != 2:
        raise ValueError(f"Input tensor H must be 2-dimensional (got {H.dim()} dimensions).")

    n_rows, n_cols = H.shape
    if n_rows != n_cols:
        raise ValueError(f"Input tensor H must be square (got shape {H.shape}).")

    n = n_rows

    if not isinstance(l, list) or not all(isinstance(x, int) and x > 0 for x in l):
        raise ValueError("Input l must be a list of positive integers.")

    if sum(l) != n:
        raise ValueError(f"The sum of elements in l ({sum(l)}) must equal the dimension of H ({n}).")

    # --- Block Decomposition ---
    block_matrix = []
    # Use torch.cumsum for indices calculation
    indices = torch.tensor([0] + l).cumsum(dim=0) # Start indices [0, l1, l1+l2, ...]

    for i in range(len(l)):
        block_row = []
        rows = slice(indices[i].item(), indices[i+1].item()) # Row slice for the i-th block row

        for j in range(len(l)):
            cols = slice(indices[j].item(), indices[j+1].item()) # Column slice for the j-th block col

            # Extract the block using tensor slicing
            block = H[rows, cols]
            block_row.append(block)

        block_matrix.append(block_row)

    return block_matrix

def _get_block_parameter_counts(model):
  """
  Calculates the number of parameters for each layer in a PyTorch model.

  Args:
    model: A PyTorch nn.Module object.

  Returns:
    A list of integers, where each integer represents the number of
    parameters in the corresponding layer of the model.
  """
  param_counts = []
  for block in model.blocks:
    num_params = sum(p.numel() for p in block.parameters())
    param_counts.append(num_params)
  return param_counts
 
def _compute_block_operator_norms(block_matrix):
    """
    Compute the operator norm (largest singular value) of each block in a block matrix.

    Args:
        block_matrix (list of list of torch.Tensor): nested list of blocks of shape (n, m).

    Returns:
        torch.Tensor: a tensor of shape (n, m) where each entry is the operator norm of the
                      corresponding block.
    """
    if not block_matrix:
        return torch.empty(0)
    n = len(block_matrix)
    m = len(block_matrix[0])
    device = block_matrix[0][0].device
    norms = torch.zeros((n, m), device=device)
    for i in range(n):
        for j in range(m):
            block = block_matrix[i][j]
            # compute largest singular value
            try:
                op_norm = torch.linalg.norm(block, ord=2)
            except AttributeError:
                # fallback to SVD
                svals = torch.linalg.svdvals(block)
                op_norm = svals.max()
            norms[i, j] = op_norm
    return norms

# -----------------------------------------------------------------------------
#  Fast Hessian spectrum utilities
# -----------------------------------------------------------------------------

def _flatten_params(model):
    """Return a flat 1‑D view of all parameters (with gradients) and a list of
    slices that map every *block* (as defined by ``model.blocks``) to its
    position in the flat vector.

    The helper is central to the fast Hessian‑vector‑product routines: we need a
    deterministic mapping from a *block index* to the contiguous range of
    coordinates that correspond to that block in the flattened parameter
    vector.
    """
    # Build a flat list of parameters while recording the start/stop indices
    flat_params: list[torch.Tensor] = []
    block_slices: list[slice] = []

    cursor = 0
    for blk in model.blocks:
        blk_tensors = list(blk.parameters())
        flat_params.extend(blk_tensors)

        blk_numel = sum(p.numel() for p in blk_tensors)
        block_slices.append(slice(cursor, cursor + blk_numel))
        cursor += blk_numel

    # ``torch.nn.utils.parameters_to_vector`` keeps the order of the supplied
    # iterable, so we can safely flatten the *concatenated* list.
    flat_vector = torch.nn.utils.parameters_to_vector(flat_params)
    return flat_vector, block_slices


def _make_hvp_fn(model, inputs, targets, criterion):
    """Return a closure that computes *one* Hessian‑vector product (HVP).

    The closure recreates the backwards graph each call so that the autograd
    engine does not accumulate computational history across iterations.  Given
    the small number of HVP calls we expect (≈ few × 10²), the overhead is
    negligible compared with the cost of the forward and backward passes.
    """

    # Keep a reference to the *parameter list* once so that we do not have to
    # traverse ``model.parameters()`` on every invocation.
    params = list(model.parameters())

    def hvp(vec: torch.Tensor) -> torch.Tensor:
        """Compute H·vec where *H* is the Hessian of *loss* w.r.t. *params*.

        Args
        ----
        vec (torch.Tensor): 1‑D tensor with the same number of elements as the
            concatenation of ``params``.
        Returns
        -------
        torch.Tensor: the Hessian‑vector product, shape = vec.shape.
        """
        if vec.requires_grad:
            vec = vec.detach()

        # Forward pass & gradient
        model.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # First‑order gradients (create_graph=True so we can differentiate them)
        first_grads = torch.autograd.grad(
            loss, params, create_graph=True, retain_graph=True, allow_unused=False
        )
        grad_flat = torch.cat([g.reshape(-1) for g in first_grads])

        # Dot(grad, vec) – scalar
        grad_dot_vec = torch.dot(grad_flat, vec)

        # Second backward: gradient of the dot product is the HVP.
        hvp_parts = torch.autograd.grad(
            grad_dot_vec, params, retain_graph=False, allow_unused=False
        )
        hvp_flat = torch.cat([g.reshape(-1) for g in hvp_parts])
        return hvp_flat.detach()

    return hvp


def _power_iteration(matvec, dim, *, max_iter=50, tol=1e-6, device=None):
    """Compute *largest* eigenvalue of a symmetric operator by power iteration.

    Args
    ----
    matvec (callable): function v ↦ A v.
    dim (int): size of the input vector space.
    max_iter (int): maximum number of power iterations.
    tol (float): relative convergence tolerance on the eigenvalue.
    device: torch device for the work vectors.

    Returns
    -------
    (eigval, eigvec): tuple of the dominant eigenvalue (float) and the
    corresponding eigenvector (torch.Tensor).
    """
    rng = torch.Generator(device=device)
    v = torch.randn(dim, device=device, generator=rng)
    v = v / v.norm()

    eigval = None
    for _ in range(max_iter):
        Av = matvec(v)
        new_eigval = torch.dot(v, Av)
        if eigval is not None and torch.isfinite(eigval):
            if torch.abs(new_eigval - eigval) / (torch.abs(eigval) + 1e-12) < tol:
                eigval = new_eigval
                break
        eigval = new_eigval
        v = Av / Av.norm()

    return eigval.item(), v.detach()


def fast_hessian_spectrum(model, inputs, targets, criterion, *, max_iter=50):
    """Efficiently estimate two scalar quantities from the Hessian *H*:

        1.  λ_min = min eigval(diag(H))   – the smallest eigenvalue of the *block‑diagonal* part.
        2.  λ_max = max eigval(H − diag(H)) – the largest eigenvalue of the off‑diagonal part.

    The implementation *never* materialises the full Hessian.  It relies only
    on Hessian‑vector products (HVPs) which are computed via automatic
    differentiation.  The block structure is defined by ``model.blocks`` – each
    entry in that iterable is treated as one block.  All parameters belonging
    to a given block are contiguous in the flattened parameter vector, so the
    block‑diagonal projector is cheap to apply.
    """

    device = next(model.parameters()).device

    # ---------------------------------------------------------------------
    # Pre‑computation:  flatten parameters and build slice indices per block
    # ---------------------------------------------------------------------
    _, block_slices = _flatten_params(model)
    total_params = block_slices[-1].stop

    # Create HVP closure
    hvp_fn = _make_hvp_fn(model, inputs, targets, criterion)

    # ---------------------------------------------------------------
    # Helper:  block‑diagonal Hessian‑vector product  (diag(H)·v)
    # ---------------------------------------------------------------
    def diag_hvp(v: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(v)
        # We avoid an extra ``detach`` here because ``v`` is already detached in
        # the power‑iteration routines and we never build a computational graph
        # inside this helper.
        for sl in block_slices:
            if sl.stop - sl.start == 0:
                continue
            v_block = v[sl]
            if v_block.abs().sum() == 0:
                continue  # Skip empty vectors to save work

            # Build a *sparse* full‑length vector with only the current block
            w = torch.zeros(total_params, device=device)
            w[sl] = v_block
            out_block = hvp_fn(w)[sl]
            out[sl] = out_block  # No need to clone – ``out`` is distinct.
        return out

    # ---------------------------------------------------------------
    # λ_min  (diag(H)) – iterate *per block* on the *negated* operator to get
    # the smallest eigenvalue.
    # ---------------------------------------------------------------
    min_eig_diag = float("inf")
    for sl in block_slices:
        m = sl.stop - sl.start
        if m == 0:
            continue

        # Build matvec for the *negated* block: v ↦ - (diag(H) v)
        def block_neg_matvec(v_local, sl=sl):  # default arg binds slice
            full_v = torch.zeros(total_params, device=device)
            full_v[sl] = v_local
            # Extract only the block component after HVP
            res = -hvp_fn(full_v)[sl]
            return res

        eigval_neg, _ = _power_iteration(block_neg_matvec, m, max_iter=max_iter, device=device)
        # Convert back to *actual* eigenvalue of the block
        min_eig_diag = min(min_eig_diag, -eigval_neg)

    # -----------------------------------------------------------------
    # λ_max  of  H − diag(H)  via power iteration with a custom matvec.
    # Each iteration requires  (1 + #blocks_nonzero)  HVPs.
    # -----------------------------------------------------------------
    def offdiag_matvec(v: torch.Tensor) -> torch.Tensor:
        # Full Hessian‑vector product
        Hv = hvp_fn(v)
        # Subtract the block‑diagonal piece
        diagHv = diag_hvp(v)
        return Hv - diagHv

    max_eig_offdiag, _ = _power_iteration(offdiag_matvec, total_params, max_iter=max_iter, device=device)

    return min_eig_diag, max_eig_offdiag

# =============================================================================
#  Measurement: Block‑wise off‑diagonal singular values of Hessian
# =============================================================================


class HessianBlockInteractionMeasurement(MeasurementUnit):
    """Measurement unit that builds a *block interaction matrix* H′ of size
    (B × B), where *B* is the number of parameter blocks in ``model.blocks``.

    •  H′[i, i] = 0.
    •  For i ≠ j, H′[i, j] = σ_max(H_ij), the largest singular value of the
       off‑diagonal Hessian block that couples block *i* with block *j*.

    The class relies only on Hessian‑vector products and therefore scales to
    large models (millions of parameters) as long as the number of blocks B is
    modest (2 – 12 as in your use‑case).
    A heat‑map of H′ is stored in
        figures/Measurements/Hessian_Measurement/block_offdiag_eig_heatmap.pdf
    """

    def __init__(self, max_iter: int = 30, scale_offdiag: bool = True):
        """Create the measurement unit.

        Parameters
        ----------
        max_iter : int, default=30
            Power‑iteration steps used to estimate each block singular value.
        scale_offdiag : bool, default=False
            If *True*, each row *i* of the returned matrix H′ is divided by the
            diagonal entry H′[i,i].  This normalises the self‑interaction to 1
            and expresses off‑diagonal couplings relative to it.
        """

        super().__init__("Hessian Block Interaction Measurement")
        self.max_iter = max_iter
        self.scale_offdiag = scale_offdiag

        # ------------------------------------------------------------------
        # Prepare figure directory: clear old content once per session
        # ------------------------------------------------------------------
        self._fig_dir = os.path.join("figures", "Measurements", "Hessian_Measurement")
        if os.path.isdir(self._fig_dir):
            for f in os.listdir(self._fig_dir):
                existing = os.path.join(self._fig_dir, f)
                if os.path.isfile(existing):
                    os.remove(existing)
        else:
            os.makedirs(self._fig_dir, exist_ok=True)

        self._fig_counter = 0  # enumerate successive calls

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _diag_hvp_two_blocks(hvp_fn, slice_i, slice_j, total_dim, v_full, buffer):
        """Compute (diag H)·v, but only for blocks *i* and *j*.

        Args
        ----
        hvp_fn: function performing one Hessian‑vector product
        slice_i, slice_j: slices corresponding to blocks i and j
        total_dim: full parameter dimension
        v_full (Tensor): full‑length vector containing the current iterate
        buffer (Tensor): pre‑allocated tensor for temporary vectors
        """
        # Contribution from block i
        buffer.zero_()
        buffer[slice_i] = v_full[slice_i]
        diag_i = hvp_fn(buffer)[slice_i]

        # Contribution from block j
        buffer.zero_()
        buffer[slice_j] = v_full[slice_j]
        diag_j = hvp_fn(buffer)[slice_j]

        out = torch.zeros_like(v_full)
        out[slice_i] = diag_i
        out[slice_j] = diag_j
        return out

    def _pair_sigma_max(self, hvp_fn, slice_i, slice_j, total_dim, device):
        """Return σ_max(H_ij) via power iteration on the symmetric matrix
        M = [[0, H_ij], [H_ji, 0]].
        """
        dim_i = slice_i.stop - slice_i.start
        dim_j = slice_j.stop - slice_j.start
        sub_dim = dim_i + dim_j

        # Pre‑allocated tensors to minimise allocations inside matvec
        full_vec = torch.zeros(total_dim, device=device)
        buffer = torch.zeros_like(full_vec)

        def matvec(local_v):
            # local_v is concatenation of (v_i, v_j)
            v_i = local_v[:dim_i]
            v_j = local_v[dim_i:]

            full_vec.zero_()
            full_vec[slice_i] = v_i
            full_vec[slice_j] = v_j

            Hv = hvp_fn(full_vec)
            diagHv = self._diag_hvp_two_blocks(
                hvp_fn, slice_i, slice_j, total_dim, full_vec, buffer
            )
            off = Hv - diagHv

            return torch.cat([off[slice_i], off[slice_j]])

        eigval, _ = _power_iteration(matvec, sub_dim, max_iter=self.max_iter, device=device)
        return abs(eigval)

    def _block_sigma_max(self, hvp_fn, slice_i, total_dim, device):
        """Largest eigenvalue magnitude of the *diagonal* block H_ii."""
        dim_i = slice_i.stop - slice_i.start

        full_vec = torch.zeros(total_dim, device=device)

        def matvec(v_local):
            full_vec.zero_()
            full_vec[slice_i] = v_local
            return hvp_fn(full_vec)[slice_i]

        eigval, _ = _power_iteration(matvec, dim_i, max_iter=self.max_iter, device=device)
        return abs(eigval)

    # ------------------------------------------------------------------
    # MeasurementUnit interface implementation
    # ------------------------------------------------------------------

    def measure(self, frame):
        model = frame.center_model
        device = next(model.parameters()).device

        # Fetch a single mini‑batch
        data_iter = iter(frame.train_loader)
        try:
            inputs, targets = next(data_iter)
        except StopIteration:
            return torch.tensor([])

        inputs, targets = inputs.to(device), targets.to(device)
        criterion = getattr(frame, "criterion", torch.nn.MSELoss())

        # Prepare HVP function and block slices
        _, block_slices = _flatten_params(model)
        total_dim = block_slices[-1].stop
        B = len(block_slices)

        hvp_fn = _make_hvp_fn(model, inputs, targets, criterion)

        # Allocate result matrix H′
        Hprime = torch.zeros(B, B, device=device)

        # Diagonal blocks
        for i in range(B):
            Hprime[i, i] = self._block_sigma_max(
                hvp_fn, block_slices[i], total_dim, device
            )

        # Off‑diagonal blocks (compute only for i<j, symmetry exploited)
        for i in range(B):
            for j in range(i + 1, B):
                sigma = self._pair_sigma_max(
                    hvp_fn, block_slices[i], block_slices[j], total_dim, device
                )
                Hprime[i, j] = sigma
                Hprime[j, i] = sigma

        # ------------------ visualisation ------------------
        # Optional scaling of rows by their diagonal terms
        if self.scale_offdiag:
            for i in range(B):
                diag_val = Hprime[i, i].abs().clamp_min(1e-12)  # avoid div‑zero
                Hprime[i] = Hprime[i] / diag_val

        arr = Hprime.cpu().numpy()
        # Zero out lower triangle for visual clarity
        arr_plot = np.triu(arr)

        plt.figure(figsize=(6, 5))
        plt.imshow(arr_plot, cmap="Greens", interpolation="nearest")
        plt.colorbar()
        plt.title("Block interaction matrix σ_max (upper‑triangular)")
        plt.tight_layout()
        filename = f"block_offdiag_eig_heatmap_{self._fig_counter:03d}.pdf"
        plt.savefig(os.path.join(self._fig_dir, filename))
        plt.close()

        self._fig_counter += 1

        return Hprime.detach()

    # Override logger to handle matrix output
    def log_measurement(self, measurement):
        array_str = np.array2string(measurement.cpu().numpy(), precision=6, separator=", ")
        self.reporter.write(array_str + "\n")
        self.reporter.flush()