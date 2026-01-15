import torch
from typing import Optional, Tuple
import math

# -----------------------------
# Utilities: closure and helpers
# Convert from skbio functions to torch
# -----------------------------

def torch_closure(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # Normalize compositions along `dim` so they sum to 1
    return x / x.sum(dim=dim, keepdim=True)

def torch_multi_replace(x: torch.Tensor, delta: Optional[float] = None, dim: int = -1) -> torch.Tensor:
    """
    Multiplicative replacement of zeros (like skbio.multi_replace).
    Operates on compositions, returns strictly positive closed compositions.
    """
    x = torch_closure(x, dim=dim)
    z_mask = (x == 0)
    D = x.shape[dim]
    if delta is None:
        delta = (1.0 / D) ** 2
    # Count zeros per composition
    tot = z_mask.sum(dim=dim, keepdim=True)  # number of zeros per row
    zcnts = 1.0 - tot * delta
    if torch.any(zcnts < 0):
        raise ValueError("Multiplicative replacement created negative proportions. Use a smaller delta.")
    # Replace zeros with delta, scale non-zeros by zcnts
    # We need broadcasting that respects `dim`:
    # Ensure zcnts has same shape as x for broadcasting
    out = torch.where(z_mask, torch.tensor(delta, dtype=x.dtype, device=x.device), zcnts * x)
    # Already sums to 1 per row due to construction; squeeze not needed in torch
    return out

def _move_axis_to_last(x: torch.Tensor, dim: int) -> Tuple[torch.Tensor, Tuple[int, ...], Tuple[int, ...]]:
    """
    Move axis `dim` to the last position, returning the permuted tensor and permutations.
    """
    ndim = x.ndim
    dim = dim % ndim
    perm = tuple([i for i in range(ndim) if i != dim] + [dim])
    inv = [0] * ndim
    for i, p in enumerate(perm):
        inv[p] = i
    inv = tuple(inv)
    return x.permute(perm), perm, inv

# ----------------------------------------
# CLR transform and its inverse (skbio-like)
# ----------------------------------------

def clr(x: torch.Tensor, dim: int = -1, validate: bool = True) -> torch.Tensor:
    """
    Centre log-ratio transform along axis `dim`.
    Requires strictly positive compositions if validate=True.
    """
    if validate and torch.any(x <= 0):
        raise ValueError("CLR requires strictly positive components; found zeros or negatives.")
    x = torch_closure(x, dim=dim)
    # Work along last axis
    x_last, perm, inv = _move_axis_to_last(x, dim)
    l = torch.log(x_last)
    clr_last = l - l.mean(dim=-1, keepdim=True)
    return clr_last.permute(inv)

def clr_inv(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Inverse CLR transform along axis `dim`.
    Implements skbio's _clr_inv: softmax-like exponentiation and closure.
    """
    z_last, perm, inv = _move_axis_to_last(z, dim)
    # For numerical stability: subtract max before exp
    diff = torch.exp(z_last - z_last.max(dim=-1, keepdim=True).values)
    out_last = torch_closure(diff, dim=-1)
    return out_last.permute(inv)

# ------------------------------------------
# Gram–Schmidt basis (skbio._gram_schmidt_basis)
# ------------------------------------------

def gram_schmidt_basis(D: int, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Return a (D-1, D) orthonormal basis in the simplex (CLR coordinates),
    following skbio._gram_schmidt_basis.
    """
    # skbio builds a (D, D-1) intermediate, then transposes
    basis = torch.zeros((D, D - 1), dtype=dtype, device=device)
    for j in range(D - 1):
        i = j + 1
        e = torch.cat([
            torch.full((i,), 1.0 / i, dtype=dtype, device=device),
            torch.tensor([-1.0], dtype=dtype, device=device),
            torch.zeros((D - i - 1,), dtype=dtype, device=device)
        ]) * math.sqrt(i / (i + 1))
        basis[:, j] = e
    return basis.T  # shape (D-1, D)

def _check_basis_orthonormal(B: torch.Tensor, tol: float = 1e-4) -> None:
    """
    Check B @ B.T ≈ I within tolerance.
    """
    Dm1 = B.shape[0]
    eye = torch.eye(Dm1, dtype=B.dtype, device=B.device)
    prod = B @ B.T
    if not torch.all(torch.abs(prod - eye) < (tol * eye + 1e-6)):
        raise ValueError("Basis is not orthonormal.")

# ------------------------------------------
# ILR transform and its inverse (skbio.ilr)
# ------------------------------------------

def ilr_transform(x: torch.Tensor,
                  basis: Optional[torch.Tensor] = None,
                  dim: int = -1,
                  validate: bool = True,
                  handle_zeros: bool = False,
                  delta: Optional[float] = None) -> torch.Tensor:
    """
    ILR transform along axis `dim`.
    - x: compositions (must be strictly positive if validate=True).
    - basis: (D-1, D) orthonormal basis in CLR coordinates. If None, Gram–Schmidt basis is used.
    - handle_zeros: if True, apply multiplicative replacement on zeros before transform (like skbio.multi_replace).
    """
    # Move axis to last for computation
    x_last, perm, inv = _move_axis_to_last(x, dim)
    D = x_last.shape[-1]

    # Handle zeros if requested; otherwise require strictly positive
    if handle_zeros:
        x_last = torch_multi_replace(x_last, delta=delta, dim=-1)
    elif validate and torch.any(x_last <= 0):
        raise ValueError("ILR requires strictly positive components; set handle_zeros=True or pre-process zeros.")

    # CLR along last axis
    l = torch.log(torch_closure(x_last, dim=-1))
    clr_last = l - l.mean(dim=-1, keepdim=True)

    # Prepare basis
    if basis is None:
        B = gram_schmidt_basis(D, device=x.device, dtype=x.dtype)
    else:
        B = basis.to(device=x.device, dtype=x.dtype)
        # Optional validation: orthonormality and shape
        if validate:
            if B.ndim != 2 or B.shape != (D - 1, D):
                raise ValueError(f"Basis must have shape ({D-1}, {D}); got {tuple(B.shape)}.")
            _check_basis_orthonormal(B)

    # Project CLR onto basis: tensordot along last axis of clr and columns of basis
    # Equivalent to matmul with B.T
    ilr_last = clr_last @ B.T  # result has last dim D-1
    return ilr_last.permute(inv)

def ilr_transform_inverse(z: torch.Tensor,
                          basis: Optional[torch.Tensor] = None,
                          dim: int = -1,
                          validate: bool = True) -> torch.Tensor:
    """
    Inverse ILR transform along axis `dim`.
    - z: ILR coordinates with size D-1 along `dim`.
    - basis: (D-1, D) orthonormal basis. If None, Gram–Schmidt basis for D = z.shape[dim] + 1.
    Returns a composition (sums to 1 along `dim`).
    """
    # Move axis to last
    z_last, perm, inv = _move_axis_to_last(z, dim)
    Dm1 = z_last.shape[-1]
    D = Dm1 + 1

    # Prepare basis
    if basis is None:
        B = gram_schmidt_basis(D, device=z.device, dtype=z.dtype)
    else:
        B = basis.to(device=z.device, dtype=z.dtype)
        if validate:
            if B.ndim != 2 or B.shape != (Dm1, D):
                raise ValueError(f"Basis must have shape ({Dm1}, {D}); got {tuple(B.shape)}.")
            _check_basis_orthonormal(B)

    # Project back to CLR space: z * basis (along basis rows)
    clr_last = z_last @ B  # result last dim D

    # CLR inverse: exp-shift + closure (softmax-like), along last axis
    diff = torch.exp(clr_last - clr_last.max(dim=-1, keepdim=True).values)
    x_last = torch_closure(diff, dim=-1)
    return x_last.permute(inv)