# tests/test_compositional_ilr.py
import numpy as np
import pandas as pd
import pytest

# Skip tests if scikit-bio is not available
skbio = pytest.importorskip("skbio.stats.composition")
from skbio.stats.composition import closure, multiplicative_replacement, ilr, ilr_inv

# Import your class from the package
from regression.preprocessing import CompositionalILR

def make_composition_df(n=50, D=6, seed=123, inject_zeros=True):
    """
    Create a synthetic compositional DataFrame with optional zeros.
    Rows roughly Dirichlet; optionally force some zeros to exercise zero replacement.
    """
    rng = np.random.default_rng(seed)
    # Dirichlet draws (strictly positive)
    alpha = np.ones(D)
    Y = rng.dirichlet(alpha, size=n)
    if inject_zeros:
        # Randomly set ~10% of entries to zero
        mask = rng.random(Y.shape) < 0.10
        Y[mask] = 0.0
        # Re-close rows so sums remain 1 (closure will be applied anyway)
        Y = closure(Y)
    cols = [f"part_{j}" for j in range(D)]
    idx = [f"s{i}" for i in range(n)]
    return pd.DataFrame(Y, index=idx, columns=cols)

def test_transform_matches_manual_pipeline():
    Y_comp_df = make_composition_df(n=40, D=7, inject_zeros=True)
    # Manual pipeline
    Y_closed = closure(Y_comp_df.values)
    Y_imp = multiplicative_replacement(Y_closed)
    Y_ilr_manual = ilr(Y_imp).astype(np.float64)

    # Class transform
    tx = CompositionalILR(zero_replacement=True).fit(Y_comp_df.values)
    Y_ilr_class = tx.transform(Y_comp_df.values)

    # Compare
    assert Y_ilr_class.shape == Y_ilr_manual.shape
    assert Y_ilr_class.dtype == np.float64
    assert np.allclose(Y_ilr_class, Y_ilr_manual, atol=1e-10)

def test_inverse_round_trip_positive_data():
    # Build strictly positive compositions (no zeros) for a clean round-trip
    Y_comp_df = make_composition_df(n=30, D=5, inject_zeros=False)
    tx = CompositionalILR(zero_replacement=False).fit(Y_comp_df.values)

    Y_ilr = tx.transform(Y_comp_df.values)  # closure + ilr (no zero replacement)
    Y_comp_rec = tx.inverse_transform(Y_ilr)  # ilr_inv

    # Rows should sum to ~1
    row_sums = Y_comp_rec.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-12)

    # Values should be strictly positive
    assert np.all(Y_comp_rec > 0.0)

    # Round-trip should be close to the closed original
    Y_closed = closure(Y_comp_df.values)
    assert np.allclose(Y_comp_rec, Y_closed, atol=1e-10)

def test_zero_replacement_enforces_strict_positivity():
    Y_comp_df = make_composition_df(n=20, D=4, inject_zeros=True)
    tx = CompositionalILR(zero_replacement=True).fit(Y_comp_df.values)

    Y_ilr = tx.transform(Y_comp_df.values)
    # ilr output shape and dtype
    assert Y_ilr.shape == (Y_comp_df.shape[0], Y_comp_df.shape[1] - 1)
    assert Y_ilr.dtype == np.float64

    # After inverse transform, compositions are positive and sum to 1
    Y_comp_rec = tx.inverse_transform(Y_ilr)
    assert np.all(Y_comp_rec > 0.0)
    assert np.allclose(Y_comp_rec.sum(axis=1), 1.0, atol=1e-12)

def test_no_zero_replacement_equals_manual_on_positive_data():
    Y_comp_df = make_composition_df(n=25, D=6, inject_zeros=False)
    tx = CompositionalILR(zero_replacement=False).fit(Y_comp_df.values)

    # Manual: closure then ilr
    Y_ilr_manual = ilr(closure(Y_comp_df.values)).astype(np.float64)
    Y_ilr_class = tx.transform(Y_comp_df.values)

    assert np.allclose(Y_ilr_class, Y_ilr_manual, atol=1e-12)