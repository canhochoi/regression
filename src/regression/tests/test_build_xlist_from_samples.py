import unittest

import numpy as np
import pandas as pd
from scipy import sparse

from regression.preprocessing import PseudobulkProcessor


class _DummyObs:
    def __init__(self, index: pd.Index):
        self.index = index


class _DummyAdata:
    def __init__(self, X, obs_names, layers=None):
        self.X = X
        self.obs_names = pd.Index(obs_names)
        self.obs = _DummyObs(self.obs_names)
        self.layers = {} if layers is None else dict(layers)


def _make_processor(adata, cells_by_samples, sample_ids_in_order):
    proc = PseudobulkProcessor.__new__(PseudobulkProcessor)
    proc.adata = adata
    proc.cells_by_samples = cells_by_samples
    # only index matters for the assert
    proc.cell_type_proportions = pd.DataFrame(index=pd.Index(sample_ids_in_order))
    return proc


class TestBuildXListFromSamples(unittest.TestCase):
    def test_sparse_csr_counts(self):
        counts = sparse.csr_matrix(
            np.array(
                [
                    [1, 0, 2],
                    [0, 3, 0],
                    [4, 0, 5],
                    [0, 6, 0],
                ],
                dtype=np.float32,
            )
        )
        adata = _DummyAdata(X=counts, obs_names=["c0", "c1", "c2", "c3"], layers={"counts": counts})
        cells_by_samples = {"s0": ["c0", "c2"], "s1": ["c1", "c3"]}
        proc = _make_processor(adata, cells_by_samples, sample_ids_in_order=["s0", "s1"])

        X_list = PseudobulkProcessor._build_xlist_from_samples(proc)
        self.assertEqual(len(X_list), 2)
        np.testing.assert_array_equal(X_list[0].toarray(), counts[[0, 2], :].toarray())
        np.testing.assert_array_equal(X_list[1].toarray(), counts[[1, 3], :].toarray())

    def test_sparse_non_csr_counts(self):
        base = np.array([[1, 0], [0, 2], [3, 0]], dtype=np.float32)
        counts_coo = sparse.coo_matrix(base)
        adata = _DummyAdata(X=counts_coo, obs_names=["c0", "c1", "c2"], layers={"counts": counts_coo})
        cells_by_samples = {"s0": ["c2", "c0"]}
        proc = _make_processor(adata, cells_by_samples, sample_ids_in_order=["s0"])

        X_list = PseudobulkProcessor._build_xlist_from_samples(proc)
        self.assertEqual(len(X_list), 1)
        np.testing.assert_array_equal(X_list[0].toarray(), base[[2, 0], :])

    def test_dense_counts_fallback(self):
        dense = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        adata = _DummyAdata(X=dense, obs_names=["c0", "c1", "c2"], layers={"counts": dense})
        cells_by_samples = {"s0": ["c1", "c2"]}
        proc = _make_processor(adata, cells_by_samples, sample_ids_in_order=["s0"])

        X_list = PseudobulkProcessor._build_xlist_from_samples(proc)
        self.assertTrue(sparse.isspmatrix_csr(X_list[0]))
        np.testing.assert_array_equal(X_list[0].toarray(), dense[[1, 2], :])

    def test_missing_cell_id_raises(self):
        counts = sparse.csr_matrix(np.eye(2, dtype=np.float32))
        adata = _DummyAdata(X=counts, obs_names=["c0", "c1"], layers={"counts": counts})
        cells_by_samples = {"s0": ["c0", "c_missing"]}
        proc = _make_processor(adata, cells_by_samples, sample_ids_in_order=["s0"])

        with self.assertRaises(KeyError):
            PseudobulkProcessor._build_xlist_from_samples(proc)


if __name__ == "__main__":
    unittest.main()
