import pytest
import numpy as np
import torch

# Skip tests if scikit-bio is not available
skbio = pytest.importorskip("skbio.stats.composition")
from skbio.stats.composition import ilr, ilr_inv

from regression.neural_network.ilr_torch import ilr_transform, ilr_transform_inverse

import numpy as np
from skbio.stats.composition import ilr, ilr_inv


p = torch.tensor([[0.01,  0.09, 0.3, 0.4, 0.2],
                  [0.2, 0.15, 0.2, 0.05, 0.4]], dtype=torch.float32)

p_inv = ilr_transform(p, dim = 1)

def test_ilr_transform():
    # ILR transform (default Gram–Schmidt basis)
    assert np.allclose(p_inv.numpy(), ilr(p.numpy()))

def test_ilr_inv_transform():
    # Inverse back to composition
    p_rec = ilr_transform_inverse(p_inv, dim=1)
    assert np.allclose(p.numpy(), ilr_inv(p_inv.numpy()))
    assert torch.allclose(p, p_rec)


