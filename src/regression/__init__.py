# src/myrrr/__init__.py
# from .estimators import RidgeRRR, RidgeRRRCV
# from .preprocessing import CompositionalILR, StandardScalerX, CenterY, TrainTestSplit

# __all__ = ["RidgeRRR", "RidgeRRRCV", "CompositionalILR", "StandardScalerX", "CenterY", "TrainTestSplit"]

# src/myrrr/__init__.py
# Expose subpackages lazily, and optionally a curated set of top-level names.

import importlib

# Public subpackages (scikit-learn style)
__all__ = ["estimators", "preprocessing"]

def __getattr__(name):
    # Lazy-load subpackages on demand
    if name in __all__:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"{__name__} has no attribute {name!r}")

def __dir__():
    return sorted(__all__)

# # Optional: curated convenience imports (keep this list small and stable)
# from .preprocessing import CompositionalILR, StandardScalerX, CenterY, TrainTestSplit  # noqa: E402
# from .estimators import RidgeRRR, RidgeRRRCV  # noqa: E402

# # Extend __all__ with curated names for from myrrr import ...
# __all__ += ["CompositionalILR", "StandardScalerX", "CenterY", "TrainTestSplit", "RidgeRRR", "RidgeRRRCV"]