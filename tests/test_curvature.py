import sympy as sp
import pytest
from diffgeo import create_metric, christoffel_symbols, riemann_tensor, ricci_tensor, scalar_curvature

def test_unit_sphere_scalar_curvature():
    """Test scalar curvature for a unit 2-sphere: should be 2."""

    # 2-sphere metric function
    def create_func(theta, _):
        return [
            [1, 0],
            [0, sp.sin(theta)**2]
        ]

    # Build objects
    metric = create_metric("theta phi", create_func)
    gamma = christoffel_symbols(metric)
    riemann = riemann_tensor(metric, gamma)
    ricci = ricci_tensor(metric, riemann)
    scalar = scalar_curvature(metric, ricci)

    # Assert scalar curvature
    assert scalar == 2, f"Scalar curvature mismatch: {scalar}"

    # Optional: print nonzero entries for inspection
    print("\nRiemann tensor (nonzero entries):")
    for key, val in riemann._data.items():
        print(f"{key}: {val}")

    print("\nRicci tensor (nonzero entries):")
    for key, val in ricci._data.items():
        print(f"{key}: {val}")

    print("\nScalar curvature:", scalar)