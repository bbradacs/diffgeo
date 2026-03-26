import pytest
from diffgeo import create_metric, christoffel_symbols, metric, riemann_tensor, ricci_tensor, scalar_curvature

def test_unit_sphere_scalar_curvature():
    """Test scalar curvature for a unit 2-sphere: should be 2."""

    # 2-sphere metric function
    def create_func(trig, theta, _):
        return [
            [1, 0],
            [0, trig.sin(theta)**2]
        ]

    # Build objects
    g_down = create_metric("theta phi", create_func)
    g_up = g_down.inv()
    gamma = christoffel_symbols(g_down, g_up)
    riemann = riemann_tensor(g_down, gamma)
    ricci = ricci_tensor(g_down, riemann)
    scalar = scalar_curvature(g_up, ricci)

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