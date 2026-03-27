import pytest
import sympy as sp
from diffgeo import create_metric_tensors, create_gamma_tensor, create_riemann_tensor, create_ricci_tensor, create_scalar_tensor

def test_flat_space():

    g_down, g_up = create_metric_tensors("x y", lambda *_ : [[1, 0], [0, 1]])

    gamma = create_gamma_tensor(g_down, g_up)

    for k in range(2):
        for i in range(2):
            for j in range(2):
                assert gamma[k, i, j] == 0

def test_curved_space():
    def create_func(sp, r, theta, phi):
        return [
            [r, 0, 0],
            [0, r, 0],
            [0, 0, r**2 * sp.sin(theta)**2]
        ]

    g_down, g_up = create_metric_tensors("r theta phi", create_func)
    gamma = create_gamma_tensor(g_down, g_up)
    assert(gamma.dim == 3)
    assert(gamma.rank == 3)


def test_spherically_symmetric() :
    import sympy as sp

    def create_func(trig, t, r, theta, phi):
        f = trig.Function('f')
        h = trig.Function('h')
        sin = trig.sin
        return [
            [-f(r), 0, 0, 0],
            [0, h(r), 0, 0],
            [0, 0, r**2, 0],
            [0, 0, 0, r**2 * sin(theta)**2]
        ]

    g_down_t, g_up_t = create_metric_tensors("t r theta phi", create_func)
    gamma = create_gamma_tensor(g_down_t, g_up_t)
    riemann = create_riemann_tensor(gamma)
    ricci = create_ricci_tensor(riemann)
    scalar = create_scalar_tensor(g_up_t, ricci)

    print("\nChristoffel Symbols (nonzero entries):")
    for key, val in gamma.items:
        print(f"{key}: {val}")

    print("\nRiemann tensor (nonzero entries):")
    for key, val in riemann.items:
        print(f"{key}: {val}")

    print("\nRicci tensor (nonzero entries):")
    for key, val in ricci.items:
        print(f"{key}: {val}")

    print("\nRicci scalar:")
    print(f"scalar.rank: {scalar.rank}")
    for key, val in scalar.items:
        print(f"{key}: {val}")

def test_unit_sphere_tensor():
    """Test scalar curvature for a unit 2-sphere: should be 2."""

    # 2-sphere metric function
    def create_func(trig, theta, _):
        sin = trig.sin
        return [
            [1, 0],
            [0, sin(theta)**2]
        ]

    # Build objects
    g_down_t, g_up_t = create_metric_tensors("theta phi", create_func)
    gamma   = create_gamma_tensor(g_down_t, g_up_t)
    riemann = create_riemann_tensor(gamma)
    ricci   = create_ricci_tensor(riemann)
    scalar  = create_scalar_tensor(g_up_t, ricci)


    # print nonzero entries for inspection
    print("\nRiemann tensor (nonzero entries):")
    for key, val in riemann.items:
        print(f"{key}: {val}")

    print("\nRicci tensor (nonzero entries):")
    for key, val in ricci.items:
        print(f"{key}: {val}")

    print("\nScalar curvature:")
    print(f"scalar.rank: {scalar.rank}")
    for key, val in scalar.items:
        print(f"{key}: {val}")

    # Assert scalar curvature
    assert scalar[()] == 2, f"Scalar curvature mismatch: {scalar}"


