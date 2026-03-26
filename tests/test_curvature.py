import pytest
from diffgeo import create_metrics, christoffel_symbols, metric, riemann_tensor, ricci_tensor, scalar_curvature
from diffgeo.utils import christoffel_terms

def test_often_used() :
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

    g_down, g_up = create_metrics("t r theta phi", create_func)
    gamma = christoffel_symbols(g_down, g_up)
    riemann = riemann_tensor(g_down, gamma)
    ricci = ricci_tensor(g_down, riemann)
    
    terms = christoffel_terms(g_down, gamma)
    simplified = {k: sp.simplify(v) for k, v in terms.items()}
    print("\nChristoffel Symbols (nonzero entries):")
    for key, val in terms.items():
        print(f"{key}: {val}")

    print("\nRiemann tensor (nonzero entries):")
    for key, val in riemann._data.items():
        print(f"{key}: {val}")

    print("\nRicci tensor (nonzero entries):")
    for key, val in ricci._data.items():
        print(f"{key}: {val}")
