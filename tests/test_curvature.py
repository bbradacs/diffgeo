import sympy as sp
from diffgeo import Metric, create_metric, christoffel_symbols, riemann_tensor, ricci_tensor, scalar_curvature

def test_curvature():

    def create_func(theta, phi):
        return [
            [1, 0],
            [0, sp.sin(theta)**2]
        ]

    metric = create_metric("theta phi", create_func)
    gamma = christoffel_symbols(metric)
    Riemann = riemann_tensor(metric, gamma)
    Ricci = ricci_tensor(metric, Riemann)
    R_scalar = scalar_curvature(metric, Ricci)

    print("Riemann:", Riemann)
    print("Ricci:", Ricci)
    print("Scalar Curvature:", R_scalar)    