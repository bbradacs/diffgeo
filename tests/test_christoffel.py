import sympy as sp
from diffgeo import Metric, Gamma, christoffel_symbols, christoffel_terms, create_metric

def test_flat_space():

    metric = Metric("x y", [
        [1, 0],
        [0, 1]
    ])

    gamma = christoffel_symbols(metric)

    for k in range(2):
        for i in range(2):
            for j in range(2):
                assert gamma[k, i, j] == 0

def test_curved_space():
    def create_func(r, theta, phi):
        return [
            [r, 0, 0],
            [0, r, 0],
            [0, 0, r**2 * sp.sin(theta)**2]
        ]

    metric = create_metric("r theta phi", create_func)
    gamma = christoffel_symbols(metric)
    terms = christoffel_terms(metric, gamma)
    simplified = {k: sp.simplify(v) for k, v in terms.items()}
    print(simplified)

