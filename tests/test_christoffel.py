import sympy as sp
from diffgeo import Metric, christoffel_symbols

def test_flat_space():
    x, y = sp.symbols('x y')

    metric = Metric((x, y), [
        [1, 0],
        [0, 1]
    ])

    Gamma = christoffel_symbols(metric)

    for k in range(2):
        for i in range(2):
            for j in range(2):
                assert Gamma[k][i][j] == 0
