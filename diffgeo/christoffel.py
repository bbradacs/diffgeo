import sympy as sp
from .derivatives import d

def christoffel_symbols(metric):
    coords = metric.coords
    g = metric.g
    g_inv = metric.g_inv
    dim = metric.dim

    def gamma(k, i, j):
        return sp.simplify(
            sp.Rational(1, 2) * sum(
                g_inv[k, l] * (
                    d(g[j, l], coords[i]) +
                    d(g[i, l], coords[j]) -
                    d(g[i, j], coords[l])
                )
                for l in range(dim)
            )
        )

    return [
        [
            [gamma(k, i, j) for j in range(dim)]
            for i in range(dim)
        ]
        for k in range(dim)
    ]