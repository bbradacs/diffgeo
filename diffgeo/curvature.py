import sympy as sp
from .derivatives import d

def riemann_tensor(metric, Gamma):
    coords = metric.coords
    dim = metric.dim

    def R(k, l, i, j):
        return sp.simplify(
            # derivative terms
            d(Gamma[k][j][l], coords[i]) -
            d(Gamma[k][i][l], coords[j])
            +
            # quadratic terms
            sum(
                Gamma[k][i][m] * Gamma[m][j][l]
                for m in range(dim)
            )
            -
            sum(
                Gamma[k][j][m] * Gamma[m][i][l]
                for m in range(dim)
            )
        )

    return [
        [
            [
                [R(k, l, i, j) for j in range(dim)]
                for i in range(dim)
            ]
            for l in range(dim)
        ]
        for k in range(dim)
    ]

def ricci_tensor(metric, R):
    dim = metric.dim

    def ricci(i, j):
        return sp.simplify(
            sum(
                R[k][i][k][j]
                for k in range(dim)
            )
        )

    return [
        [ricci(i, j) for j in range(dim)]
        for i in range(dim)
    ]

def scalar_curvature(metric, Ricci):
    dim = metric.dim

    return sp.simplify(
        sum(
            metric.g_inv[i, j] * Ricci[i][j]
            for i in range(dim)
            for j in range(dim)
        )
    )
