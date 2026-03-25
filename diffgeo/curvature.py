import sympy as sp
from .derivatives import d
import diffgeo
from diffgeo import create_metric

def riemann_tensor(metric, Gamma):
    coords = metric.coords
    dim = metric.dim

    def R(k, l, i, j):
        return sp.simplify(
            # derivative terms
            d(Gamma[k, j, l], coords[i]) -
            d(Gamma[k, i, l], coords[j])
            +
            # quadratic terms
            sum(Gamma[k, i, m] * Gamma[m, j, l] for m in range(dim)) -
            sum(Gamma[k, j, m] * Gamma[m, i, l] for m in range(dim))
        )

    # Build dictionary instead of nested lists
    R_dict = {}
    for k in range(dim):
        for l in range(dim):
            for i in range(dim):
                for j in range(dim):
                    val = R(k, l, i, j)
                    if val != 0:  # optional: store only nonzero
                        R_dict[(k, l, i, j)] = val

    return R_dict

def ricci_tensor(metric, R):
    dim = metric.dim

    def ricci(i, j):
        return sp.simplify(
            sum(R.get((k, i, k, j), 0) for k in range(dim))
        )

    # Build dictionary instead of nested lists
    Ric_dict = {}
    for i in range(dim):
        for j in range(dim):
            val = ricci(i, j)
            if val != 0:  # optional: store only nonzero entries
                Ric_dict[(i, j)] = val

    return Ric_dict

def scalar_curvature(metric, Ricci):
    dim = metric.dim

    return sp.simplify(
        sum(
            metric.inv[i, j] * Ricci[i][j]
            for i in range(dim)
            for j in range(dim)
        )
    )
