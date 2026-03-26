from itertools import product
import sympy as sp
from diffgeo import d

class Riemann:
    def __init__(self, data):
        """
        data: dict with keys (k, l, i, j) and values Riemann^k_{lij}
        """
        self._data = data

    def __getitem__(self, key):
        """Riemann[k, l, i, j]"""
        return self._data.get(key, 0)  # default to 0 if missing

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

    # Build dictionary
    R_dict = {}
    for idx in product(range(dim), repeat=4):  # 4 indices: k, l, i, j
        val = R(*idx)
        if val != 0:  # optional: store only nonzero
            R_dict[idx] = val

    return Riemann(R_dict)
