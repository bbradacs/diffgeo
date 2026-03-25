import sympy as sp
from .derivatives import d

class Gamma:
    def __init__(self, data):
        """
        data: dict with keys (k, i, j) and values Γ^k_{ij}
        """
        self._data = data

    def __getitem__(self, key):
        """Access Christoffel symbols like Gamma[k, i, j]"""
        return self._data.get(key, 0)  # default to 0 if missing

def christoffel_symbols(g):
    """
    Compute Christoffel symbols from a Metric object g and return a Gamma object.
    """
    coords = g.coords
    dim = g.dim
    
    gamma_dict = {}
    
    for k in range(dim):
        for i in range(dim):
            for j in range(dim):
                s = sum(
                    g.inv[k, l] * (
                        d(g[l, i], coords[j]) +
                        d(g[l, j], coords[i]) -
                        d(g[i, j], coords[l])
                    )
                    for l in range(dim)
                )
                gamma_dict[(k, i, j)] = sp.simplify(s / 2)
    
    return Gamma(gamma_dict)