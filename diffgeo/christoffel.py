import sympy as sp
from diffgeo import Tensor, d

class Gamma(Tensor):
    def __init__(self, data, dim):
        super().__init__(data, ['up', 'down', 'down'], dim)
        """
        data: dict with keys (k, i, j) and values Γ^k_{ij}
        """
        self._data = data

    def __getitem__(self, key):
        """Access Christoffel symbols like Gamma[k, i, j]"""
        return self._data.get(key, 0)  # default to 0 if missing

def christoffel_symbols(g_down, g_up):
    """
    Compute Christoffel symbols from a Metric object g_mn and its inverse g^mn, and return a Gamma object.
    """
    coords = g_down.coords
    dim = g_down.dim
    
    gamma_dict = {}
    
    for k in range(dim):
        for i in range(dim):
            for j in range(dim):
                s = sum(
                    g_up[k, l] * (
                        d(g_down[l, i], coords[j]) +
                        d(g_down[l, j], coords[i]) -
                        d(g_down[i, j], coords[l])
                    )
                    for l in range(dim)
                )
                gamma_dict[(k, i, j)] = sp.simplify(s / 2)
    
    return Gamma(gamma_dict, dim)
