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


