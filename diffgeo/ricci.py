from itertools import product
import sympy as sp
from diffgeo import Tensor

class Ricci(Tensor):
    def __init__(self, data, dim):
        super().__init__(data, ['down', 'down'], dim)
        """
        data: dict with keys (i, j) and values Ricci^i_{j}
        """
        self._data = data

    def __getitem__(self, key):
        """Ricci[i, j]"""
        return self._data.get(key, 0)  # default to 0 if missing



