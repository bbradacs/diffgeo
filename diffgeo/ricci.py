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

def ricci_tensor(metric, riemann):
    dim = metric.dim

    def ricci(i, j):
        return sp.simplify(
            sum(riemann[k, i, k, j] for k in range(dim))
        )

    # Build dictionary instead of nested lists
    ricci_dict = {}
    for idx in product(range(dim), repeat=2):  # 2 indices: i, j
        val = ricci(*idx)
        if val != 0:  # optional: store only nonzero entries
            ricci_dict[idx] = val

    return Ricci(ricci_dict, dim)

