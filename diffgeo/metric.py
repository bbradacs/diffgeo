import sympy as sp

class Metric:
    def __init__(self, coords, g_matrix):
        self.coords = coords
        self.g = sp.Matrix(g_matrix)
        self.g_inv = self.g.inv()
        self.dim = len(coords)