class Metric:
    def __init__(self, coords, g):
        self.coords = coords        # symbols (x, y, ...)
        self.g = g                  # matrix (sympy Matrix)
        self.g_inv = g.inv()