import sympy as sp

class Metric:

    # coordsStr example "x y" or "r theta" or "t x1 x2 x3"
    # g_matrix example [[1, 0], [0, 1]] for flat space in 2D
    def __init__(self, coords_str, g_matrix):
        self._coords_str = coords_str
        self._coords = sp.symbols(coords_str)  # returns tuple
        self._g = sp.Matrix(g_matrix)
        self._dim = len(self._coords)
        self._inv = None  # lazy

    def __getitem__(self, idx):
        return self._g[idx]

    @property
    def coords(self):
        return self._coords 

    @property
    def dim(self):
        return self._dim

    @property
    def inv(self):
        if self._inv is None:
            self._build_inverse()
        return self._inv

    def _build_inverse(self):
        g_inv_matrix = self._g.inv()

        inv_metric = Metric(self._coords_str, g_inv_matrix)

        # Link them together
        inv_metric._inv = self
        self._inv = inv_metric


# Example:  metric = create_metric("r theta", lambda r, theta: [[1, 0], [0, r**2]] )
def create_metric(coords_str, g_matrix_func):
    """
    coords_str : string like "r theta"
    g_matrix_func : function(symbols...) -> nested list representing metric
    """
    # 1. Create symbols
    coords = sp.symbols(coords_str)

    # 2. Build the matrix using the symbols
    g_matrix = g_matrix_func(*coords)

    # 3. Construct Metric
    return Metric(coords_str, g_matrix)

    
