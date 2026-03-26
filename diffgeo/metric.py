import sympy as sp

class Metric:
    def __init__(self, coords, matrix, role = "g_down"):
        if(isinstance(coords, str)):
            self._sp_coords = sp.symbols(coords)
        else:
            self._sp_coords = coords   
        self._sp_matrix = sp.Matrix(matrix)
        self._dim = len(self._sp_coords)
        self._role = role

    def __getitem__(self, idx):
        return self._sp_matrix[idx]

    @property
    def coords(self):
        return self._sp_coords 

    @property
    def dim(self):
        return self._dim

    @property
    def role(self):
        return self._role

    def inv(self):
        role = "g_up" if self._role == "g_down" else "g_down"
        return Metric(self._sp_coords, self._sp_matrix.inv(), role=role)

# Example:  metric = create_metric("r theta", lambda r, theta: [[1, 0], [0, r**2]] )
def create_metric(coords_str, g_matrix_func):
    sp_coords = sp.symbols(coords_str)
    sp_matrix = g_matrix_func(*sp_coords)
    return Metric(sp_coords, sp_matrix)

    
