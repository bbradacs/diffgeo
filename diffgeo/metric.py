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

# Example:  metric = create_metric("r theta", lambda trig, r, theta: [[1, 0], [0, r**2]] )
def create_covariant_metric(coords_str, create_matrix_func):
    sp_coords = sp.symbols(coords_str)
    sp_matrix = sp.Matrix(create_matrix_func(sp, *sp_coords))
    return Metric(sp_coords, sp_matrix, role="g_down")

def create_contravariant_metric(coords_str, create_matrix_func):
    sp_coords = sp.symbols(coords_str)
    sp_matrix = sp.Matrix(create_matrix_func(sp, *sp_coords)).inv()
    return Metric(sp_coords, sp_matrix, role="g_up")

def create_metrics(coords_str, create_matrix_func):
    covariant = create_covariant_metric(coords_str, create_matrix_func)
    contravariant = covariant.inv()
    return covariant, contravariant

    
