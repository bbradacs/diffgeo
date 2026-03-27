from diffgeo import Tensor
import sympy as sp

def create_metric_tensors(coords_str, create_matrix_func):
    sp_coords = sp.symbols(coords_str)
    matrix = sp.Matrix(create_matrix_func(sp, *sp_coords))
    dim = len(sp_coords)

    g_down_data = {
        (i,j): matrix[i,j]
        for i in range(dim)
        for j in range(dim)
    }

    g_up_matrix = matrix.inv()
    g_up_data = {
        (i,j): g_up_matrix[i,j]
        for i in range(dim)
        for j in range(dim)
    }

    g_down = Tensor(g_down_data, ['down','down'], dim, coords=sp_coords)
    g_up   = Tensor(g_up_data,   ['up','up'],     dim, coords=sp_coords)

    return g_down, g_up
