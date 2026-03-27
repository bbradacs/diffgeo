from diffgeo import Tensor, d
from itertools import product   
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

def create_gamma_tensor(g_down, g_up):
    coords = g_down._coords
    dim = g_down.dim

    data = {}

    for k in range(dim):
        for i in range(dim):
            for j in range(dim):
                val = sum(
                    g_up[k,l] * (
                        d(g_down[l,i], coords[j]) +
                        d(g_down[l,j], coords[i]) -
                        d(g_down[i,j], coords[l])
                    )
                    for l in range(dim)
                ) / 2

                val = sp.simplify(val)
                if val != 0:
                    data[(k,i,j)] = val

    return Tensor(data, ['up','down','down'], dim, coords)


def create_riemann_tensor(gamma):
    coords = gamma._coords
    dim = gamma.dim

    data = {}

    for k, l, i, j in product(range(dim), repeat=4):
        val = (
            d(gamma[k,j,l], coords[i]) -
            d(gamma[k,i,l], coords[j])
            +
            sum(gamma[k,i,m] * gamma[m,j,l] for m in range(dim)) -
            sum(gamma[k,j,m] * gamma[m,i,l] for m in range(dim))
        )

        val = sp.simplify(val)
        if val != 0:
            data[(k,l,i,j)] = val

    return Tensor(data, ['up','down','down','down'], dim, coords)

def create_ricci_tensor(riemann):
    return riemann.contract(0, 2)

def create_scalar(g_up, ricci):
    tmp = g_up.contract_with(ricci, 0, 0)
    return tmp.contract(0, 0)

