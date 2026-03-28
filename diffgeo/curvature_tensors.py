import sympy as sp
from diffgeo import Tensor
from .derivatives import d
from itertools import product   

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

def create_scalar_tensor(g_up, ricci):
    # contract first index
    tmp = g_up.contract_with(ricci, 0, 0)
    # contract remaining indices
    scalar = tmp.contract(0, 1)
    return scalar

def create_einstein_tensor(g_down, ricci, scalar):
    dim = g_down.dim
    coords = g_down._coords

    # extract scalar value (rank-0 tensor)
    scalar_val = next(iter(scalar.items))[1]

    data = {}

    for i in range(dim):
        for j in range(dim):
            val = ricci[i, j] - sp.Rational(1, 2) * g_down[i, j] * scalar_val
            val = sp.simplify(val)
            if val != 0:
                data[(i, j)] = val

    return Tensor(data, ['down', 'down'], dim, coords)
