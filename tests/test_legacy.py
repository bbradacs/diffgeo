# I've included this file so that we can see the explicit algorithms
# used to create Christoffel symbols, Riemann tensor, Ricci tensor, and scalar curvature.
# This has all been replaced by tensor calculus defined in the class Tensor but its
# difficult to see the contractions in the general algorithm
#
# Note we do not store sparse entirie; i.e., the dictionaries contain (mostly) 0 values
import sympy as sp
from itertools import product
from diffgeo import d

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
    
def _create_covariant_metric(coords_str, create_matrix_func):
    sp_coords = sp.symbols(coords_str)
    sp_matrix = sp.Matrix(create_matrix_func(sp, *sp_coords))
    return Metric(sp_coords, sp_matrix)

def _create_contravariant_metric(coords_str, create_matrix_func):
    sp_coords = sp.symbols(coords_str)
    sp_matrix = sp.Matrix(create_matrix_func(sp, *sp_coords)).inv()
    return Metric(sp_coords, sp_matrix)

def _create_metrics(coords_str, create_matrix_func):
    covariant = _create_covariant_metric(coords_str, create_matrix_func)
    contravariant = _create_contravariant_metric(coords_str, create_matrix_func)
    return covariant, contravariant
    
def _create_gamma(g_down, g_up):
    coords = g_down.coords
    dim = g_down.dim
    
    gamma_dict = {}
    
    for k in range(dim):
        for i in range(dim):
            for j in range(dim):
                s = sum(
                    g_up[k, l] * (
                        d(g_down[l, i], coords[j]) +
                        d(g_down[l, j], coords[i]) -
                        d(g_down[i, j], coords[l])
                    )
                    for l in range(dim)
                )
                gamma_dict[(k, i, j)] = sp.simplify(s / 2)
    
    return gamma_dict
    
def _create_riemann(metric, Gamma):
    coords = metric.coords
    dim = metric.dim

    def R(k, l, i, j):
        return sp.simplify(
            # derivative terms
            d(Gamma[k, j, l], coords[i]) -
            d(Gamma[k, i, l], coords[j])
            +
            # quadratic terms
            sum(Gamma[k, i, m] * Gamma[m, j, l] for m in range(dim)) -
            sum(Gamma[k, j, m] * Gamma[m, i, l] for m in range(dim))
        )

    # Build dictionary
    riemann_dict = {}
    for idx in product(range(dim), repeat=4):  # 4 indices: k, l, i, j
        val = R(*idx)
        riemann_dict[idx] = val

    return riemann_dict

def _create_ricci(metric, riemann):
    dim = metric.dim

    def ricci(i, j):
        return sp.simplify(
            sum(riemann[k, i, k, j] for k in range(dim))
        )

    # Build dictionary instead of nested lists
    ricci_dict = {}
    for idx in product(range(dim), repeat=2):  # 2 indices: i, j
        val = ricci(*idx)
        ricci_dict[idx] = val

    return ricci_dict

def _create_scalar(g_up, Ricci):
    dim = g_up.dim

    return sp.simplify(
        sum(
            g_up[i, j] * Ricci[i, j]
            for i in range(dim)
            for j in range(dim)
        )
    )    

def test_legacy():
    # 2-sphere metric function
    def create_func(trig, theta, _):
        sin = trig.sin
        return [
            [1, 0],
            [0, sin(theta)**2]
        ]

    g_down, g_up = _create_metrics("theta phi", create_func)
    gamma   = _create_gamma(g_down, g_up)
    riemann = _create_riemann(g_down, gamma)
    ricci   = _create_ricci(g_down, riemann)
    scalar  = _create_scalar(g_up, ricci)

    print("\nRiemann tensor (nonzero entries):")
    for key, val in riemann.items():
        print(f"{key}: {val}")

    print("\nRicci tensor (nonzero entries):")
    for key, val in ricci.items():
        print(f"{key}: {val}")

    print("\nScalar curvature (nonzero entries):")
    print(f"scalar: {scalar}")

    # Assert scalar curvature
    assert sp.simplify(scalar - 2) == 0, f"Scalar curvature mismatch: {scalar}"

def test_metric() :

    metric = Metric("x y", [
        [1, 0],
        [0, 1]
    ])

    assert metric[0, 0] == 1
    assert metric[0, 1] == 0
    assert metric[1, 0] == 0
    assert metric[1, 1] == 1
    

def test_metric_dim() :
    minkowski_metric, minkowski_metric_inv = _create_metrics("t x1 x2 x3", lambda sp, t, x1, x2, x3: [
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, -1]
    ])

    assert minkowski_metric.dim == 4
    assert minkowski_metric_inv.dim == 4
