import sympy as sp
from diffgeo import Metric, create_metric

def test_metric() :

    metric = Metric("x y", [
        [1, 0],
        [0, 1]
    ])

    assert metric[0, 0] == 1
    assert metric[0, 1] == 0
    assert metric[1, 0] == 0
    assert metric[1, 1] == 1

    assert metric.inv[0, 0] == 1
    assert metric.inv[0, 1] == 0
    assert metric.inv[1, 0] == 0
    assert metric.inv[1, 1] == 1
    
def test_metric_inverse() :

    metric = Metric("x y", [
        [2, 1],
        [1, 1]
    ])

    assert metric[0, 0] == 2
    assert metric[0, 1] == 1
    assert metric[1, 0] == 1
    assert metric[1, 1] == 1

    assert metric.inv[0, 0] == 1
    assert metric.inv[0, 1] == -1
    assert metric.inv[1, 0] == -1
    assert metric.inv[1, 1] == 2

    assert metric.inv.inv[0, 0] == 2
    assert metric.inv.inv[0, 1] == 1
    assert metric.inv.inv[1, 0] == 1
    assert metric.inv.inv[1, 1] == 1
    
def test_metric_dim() :
    minkowski_metric = Metric("t x1 x2 x3", [
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, -1]
    ])

    assert minkowski_metric.dim == 4
    assert minkowski_metric.inv.dim == 4

def test_create_matrix() :
    import sympy as sp

    def create_func(r, theta, phi):
        return [
            [r, 0, 0],
            [0, r, 0],
            [0, 0, r**2 * sp.sin(theta)**2]
        ]

    metric = create_metric("r theta phi", create_func)

    assert metric[0,1] == 0
