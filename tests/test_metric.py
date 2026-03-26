from diffgeo import Metric, create_metrics
def test_metric() :

    metric = Metric("x y", [
        [1, 0],
        [0, 1]
    ])

    metric_inv = metric.inv()   

    assert metric[0, 0] == 1
    assert metric[0, 1] == 0
    assert metric[1, 0] == 0
    assert metric[1, 1] == 1

    assert metric_inv[0, 0] == 1
    assert metric_inv[0, 1] == 0
    assert metric_inv[1, 0] == 0
    assert metric_inv[1, 1] == 1

def test_metric_role() :

    metric = Metric("x y", [
        [1, 0],
        [0, 1]
    ], role="g_down")

    metric_inv = metric.inv()   

    assert metric.role == "g_down"
    assert metric_inv.role == "g_up"
    
def test_metric_inverse() :

    g = Metric("x y", [
        [2, 1],
        [1, 1]
    ])

    assert g[0, 0] == 2
    assert g[0, 1] == 1
    assert g[1, 0] == 1
    assert g[1, 1] == 1
    
    g_inv = g.inv()

    assert g_inv[0, 0] == 1
    assert g_inv[0, 1] == -1
    assert g_inv[1, 0] == -1
    assert g_inv[1, 1] == 2


    
def test_metric_dim() :
    minkowski_metric = Metric("t x1 x2 x3", [
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, -1]
    ])
    minkowski_metric_inv = minkowski_metric.inv()

    assert minkowski_metric.dim == 4
    assert minkowski_metric_inv.dim == 4

def test_create_metrics() :

    def create_func(trig, r, theta, phi):
        return [
            [r, 0, 0],
            [0, r, 0],
            [0, 0, r**2 * trig.sin(theta)**2]
        ]

    g_down, g_up = create_metrics("r theta phi", create_func)
    assert g_down.role == "g_down"
    assert g_up.role == "g_up"
    assert g_down[0,1] == 0
