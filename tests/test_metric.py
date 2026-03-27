from diffgeo import Metric, create_metrics
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
    minkowski_metric, minkowski_metric_inv = create_metrics("t x1 x2 x3", lambda sp, t, x1, x2, x3: [
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, -1]
    ])

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
    assert g_down[0,1] == 0
    
