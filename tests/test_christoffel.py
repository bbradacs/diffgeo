from diffgeo import Metric, Gamma, christoffel_symbols, christoffel_terms, create_metrics

def test_flat_space():

    g_down, g_up = create_metrics("x y", lambda *_ : [[1, 0], [0, 1]])

    gamma = christoffel_symbols(g_down, g_up)

    for k in range(2):
        for i in range(2):
            for j in range(2):
                assert gamma[k, i, j] == 0

def test_curved_space():
    def create_func(sp, r, theta, phi):
        return [
            [r, 0, 0],
            [0, r, 0],
            [0, 0, r**2 * sp.sin(theta)**2]
        ]

    g_down, g_up = create_metrics("r theta phi", create_func)
    gamma = christoffel_symbols(g_down, g_up)
    terms = christoffel_terms(g_down, gamma)
    #simplified = {k: sp.simplify(v) for k, v in terms.items()}
    #print(simplified)

