from diffgeo import Tensor
from diffgeo import create_metric_tensors, create_gamma_tensor
from diffgeo import create_riemann_tensor, create_ricci_tensor, create_scalar_tensor
from diffgeo import create_einstein_tensor

def test_einstein_tensor_spherical():

    import sympy as sp

    def create_func(sp, t, r, theta, phi):

        f = sp.Function('f')
        h = sp.Function('h')
        sin = sp.sin

        return [
            [-f(r), 0, 0, 0],
            [0, h(r), 0, 0],
            [0, 0, r**2, 0],
            [0, 0, 0, r**2 * sin(theta)**2]
        ]

    g_down_t, g_up_t = create_metric_tensors("t r theta phi", create_func)

    gamma = create_gamma_tensor(g_down_t, g_up_t)
    riemann = create_riemann_tensor(gamma)
    ricci = create_ricci_tensor(riemann)
    scalar = create_scalar_tensor(g_up_t, ricci)

    einstein = create_einstein_tensor(g_down_t, ricci, scalar)

    print("\nRicci scalar:")
    for key, val in scalar.items:
        print(f"{key}: {val}")

    print("\nEinstein tensor G_{μν} (nonzero entries):")
    for key, val in einstein.items:
        print(f"{key}: {val}")

    # Take it one step further for Schwarzschild metric: f(r) = 1 - 2M/r, h(r) = 1/f(r)
    t,r,theta,phi = sp.symbols('t r theta phi')
    f = sp.Function('f')
    h = sp.Function('h')
    M = sp.Symbol('M')
    subs = {
        f(r): 1 - 2*M/r,
        h(r): 1 / (1 - 2*M/r)
    }

    for key, val in einstein.items:
        new_val = val.subs(subs)
        new_val1 = sp.together(new_val)   # combine fractions
        new_val2 = sp.simplify(new_val1)
        new_val3 = sp.factor(new_val2)
        print(key, new_val3)
