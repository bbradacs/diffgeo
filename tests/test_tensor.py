import sympy as sp
from diffgeo import Tensor
from diffgeo import create_metric_tensors


def test_metric_inverse_gives_identity():
    # Simple 2D polar metric
    # g = [[1, 0],
    #      [0, r^2]]
    def polar_metric(_, r, theta):
        return [
            [1, 0],
            [0, r**2]
        ]

    g_down, g_up = create_metric_tensors("r theta", polar_metric)

    # Contract: g_down(i,k) with g_up(k,j)
    identity = g_down.contract_with(g_up, 1, 0)

    # Should be δ^i_j
    dim = g_down.dim

    for i in range(dim):
        for j in range(dim):
            val = identity[(i, j)]

            if i == j:
                # Should simplify to 1
                assert sp.simplify(val - 1) == 0
            else:
                # Should simplify to 0
                assert sp.simplify(val) == 0

def test_tensor_contraction():
    # 2x2 tensor with one up and one down index
    data = {(0,0): 1, (0,1): 2, (1,0): 3, (1,1): 4}
    T = Tensor(data, ['up','down'], dim=2)
    # Contract over up and down index
    C = T.contract(0,1)
    # Contracted tensor is scalar: sum of diagonal elements (0,0)+(1,1)
    assert(C[( )] == 1 + 4)

def test_tensor_product():
    A = Tensor({(0,): 2, (1,): 3}, ['up'], 2)
    B = Tensor({(0,): 5, (1,): 7}, ['down'], 2)

    C = A * B  # tensor product
    assert C.rank == 2
    assert C[(0,0)] == 10
    assert C[(0,1)] == 14
    assert C[(1,0)] == 15
    assert C[(1,1)] == 21

    print("Tensor product works perfectly!")

def test_tensor_bilinearity():
    A = Tensor({(0,): 2, (1,): 3}, ['up'], 2)
    B = Tensor({(0,): 5, (1,): 7}, ['down'], 2)

    # Tensor-tensor product
    C = A * B
    assert C.rank == 2
    assert C[(0,0)] == 10
    assert C[(1,1)] == 21

    # Scalar * Tensor
    D = 3 * A
    assert D[(0,)] == 6
    assert D[(1,)] == 9

    # Tensor * Scalar
    E = B * 2
    assert E[(0,)] == 10
    assert E[(1,)] == 14

    print("Tensor bilinearity works perfectly!")
