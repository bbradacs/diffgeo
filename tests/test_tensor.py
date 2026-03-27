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

def test_vector_to_covector():
    dim = 3
    coords = ['x', 'y', 'z']
    
    # Define a simple diagonal metric g_down
    g_data = {(0,0):1, (1,1):2, (2,2):3}
    g_down = Tensor(g_data, ['down','down'], dim, coords)
    
    # Vector with components (v^0, v^1, v^2)
    v_data = {(0,):1, (1,):2, (2,):3}
    v = Tensor(v_data, ['up'], dim, coords)
    
    # Contract vector with metric down: v_i = g_ij v^j
    covector = g_down.contract_with(v, 1, 0)
    
    # Expected result: v_i = [1*1, 2*2, 3*3] = [1,4,9]
    expected_data = {(0,):1, (1,):4, (2,):9}
    
    for key, val in expected_data.items():
        assert sp.simplify(covector[key] - val) == 0 
    
    assert covector.indices == ['down']

def test_covector_to_vector():
    dim = 3
    coords = ['x', 'y', 'z']
    
    # Define a simple inverse metric g_up
    g_data = {(0,0):1, (1,1):1/2, (2,2):1/3}
    g_up = Tensor(g_data, ['up','up'], dim, coords)
    
    # Covector with components (w_0, w_1, w_2)
    w_data = {(0,):1, (1,):4, (2,):9}
    w = Tensor(w_data, ['down'], dim, coords)
    
    # Contract covector with inverse metric: w^i = g^ij w_j
    vector = g_up.contract_with(w, 0, 0)
    
    # Expected result: w^i = [1*1, 4*(1/2), 9*(1/3)] = [1,2,3]
    expected_data = {(0,):1, (1,):2, (2,):3}
    
    for key, val in expected_data.items():
        assert sp.simplify(vector[key] - val) == 0
    
    assert vector.indices == ['up']

def test_raise_lower_raise_vector():
    dim = 3
    coords = ['x', 'y', 'z']

    # Define a simple diagonal metric g_down and its inverse g_up
    g_down_data = {(0,0):1, (1,1):2, (2,2):3}
    g_up_data   = {(0,0):1, (1,1):1/2, (2,2):1/3}

    g_down = Tensor(g_down_data, ['down','down'], dim, coords)
    g_up   = Tensor(g_up_data,   ['up','up'],   dim, coords)

    # Original vector v^i
    v_data = {(0,):1, (1,):2, (2,):3}
    v = Tensor(v_data, ['up'], dim, coords)

    # Lower the index: v_i = g_ij v^j
    covector = g_down.contract_with(v, 1, 0)

    # Raise the index back: v^i_recovered = g^ij v_j
    v_recovered = g_up.contract_with(covector, 0, 0)

    # Check values
    for key in v_data.keys():
        assert sp.simplify(v_recovered[key] - v[key]) == 0

    # Check index type
    assert v_recovered.indices == ['up']
    
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
