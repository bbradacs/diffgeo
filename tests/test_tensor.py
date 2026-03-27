from diffgeo import Tensor

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
    