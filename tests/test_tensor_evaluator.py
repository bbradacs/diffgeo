import sympy as sp
from diffgeo import Tensor
from diffgeo import TensorEvaluator


def test_tensor_evaluator():
    # Symbolic tensor
    x, y = sp.symbols('x y')
    data = {(0,0): x**2 + y, (0,1): x*y, (1,0): y**2, (1,1): x + y}
    T = Tensor(data, ['up','down'], dim=2, coords=[x,y])
    evaluator = TensorEvaluator(T)
    # Evaluate at a point
    Teval = evaluator.at(x=2, y=3)
    assert(Teval[(0,0)] == 2**2 + 3)
    assert(Teval[(0,1)] == 2*3)
    assert(Teval[(1,0)] == 3**2)
    assert(Teval[(1,1)] == 2 + 3)