from .tensor import Tensor
import sympy as sp
import numpy as np

class TensorEvaluator:
    def __init__(self, tensor):
        """
        tensor: symbolic Tensor with coords defined
        """
        if tensor._coords is None:
            raise ValueError("Tensor must define coords for evaluation")

        self.tensor = tensor
        self.variables = tuple(tensor._coords)

        # Precompile components using NumPy backend
        self._compiled = {}
        for key, expr in tensor.items:
            # 'numpy' backend handles floats, np scalars, and arrays
            func = sp.lambdify(self.variables, expr, modules='numpy')
            self._compiled[key] = func

    def at(self, *args, **kwargs):
        """
        Evaluate tensor at a point or array of points.
        Supports:
            eval_T.at(1.0, np.pi/4)                       # single point
            eval_T.at(r=np.array([1,2,3]), theta=np.pi/4) # vectorized
        Returns a Tensor with values as floats or NumPy arrays.
        """
        if args and kwargs:
            raise ValueError("Use either positional or keyword arguments, not both")

        # Keyword argument support
        if kwargs:
            try:
                args = tuple(kwargs[var.name] for var in self.variables)
            except KeyError as e:
                raise ValueError(f"Missing value for coordinate: {e}")

        if len(args) != len(self.variables):
            raise ValueError(f"Expected {len(self.variables)} values")

        # Convert inputs to arrays for broadcasting
        args = tuple(np.asarray(a) for a in args)
        new_data = {}

        for key, func in self._compiled.items():
            val = func(*args)

            # Convert 0-dim arrays to scalar
            if isinstance(val, np.ndarray) and val.shape == ():
                val = float(val)
            elif isinstance(val, np.generic):
                val = float(val)

            # Skip exact zeros
            if isinstance(val, float) and abs(val) < 1e-12:
                continue

            new_data[key] = val

        return Tensor(new_data, self.tensor.indices, self.tensor.dim, self.tensor._coords)
    