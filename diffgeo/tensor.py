from itertools import product
import copy
import sympy as sp

# Model a Tensor
# 
# Components       dict
# Dimension        dim
# Coordinates      from Metric
# Index structure  (generic technique for contraction)

class Tensor:
    def __init__(self, data, indices, dim, coords=None):
        """
        data: dict mapping index tuples → values
        indices: list like ['up', 'down', 'down']
        dim: dimension of manifold
        coords: optional (for later use)
        """
        self._data = data
        self._indices = indices
        self._dim = dim
        self._coords = coords

    def __getitem__(self, key):
        return self._data.get(key, 0)

    @property
    def items(self):
        """Return a view of (index_tuple, value) pairs."""
        return self._data.items()
    
    @property
    def indices(self):
        return self._indices

    @property
    def rank(self):
        return len(self._indices)

    @property
    def dim(self):
        return self._dim
    
    def copy(self):
        """Deep copy of the tensor"""
        return Tensor(copy.deepcopy(self._data),
                      self._indices[:],
                      self._dim,
                      self._coords[:] if self._coords else None)
    

    def tensor_product(self, other):
        """Outer product: concatenate indices and multiply components"""
        new_indices = self._indices + other._indices
        new_data = {}
        for key1, val1 in self._data.items():
            for key2, val2 in other._data.items():
                new_key = key1 + key2
                new_data[new_key] = sp.simplify(val1 * val2)
        coords = self._coords or other._coords
        return Tensor(new_data, new_indices, self._dim, coords)

    def __mul__(self, other):
        """Tensor * Tensor => tensor product, Tensor * scalar => scale"""
        if isinstance(other, Tensor):
            return self.tensor_product(other)
        elif isinstance(other, (int, float, sp.Basic)):
            # scale all components by scalar
            new_data = {k: v*other for k,v in self.items}
            return Tensor(new_data, self._indices, self._dim, self._coords)
        else:
            raise TypeError(f"Cannot multiply Tensor by {type(other)}")

    def __rmul__(self, other):
        """scalar * Tensor => scale"""
        if isinstance(other, (int, float, sp.Basic)):
            return self * other  # delegate to __mul__
        else:
            raise TypeError(f"Cannot multiply {type(other)} by Tensor")
       
    def contract(self, i, j):
        if self._indices[i] == self._indices[j]:
            raise ValueError("Can only contract one up with one down index")

        new_indices = [idx for k, idx in enumerate(self._indices) if k not in (i, j)]
        new_data = {}

        for key, val in self._data.items():
            # sum over the contracted index
            if key[i] == key[j]:
                reduced_key = tuple(key[m] for m in range(len(key)) if m not in (i, j))
                new_data[reduced_key] = new_data.get(reduced_key, 0) + val

        return Tensor(new_data, new_indices, self._dim, self._coords)


    def contract_with(self, other, i, j):
        if self._indices[i] == other._indices[j]:
            raise ValueError("Must contract one up with one down")

        if self._dim != other._dim:
            raise ValueError("Dimension mismatch")

        new_indices = (
            [idx for k, idx in enumerate(self._indices) if k != i] +
            [idx for k, idx in enumerate(other._indices) if k != j]
        )

        new_data = {}

        # iterate over ALL index combinations of result tensor
        for result_key in product(range(self._dim), repeat=len(new_indices)):

            total = 0

            # sum over contracted index
            for k in range(self._dim):

                # reconstruct full keys for self and other
                key_self = []
                key_other = []

                idx_res = 0

                # rebuild self key
                for m in range(len(self._indices)):
                    if m == i:
                        key_self.append(k)
                    else:
                        key_self.append(result_key[idx_res])
                        idx_res += 1

                # rebuild other key
                for m in range(len(other._indices)):
                    if m == j:
                        key_other.append(k)
                    else:
                        key_other.append(result_key[idx_res])
                        idx_res += 1

                total += self[tuple(key_self)] * other[tuple(key_other)]

            if total != 0:
                new_data[result_key] = sp.simplify(total)

        coords = self._coords or other._coords
        return Tensor(new_data, new_indices, self._dim, coords)   
 