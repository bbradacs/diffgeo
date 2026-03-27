from itertools import product
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
    def indices(self):
        return self._indices

    @property
    def rank(self):
        return len(self._indices)

    @property
    def dim(self):
        return self._dim
    

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
 