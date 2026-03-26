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
    """
    Contract index i with index j.
    Requires one 'up' and one 'down'.
    """
    if self._indices[i] == self._indices[j]:
        raise ValueError("Can only contract one up with one down index")

    new_indices = [
        idx for k, idx in enumerate(self._indices)
        if k not in (i, j)
    ]

    new_data = {}

    for key in self._data:
        for k in range(self._dim):
            new_key = tuple(
                key[m] if m not in (i, j) else None
                for m in range(len(key))
            )

            # replace contracted indices with k
            full_key = list(key)
            full_key[i] = k
            full_key[j] = k

            val = self._data.get(tuple(full_key), 0)

            reduced_key = tuple(
                full_key[m] for m in range(len(full_key))
                if m not in (i, j)
            )

            new_data[reduced_key] = new_data.get(reduced_key, 0) + val

    return Tensor(new_data, new_indices, self._dim)