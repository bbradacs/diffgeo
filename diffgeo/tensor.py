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
    
