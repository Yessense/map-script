from typing import Optional, List,  Union

import numpy as np

np.set_printoptions(suppress=True)


class HDVector:
    """HD vector for hyperdimensional computing"""

    def __init__(self, vector: np.ndarray = None,
                 dimension=None,
                 random_state: Optional[np.random.RandomState] = None):
        """ Create HDVector from np.ndarray or generate new"""

        if vector is None:
            # generate new vector
            if dimension is None:
                dimension = 1000
            if random_state is None:
                random_state = np.random.RandomState()

            self.dimension = dimension
            self._vector = random_state.randn(dimension)
            self._vector /= np.linalg.norm(self._vector)
            self.make_unitary()
        else:
            # given vector
            self.dimension = len(vector)
            self._vector = vector

    @property
    def vector(self):
        return self._vector

    def cycle_shift(self, n: int = 1):
        """ Cycle permutation for sequence encoding"""
        return self ** n

    def __truediv__(self, other):
        return self * (other ** -1)

    def unbind(self, other):
        """ Unbind"""
        return self / other

    def __pow__(self, power: int, modulo=None):
        vector = np.fft.ifft(np.fft.fft(self._vector) ** power).real
        return HDVector(vector=vector)

    def __mul__(self, other):
        vector = np.fft.irfft(np.fft.rfft(self._vector) * np.fft.rfft(other.vector),
                              n=self.dimension)
        return HDVector(vector=vector)

    def __add__(self, other):
        vector = self._vector + other.vector
        vector /= np.linalg.norm(vector)
        return HDVector(vector=vector)

    def __repr__(self):
        return self._vector.__repr__()

    def make_unitary(self):
        fft_val = np.fft.fft(self._vector)
        fft_imag = fft_val.imag
        fft_real = fft_val.real
        fft_norms = np.sqrt(fft_imag ** 2 + fft_real ** 2)
        invalid = fft_norms <= 0.0
        fft_val[invalid] = 1.0
        fft_norms[invalid] = 1.0
        fft_unit = fft_val / fft_norms
        self._vector = np.array((np.fft.ifft(fft_unit, self.dimension)).real)

    def sim(self, other):
        return np.dot(self.vector, other.vector) / (np.linalg.norm(self.vector) * np.linalg.norm(other.vector))


def sim(v1: HDVector, v2: HDVector):
    """ Similarity measure of two vectors"""
    return np.dot(v1.vector, v2.vector) / (np.linalg.norm(v1.vector) * np.linalg.norm(v2.vector))


def bundle(*hd_vectors):
    """ Weighted sum of multiple vectors"""
    vector: HDVector
    vectors = [vector.vector for vector in hd_vectors]
    out = np.sum(vectors, axis=0)
    out /= np.linalg.norm(out)
    return HDVector(vector=out)


class ItemMemory():
    """ Store HDVectors """

    def __init__(self,
                 name: str = '',
                 dimension: int = 1000,
                 random_state: Optional[np.random.RandomState] = None,
                 init_vectors: Optional[List[str]] = None,
                 *args, **kwargs):

        if random_state is None:
            random_state = np.random.RandomState(1)
        if init_vectors is None:
            init_vectors = []

        self.random_state = random_state
        self.dimension: int = dimension
        self.init_vectors = init_vectors
        self.item_memory_name: str = name

        self.memory: np.ndarray = np.zeros((1, self.dimension))
        self._names: List[str] = []
        self.item_count: int = 0

        self.create_init_vectors()

        self._index = -1

    def create_init_vectors(self) -> None:
        """ Create init vectors f.e. shift, end """
        for name in self.init_vectors:
            self.__setitem__(name, self.generate())

    def generate(self) -> HDVector:
        return HDVector(dimension=self.dimension,
                        random_state=self.random_state)

    def __len__(self) -> int:
        return self.item_count

    def __setitem__(self, name: str, hd_vector: HDVector) -> None:
        """Add hypervector to a memory."""
        if name in self._names:
            raise NameError(f"Vector {name!r} already exists.")

        if not self.item_count:
            self.memory[self.item_count, :] = hd_vector.vector
        else:
            self.memory = np.append(self.memory, hd_vector.vector[np.newaxis], axis=0)

        self._names.append(name)
        self.item_count += 1

    def append_batch(self, names: List[str]) -> None:
        """Add batch of hypervectors."""

        new_memory = np.zeros((self.item_count + len(names), self.dimension))
        new_memory[:self.item_count] = self.memory

        for name in names:
            self._names.append(name)
            new_memory[self.item_count] = self.generate().vector
            self.item_count += 1

        self.memory = new_memory

    def distances(self, hd_vec_query: HDVector):
        return 1 - self.similarities(hd_vec_query)

    def similarities(self, hd_vec_query: HDVector):
        result = np.zeros(self.item_count)

        for i in range(self.item_count):
            result[i] = sim(HDVector(self.memory[i, :]), hd_vec_query)
        return result

    def search(self, hd_vec_query: HDVector,
               n_top: Optional[int] = 1,
               threshold: Optional[float] = None,
               names: bool = False) -> Union[None, int, str, List[int], List[str]]:
        if n_top is None or n_top > self.item_count:
            n_top = self.item_count

        similarities = self.similarities(hd_vec_query)
        most_similar = np.argsort(similarities)[-n_top:]
        thresholded = np.arange(len(similarities))

        if threshold:
            thresholded = thresholded[similarities > threshold]

        indices = np.intersect1d(most_similar, thresholded)

        if not len(indices):
            return None

        if names:
            out: List[str] = []
            for i in indices:
                out.append(self._names[i])
        else:
            out = indices

        if n_top == 1:
            return out[0]
        return out

    def __iter__(self):
        return self

    def __next__(self):
        self._index += 1
        if self._index >= len(self.memory):
            self._index = -1
            raise StopIteration
        else:
            return self._names[self._index], HDVector(self.memory[self._index])

    @property
    def names(self) -> List[str]:
        """Get names of stored entities."""
        return self._names

    @property
    def name(self) -> str:
        """Get name of ItemMemory."""
        return self.item_memory_name

    def get_name(self, index) -> str:
        """Get name of entity by its index."""

        return self._names[index]

    def get_vector(self, key: str) -> HDVector:
        if key in self.names:
            idx = self._names.index(key)
            vector = self.memory[idx, :].flatten()
            return HDVector(vector=vector)
        else:
            vector = self.generate()
            self.__setitem__(key, vector)
            return vector

    def __getitem__(self, key: str) -> Optional[HDVector]:
        if key not in self.names:
            return None
        else:
            idx = self._names.index(key)

            vector = self.memory[idx, :].flatten()
            return HDVector(vector=vector)

    @property
    def dim(self):
        """Get dim."""
        return self.dimension
