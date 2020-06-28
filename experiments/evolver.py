import heapq
import numpy as np

from abc import ABCMeta
from enum import Enum
from typing import Iterable

class Evolver(metaclass = ABCMeta):

    def __init__(self):
        self._child_heap = []
        heapify(self._child_heap)
        self._num_parents = 2
        self._parents = [[0, self.init_child()] for i in range(self._num_parents)]
        self._generation = 0    
        self._generation_priorities = []

    def add_child(self, child, priority):
        entry = [priority, child]
        
        if len(self._child_heap) >= self._num_parents:
            heapreaplce(self._child_heap, entry)

        else:
            heappush(self._child_heap, entry)
        
        self._generation_priorities.append(priority)
        
    def spawn_child(self):
        return self.mutate(self.crossover(self._parents[0][1], 
                                          self._parents[1][1]))   

    def update_parents(self):
        new_parents = nlargest(self_num_parents, self._child_heap)
        self._generation_priorities = []

    def summarize_generation(self):

        return {
            'generation': self.generation,
            'size': len(self._generation_priorities)
            'mean': np.mean(self._generation_priorities),
            'std': np.std(self._generation_priorities)}

    @abstractmethod
    def init_child(self):
        ...

    @abstractmethod
    def crossover(self, p1, p2):
        ...

    @abstractmethod
    def mutate(self, p):
        ...


class CrossoverType(Enum):
    UNIFORM = 1

class MutationType(Enum):
    FLIP_BIT = 1


class VectorEvolver(Evolver):

    def __init__(self, 
                 size: int, 
                 crossover_type: CrossoverType, 
                 mutation_type: MutationType):
        self._vec_size = size
        self.crossover_type = crossover_type
        self.mutation_type = mutation_type
        super().__init__()

    def init_child(self, vec_size = None):
        if vec_size is None:
            vec_size = self._vec_size

        return np.random.randint(low=0, high=1, size=self.vec_size)

    def crossover(self, p1, p2):
        c = np.copy(p1)

        if self.crossover_type == UNIFORM:
            crossover_bits = np.random.rand(self.size) < 0.5 
            c[crossover_bits] = p2[crossover_bits]
        
        return c

    def mutate(self, p):
        if self.mutation_type == FLIP_BIT:
            mutation_bits = np.random.rand(self.size) < (1 / self.size)
            p[mutation_bits] = 1 - p[mutation_bits]

        return p

class MatrixEvolver(VectorEvolver):

    def __init__(self, sizes: Iterable[Iterable[int]]):
        self.sizes = sizes
        self.mtrx_params = [np.product(s) for s in self.sizes]
        self.total_params = np.sum(mask_params)
        super()

    def init_child(self):
        super().init_child()

    def generate_matrices(self) -> Iterable[np.ndarray]:
        return [np.zeros(s) for s in self.sizes()]

    def matrices_to_vec(self, masks: Iterable[np.ndarray]) -> np.ndarray:
        vec = np.zeros(self.params)
        idx = 0

        for i, s in enumerate(self.sizes):
            params = self.mask_params[i]
            vec[idx : params] = 

        assert idx == self.num_params
        return

    def vec_to_masks(self):
        pass


        

if __name__ == "__main__":
    pass
