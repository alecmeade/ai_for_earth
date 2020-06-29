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
        self._parents = nlargest(self_num_parents, self._child_heap)
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

    def init_child(self):
        return np.random.randint(low=0, high=1, size=self._vec_size)

    def crossover(self, p1, p2):
        c = np.copy(p1)

        if self.crossover_type == UNIFORM:
            crossover_bits = np.random.rand(self._vec_size) < 0.5 
            c[crossover_bits] = p2[crossover_bits]
        
        return c

    def mutate(self, p):
        if self.mutation_type == FLIP_BIT:
            mutation_bits = np.random.rand(self.size) < (1 / self._vec_size)
            p[mutation_bits] = 1 - p[mutation_bits]

        return p

class MatrixEvolver(VectorEvolver):
    def __init__(self, 
                 sizes: Iterable[Iterable[int]],
                 crossover_type: CrossoverType,
                 mutation_type: MutationType):
        self._matrix_sizes = sizes
        self._matrix_params = [np.product(s) for s in self.sizes]
        self._total_params = np.sum(self._matrix_params)
        super().__init__(self._total_params, crossover_type, mutation_type)

    def vec_to_matrices(self, vec):
        matrices = []
        idx = 0
        for s in self._matrix_sizes:
            m = np.zeros(s)
            m[:] = vec[idx : idx + s]
            matrices.append(m)
            idx += s

        return matrices

    def matrices_to_vec(self, matrices):
        vec = np.zeros(self._total_params)
        idx = 0

        for i, s in enumerate(self._matrix_sizes):
            vec[idx : idx + s] = matrices[i][:]
            idx += s

        return vec

    def spawn_child(self):
        return self.vec_to_matrices(super().spawn_child())

    def add_child(self, child, priority):
        return super().add_child(self.matrices_to_vec(child), priority)


if __name__ == "__main__":
    v = VectorEvolver(10, CrossoverType.UNIFORM, MutationType.FLIP_BIT)
    v.summarize_generation()
    
    for j in range(0, 3):
        for i in range(0, 10): 
            c = v.spawn_child()
            v.add_child(c, np.random.rand())
        print(v.summarize_generation())
        v.update_parents()

