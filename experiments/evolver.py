import heapq
import numpy as np
import uuid

from abc import ABCMeta, abstractmethod
from enum import Enum
from typing import Iterable

class Evolver(metaclass = ABCMeta):
    """An abstract class for evolving an object using a genetic algorithm."""

    def __init__(self):
        
        self._child_heap = []
        self._child_dict = {}
        heapq.heapify(self._child_heap)
        self._num_parents = 2
        self._parents = [self.init_child() for i in range(self._num_parents)]
        self._generation = 0    
        self._generation_priorities = []

    def add_child(self, child, priority):
        """
        
        Args:

        """
        cid = uuid.uuid1()
        self._child_dict[cid] = child
        entry = [priority, cid]
        
        if len(self._child_heap) >= self._num_parents:
            heapq.heapreplace(self._child_heap, entry)

        else:
            heapq.heappush(self._child_heap, entry)
        
        self._generation_priorities.append(priority)
        
    def spawn_child(self):
        """
        
        Args:

        Returns:

        """
        return self.mutate(self.crossover(self._parents[0], 
                                          self._parents[1]))   

    def update_parents(self):
        """"""
        parents_cid = heapq.nlargest(self._num_parents, self._child_heap)
        self._parents = []
        for priority, pcid in parents_cid:
            self._parents.append(self._child_dict[pcid])
        
        self._child_dict = {}
        self._child_heap = []
        heapq.heapify(self._child_heap)
        self._generation_priorities = []
        self._generation += 1

    def summarize_generation(self):
        return {
            'generation': self._generation,
            'mean': round(np.mean(self._generation_priorities), 2),
            'std': round(np.std(self._generation_priorities), 2)}

    @abstractmethod
    def init_child(self):
        """"""
        ...

    @abstractmethod
    def crossover(self, p1, p2):
        """
        
        Args:

        Returns:

        """
        ...

    @abstractmethod
    def mutate(self, p):
        """
        
        Args:

        """
        ...


class CrossoverType(Enum):
    """"""
    UNIFORM = 1 #

class MutationType(Enum):
    """"""
    FLIP_BIT = 1 #


class VectorEvolver(Evolver):
    """"""

    def __init__(self, 
                 size: int, 
                 crossover_type: CrossoverType, 
                 mutation_type: MutationType):
        """

        Args:

        """
        self._vec_size = size
        self.crossover_type = crossover_type
        self.mutation_type = mutation_type
        super().__init__()

    def init_child(self):
        """"""
        return np.random.randint(low=0, high=1, size=self._vec_size)

    def crossover(self, p1, p2):
        """
        
        Args:
            p1:
            p2:
                
        Returns:
            
        """
        c = np.copy(p1)

        if self.crossover_type == CrossoverType.UNIFORM:
            crossover_bits = np.random.rand(self._vec_size) < 0.5 
            c[crossover_bits] = p2[crossover_bits]
        
        return c

    def mutate(self, p):
        """
        
        Args:
            p:
        
        """
        
        if self.mutation_type == MutationType.FLIP_BIT:
            mutation_bits = np.random.rand(self._vec_size) < (1 / self._vec_size)
            p[mutation_bits] = 1 - p[mutation_bits]

        return p

class MatrixEvolver(VectorEvolver):
    """"""
    
    def __init__(self, 
                 sizes: Iterable[Iterable[int]],
                 crossover_type: CrossoverType,
                 mutation_type: MutationType):
        """
        Args:
            sizes:
            crossover_type:
            mutation_type:
        """
        self._matrix_sizes = sizes
        self._matrix_params = [np.product(s) for s in self._matrix_sizes]
        self._total_params = np.sum(self._matrix_params)
        super().__init__(self._total_params, crossover_type, mutation_type)

    def vec_to_matrices(self, vec):
        """
        
        Args:
            vec:
                
        Returns:
            
        """
        matrices = []
        idx = 0
        for s in self._matrix_params:
            m = np.zeros(s)
            m[:] = vec[idx : idx + s]
            matrices.append(m)
            idx += s

        return matrices

    def matrices_to_vec(self, matrices):
        """
        
        Args:
            matrices: 
                
        Returns:
            
        """

        vec = np.zeros(self._total_params)
        idx = 0

        for i, s in enumerate(self._matrix_params):
            vec[idx : idx + s] = matrices[i][:]
            idx += s

        return vec

    def spawn_child(self):
        """"""
        return self.vec_to_matrices(super().spawn_child())

    def add_child(self, child, priority):
        """"""
        return super().add_child(self.matrices_to_vec(child), priority)
