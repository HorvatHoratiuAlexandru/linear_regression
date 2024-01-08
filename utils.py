from typing import List

def dot_product(v_1: List[float], v_2: List[float]) -> float:
    return sum([v_i*v_j for v_i, v_j in zip(v_1,v_2)])

def vector_mean(vectors: List[List[float]]):
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))

def vector_sum(vectors: List[List[float]]) -> List[float]:
    assert vectors

    num_elements = len(vectors[0])

    return [sum(vector[i] for vector in vectors) for i in range(num_elements)]

def scalar_multiply(n: float, vec:List[float]) -> List[float]:
    return [n*v_i for v_i in vec]