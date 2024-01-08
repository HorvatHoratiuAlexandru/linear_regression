import random
from typing import List, Dict
from utils import dot_product

STD_EPOCH = 10000
STD_LEARNING_RATE = 0.0001

def linear_reg(obs_y: List[float], obs_x: List[List[float]], opts: Dict = dict()) -> List[float]:
    epochs = opts.get("epochs") if opts.get("epochs") else STD_EPOCH
    learning_rate = opts.get("learning_rate") if opts.get("learning_rate") else STD_LEARNING_RATE

    num_params = len(obs_x[0]) + 1  # add one for the constant
    coeficients = [random.uniform(-0.1, 0.1) for _ in range(num_params)]  # Initialize with small random values

    num_obs = len(obs_y)
    for epoch in range(epochs):
        for obs_i, obs_j in zip(obs_x, obs_y):
            gradient = sqerror_gradient([1] + obs_i, obs_j, coeficients)
            coeficients = [beta - (g * learning_rate) for g, beta in zip(gradient, coeficients)]
        print(coeficients)

    return coeficients

def predict_y(x: List[float], beta: List[float]) -> float:
    assert len(x) == len(beta)
    result = dot_product(x, beta)
    return result

def error(x: List[float], y: float, beta: List[float]) -> float:
    return predict_y(x, beta) - y

def sqerror_gradient(x: List[float], y: float, beta: List[float]) -> List[float]:
    err = error(x, y, beta)
    return [2 * err * x_i for x_i in x]  # list of partial derivatives
