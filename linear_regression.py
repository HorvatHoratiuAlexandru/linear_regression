import random


from typing import List, Dict
from utils import dot_product

STD_EPOCH = 150
STD_LEARNING_RATE = 0.1

def linear_reg(obs_y:List[float], obs_x: List[List[float]], opts: Dict = dict()) -> List[float]:
    epochs = opts.get("epochs") if opts.get("epochs") else STD_EPOCH
    learning_rate = opts.get("learning_rate") if opts.get("learning_rate") else STD_LEARNING_RATE

    num_params = len(obs_x[0]) + 1 #add one for the constant
    coeficients = [random.uniform(0,1) for _ in range(num_params)]

    num_obs = len(obs_y)
    for epoch in range(epochs):
        print("___")
        gradients = [sqerror_gradient([1] + obs_i, obs_j, coeficients) for obs_i, obs_j in zip(obs_x, obs_y)]
        grad_means = [sum([g[i] for g in gradients])/num_obs for i in range(len(gradients[0]))]
        #print(grad_means)
        #print(coeficients)
        coeficients = [beta - (g * learning_rate) for g, beta in zip(grad_means, coeficients)]
        #print(coeficients)

    return coeficients

def predict_y(x: List[float], beta: List[float]) -> float:
    assert len(x) == len(beta)

    return dot_product(x, beta)

def error(x:List[float], y: float, beta: List[float]) -> float:
    return predict_y(x, beta) - y

def sqerror_gradient(x:List[float], y: float, beta: List[float]):
    err = error(x, y, beta)
    return [2*err*x_i for x_i in x] # list of partial derivatives