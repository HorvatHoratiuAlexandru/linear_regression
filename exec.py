from linear_regression import linear_reg
import random

def f_x(x_1):
    return 2 + 5*x_1

obs_x = [[random.randint(1, 65)] for _ in range(150)]
obs_y = [f_x(obs[0]) for obs in obs_x]

print(linear_reg(obs_y, obs_x))