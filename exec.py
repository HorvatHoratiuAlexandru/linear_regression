from linear_regression import linear_reg
import random

def f_x(x_1, x_2, x_3):
    return 2 + 5*x_1 - 9*x_2 + 0.5*x_3

obs_x = [[random.randint(1, 65), random.randint(1, 65), random.randint(1, 65)] for _ in range(150)]
obs_y = [f_x(obs[0], obs[1], obs[2]) for obs in obs_x]

print(linear_reg(obs_y, obs_x))