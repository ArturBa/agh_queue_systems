from restaurant import *
from probability import p_local, p_online

from geneticalgorithm2 import geneticalgorithm2 as ga
import numpy as np


def cost_function(input):
    restaurant = RestaurantSystem()
    restaurant.P(Orders.Online, np.matrix(input['p'][0]))
    restaurant.P(Orders.Local, np.matrix(input['p'][1]))
    restaurant._lambda(Orders.Local, input['lambda'][0])
    restaurant._lambda(Orders.Online, input['lambda'][0])

    restaurant.Waiter(input['waiter'][0], input['waiter'][1])
    restaurant.OnlineBuffer(input['onlineBuffer'])
    restaurant.SoupChef(input['soupChef'])
    restaurant.MainChef(input['mainChef'])
    restaurant.DesserChef(input['desserChef'])
    restaurant.Barista(input['barista'])
    restaurant.Waiter2(input['waiter2'][0], input['waiter2'][1])
    restaurant.Delivery(input['delivery'][0], input['delivery'][1])
    restaurant.Cashier(input['cashier'][0], input['cashier'][1])

    # For system -1 since they iteration start on 1 (and we need input on probability matrix as 0)
    return sum([sum([input['cost'][order][system-1] * restaurant.K_IR(system, order) for order in Orders]) for system in Systems]) +\
        sum([restaurant.M_I(system) * input['costFree'][system-1] for system in Systems])

costQueue = [
    # Waiter OnlineBuffer ChefSoup ChefMain ChefDesser Barista Waiter2 Delivery Cashier 
    [ 0,        6,          8,         9,       4,         0,   0,          19,     0 ], # Online
    [ 2,        0,          8,         9,       4,         3,   3,           0,    17 ], # Local
]

costFreeChannel = \
    [ 2,        0,          8,         9,       4,         3,   3,          17,    17 ]
    # Waiter OnlineBuffer ChefSoup ChefMain ChefDesser Barista Waiter2 Delivery Cashier 

def generic_cost_function(X):
    input = {
        'p': [p_online, p_local],
        'lambda': [150, 150],
        'waiter': [100, X[0]],
        'onlineBuffer': 300,
        'soupChef': [250, 210],
        'mainChef': [250, 210],
        'desserChef': [250, 210],
        'barista': [350],
        'waiter2': [210, X[1]],
        'delivery': [55, X[2]],
        'cashier': [120, X[3]],
        'cost': costQueue,
        'costFree': costFreeChannel
    }
    return cost_function(input)

varbound = np.array([[2, 10]]*4)
model = ga(function=generic_cost_function, dimension=4, variable_type='int', variable_boundaries=varbound,
           algorithm_parameters={'max_num_iteration': 100,
                                       'population_size': 10,
                                       'mutation_probability': 0.1,
                                       'elit_ratio': 0.01,
                                       'crossover_probability': 0.5,
                                       'parents_portion': 0.3,
                                       'max_iteration_without_improv': None})
model.run()
