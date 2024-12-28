import random
import numpy as np
import warnings
import networkx as nx
import stopit
import time
import os
import csv
from constants import *

warnings.filterwarnings('ignore')
# Function to generate an initial solution
def initial_solution(n):
    path = np.arange(n)
    return path

# Function to calculate the cost of a solution
def get_cost(solution, distance_matrix):
    total_cost = np.sum(distance_matrix[solution[:-1], solution[1:]])
    total_cost += distance_matrix[solution[-1], solution[0]]  # Return to the starting city
    return total_cost

# Function to swap cities in the solution
def swap_cities(current_solution):
    new_path = np.copy(current_solution)
    swap_city_nodes = np.arange(1, len(new_path) - 1)

    i = random.choice(swap_city_nodes)
    j = random.choice(np.delete(swap_city_nodes, np.where(swap_city_nodes == i)))

    # Swap elements in the solution
    new_path[i], new_path[j] = new_path[j], new_path[i]

    return new_path

# Function for 2-opt local search
def two_opt(current_path, distance_matrix):
    n = len(current_path)
    best_path = current_path
    improved = True

    while improved:
        improved = False
        for i in range(1, n - 2):
            for j in range(i + 1, n):
                if j - i == 1:
                    continue  # Changes nothing, skip
                new_path = np.copy(current_path)
                new_path[i:j] = np.flip(new_path[i:j])
                new_cost = get_cost(new_path, distance_matrix)
                if new_cost < get_cost(best_path, distance_matrix):
                    best_path = new_path
                    improved = True

    return best_path

# Function to run the simulated annealing algorithm for TSP with dynamic adjustment
def TSP_with_dynamic_adjustment(distance_matrix, iterations, gamma, dynamic_coefficient):
    n = len(distance_matrix)
    initial_temperature = 1000  # Set your initial temperature
    temperature = initial_temperature

    current_path = initial_solution(n)
    current_cost = get_cost(current_path, distance_matrix)
    current_path = two_opt(current_path, distance_matrix)  # Apply 2-opt to the initial solution
    best_path, best_cost = current_path, current_cost

    cost_history = []  # List to store costs at each iteration
    temperature_history = []  # List to store temperatures at each iteration

    for i in range(iterations):
        # Swap cities
        new_path = swap_cities(current_path)

        # Calculate the cost of the new solution
        new_cost = get_cost(new_path, distance_matrix)

        # Log if this new one is the best seen so far
        if new_cost < best_cost:
            best_path, best_cost = new_path, new_cost

        # Store the current cost for analysis
        cost_history.append(current_cost)
        temperature_history.append(temperature)

        # Dynamic adjustment of acceptance probability
        dynamic_term = dynamic_coefficient * (best_cost - current_cost) / best_cost
        adjusted_probability = min(1, np.exp(-(new_cost - current_cost + dynamic_term) / temperature))

        # Stay with the new solution or transition from the current state?
        if adjusted_probability > random.uniform(0, 1):
            current_path, current_cost = new_path, new_cost

        # Update temperature
        temperature = cool_temp(gamma, temperature)

    # Add the cost of returning to the source node in best_cost
    best_cost += distance_matrix[best_path[-1], best_path[0]]

    # Add the return to the source node in best_path
    best_path = np.append(best_path, [best_path[0]])

    return best_cost, best_path

# Function to cool the temperature
def cool_temp(gamma, temp):
    return gamma * temp


def grid_search(distance_matrix, iterations_values, gamma_values, dynamic_coefficient_values):
    best_cost = float('inf')
    best_parameters = None
    best_path = None

    for iterations in iterations_values:
        for gamma in gamma_values:
            for dynamic_coefficient in dynamic_coefficient_values:
                final_cost, final_path = TSP_with_dynamic_adjustment(distance_matrix, iterations, gamma, dynamic_coefficient)

                if final_cost < best_cost:
                    best_cost = final_cost
                    best_parameters = (iterations, gamma, dynamic_coefficient)
                    best_path = final_path

    return best_parameters, best_cost, best_path

if __name__ == '__main__':
    input, exec_time = [], []

    file_arr = sorted([file for file in os.listdir(folder_dir) if file[0:4]=='tsp-'])
    file_arr.sort(key=lambda x:int(x.split('-')[2]))

    with open(fr'{output_dir}\SLS.csv', 'w+') as f:
        writer = csv.writer(f)

        writer.writerow(["37"])
        writer.writerow(["18278332", "13406436"])
        writer.writerow(["SLS"])
        for files in file_arr:
            path = folder_dir + '\\' + files
            try:
                distance_matrix = np.loadtxt(fr'{path}', skiprows=1, converters=float).astype(np.int64)
            except Exception as e:
                continue
            with stopit.ThreadingTimeout(600) as to_ctx_mgr:
                assert to_ctx_mgr.state == to_ctx_mgr.EXECUTING
                dynamic_coefficient_values = [0.001,0.0050,0.0090,0.01,0.06]
                iterations_values = [5000]
                gamma_values = [0.95,0.99,0.90]

                n = distance_matrix.shape[0]

                start_time = time.time()
                best_parameters, best_cost, best_path = grid_search(distance_matrix, iterations_values, gamma_values, dynamic_coefficient_values)
                end_time = time.time()
                execution_time = round(end_time - start_time,2)

                exec_time.append(execution_time)
                input.append(n)

                writer.writerow([f'{best_cost}', f'{execution_time}'])


            if to_ctx_mgr.state == to_ctx_mgr.EXECUTED:
                print(files, best_cost, execution_time)

            elif to_ctx_mgr.state == to_ctx_mgr.TIMED_OUT:
                print('TIMED OUT')
                print(files, best_cost, execution_time)








