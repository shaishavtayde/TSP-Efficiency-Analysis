import networkx as nx
import csv
import numpy as np
import time
import sys
import stopit
import os
from constants import *


sys.setrecursionlimit(2000)
def calculate_total_distance(tour, distance_matrix):
    total_distance = 0
    for i in range(len(tour) - 1):
      total_distance += distance_matrix[tour[i], tour[i+1]]

    return total_distance

def tsp_2opt(tour, distance_matrix):
    n = len(tour)
    best_tour = tour.copy()
    best_distance = calculate_total_distance(tour, distance_matrix)
    curr_tour_dist = 0

    for i in range(1, n - 2):
        for j in range(i + 1, n-1):
            if j - i == 1:
                continue

            # Use numpy operations for reversing the tour segment
            tour[i:j] = np.flip(tour[i:j])
            new_distance = calculate_total_distance(tour, distance_matrix)

            if new_distance < best_distance:
                best_tour = tour.copy()
                best_distance = new_distance

    return best_tour, best_distance


def tsp_branch_and_bound_with_mst_optimized(distance_matrix, start):
    num_locations = distance_matrix.shape[0]
    visited = set()
    current_tour = []


    best_distance = float('inf')

    # Construct Minimum Spanning Tree (MST) of the graph
    mst = nx.minimum_spanning_tree(nx.from_numpy_array(distance_matrix))
    mst_neighbors = list(nx.dfs_preorder_nodes(mst, source=start)) + [start]
    best_tour = mst_neighbors

    # Initialize dynamic upper bound based on the initial solution
    dynamic_upper_bound = calculate_total_distance(mst_neighbors, distance_matrix)


    def bounding_function(current_tour, distance_matrix):
        # Update the dynamic upper bound based on the current best solution
        nonlocal dynamic_upper_bound
        dynamic_upper_bound = min(dynamic_upper_bound, calculate_total_distance(current_tour, distance_matrix))
        return dynamic_upper_bound

    def dfs(location, bound):
        nonlocal current_tour, best_tour, best_distance
        if location in visited:
          return
        visited.add(location)
        current_tour.append(location)
        if len(visited) == num_locations:
            distance = calculate_total_distance(current_tour, distance_matrix)
            distance += distance_matrix[current_tour[-1][start]]
            if distance < best_distance:
                best_distance = distance
                best_tour = current_tour.copy()
                bound = best_distance
            return

        curr_dist = calculate_total_distance(current_tour, distance_matrix)

        if bound <= curr_dist:
            # Prune the branch if the bounding function indicates it cannot lead to an optimal solution
            return

        for u,neighbor in mst.edges(location):
          if neighbor not in visited:
            dfs(neighbor,bound)
        visited.remove(location)
        current_tour.pop()

    dfs(start, dynamic_upper_bound)

    best_tour, best_distance = tsp_2opt(best_tour, distance_matrix)

    return best_tour, best_distance,mst


if __name__ == '__main__':
    input, exec_time = [], []
    file_arr = sorted([file for file in os.listdir(folder_dir) if file[0:4]=='tsp-'])
    file_arr.sort(key=lambda x:int(x.split('-')[2]))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(fr'{output_dir}\BnB-DFS.csv', 'w+') as f:
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
                best_tour, best_distance = np.array([]), float('inf')
                n = distance_matrix.shape[0]

                start_time = time.time()
                best_tour, best_distance,mst = tsp_branch_and_bound_with_mst_optimized(distance_matrix, 0)
                end_time = time.time()
                execution_time = round(end_time - start_time,2)

                exec_time.append(execution_time)
                input.append(n)

                writer.writerow([f'{best_distance}', f'{execution_time}'])


            if to_ctx_mgr.state == to_ctx_mgr.EXECUTED:
                print(files, best_distance, execution_time)

            elif to_ctx_mgr.state == to_ctx_mgr.TIMED_OUT:
                print('TIMED OUT')
                print(files, best_distance, execution_time)


