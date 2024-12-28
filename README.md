# Traveling Salesman Problem (TSP) Algorithm Benchmarking

## Overview

This project evaluates the efficiency of two distinct algorithms for solving the **Traveling Salesman Problem (TSP)**:
1. **Branch-and-Bound with Depth-First Search (BnB-DFS)**
2. **Stochastic Local Search (SLS)**

The goal was to gauge the performance of these algorithms over **100+ test cases**, ranging from small to large problem instances, and establish a **competition benchmark** to support our findings.

---

## Objectives
- **Algorithm Efficiency:** Measure the runtime and solution quality of BnB-DFS and SLS on diverse TSP problem instances.
- **Scalability:** Test how the algorithms perform as the size of the TSP graph increases.
- **Benchmark Creation:** Establish a competitive framework to compare algorithm efficiency and identify their strengths and limitations.

---

## Algorithms

### 1. **Branch-and-Bound with Depth-First Search (BnB-DFS)**
- **Approach:** 
  - Utilizes a depth-first search strategy combined with the branch-and-bound technique.
  - Prunes branches of the search tree that exceed a known lower bound, ensuring faster convergence.
- **Strengths:**
  - Guarantees an optimal solution.
  - Performs well on smaller graphs due to precise pruning.
- **Limitations:**
  - Computationally expensive for larger graphs.

### 2. **Stochastic Local Search (SLS)**
- **Approach:** 
  - Leverages probabilistic techniques and iterative improvement to explore the solution space.
  - Combines methods like simulated annealing and random restarts to escape local minima.
- **Strengths:**
  - Scales well with large graphs.
  - Capable of finding near-optimal solutions in less time.
- **Limitations:**
  - May not guarantee the optimal solution.
  - Performance can vary based on hyperparameter tuning.




Zip file contains 2 python files
1. BnB-DFS.py
2. SLS.py

Steps to run the files:
1. Install the dependencies from requirements.txt
2. Update the constants.py - folder_dir, output_dir
	folder_dir - path to directory containing input files [i.e. tsp-problem-1000-10000-100-5-1.txt]
	output_dir - path to directory where Results should be created
3. Run both python files.
	
