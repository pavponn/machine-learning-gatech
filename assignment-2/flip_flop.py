import time

import mlrose_hiive as ml_h
import numpy as np

from algorithms import run_ga_algorithm, run_sa_algorithm, run_rhc_algorithm, run_mimic_algorithm
from constants import SEED

problem_name = 'flip-flop'
# problem_sizes = [50, 200, 500]
problem_sizes = [30, 80, 200]
problem_small = ml_h.FlipFlopGenerator().generate(SEED, size=problem_sizes[0])
problem_medium = ml_h.FlipFlopGenerator().generate(SEED, size=problem_sizes[1])
problem_big = ml_h.FlipFlopGenerator().generate(SEED, size=problem_sizes[2])
problems = [problem_small, problem_medium, problem_big]

# problems = [problem_small]

MAX_ATTEMPTS = 50
ITERATIONS = 300
POPULATION_SIZE = 50


def run_ga_all():
    mutation_rates = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8]

    for i in range(len(problems)):
        print(f"Generating GA, {problem_sizes[i]}")
        run_ga_algorithm(problem=problems[i],
                         mutation_rates=mutation_rates, problem_name=f"{problem_name}-{problem_sizes[i]}",
                         max_attempts=MAX_ATTEMPTS, iterations=ITERATIONS, population_size=POPULATION_SIZE)


def run_sa_all():
    temperature_list = [0.1, 0.5, 1.0, 2.0, 5.0, 10]
    for i in range(len(problems)):
        print(f"Generating SA, {problem_sizes[i]}")
        run_sa_algorithm(problem=problems[i],
                         temperature_list=temperature_list, problem_name=f"{problem_name}-{problem_sizes[i]}",
                         max_attempts=MAX_ATTEMPTS, iterations=ITERATIONS)


def run_rhc_all():
    restart_list = [0]
    for i in range(len(problems)):
        print(f"Generating RHC, {problem_sizes[i]}")
        run_rhc_algorithm(problem=problems[i],
                          restart_list=restart_list, problem_name=f"{problem_name}-{problem_sizes[i]}",
                          max_attempts=MAX_ATTEMPTS, iterations=ITERATIONS)


def run_mimic_all():
    keep_percent_list = [0.1, 0.2, 0.4, 0.5, 0.7]
    for i in range(len(problems)):
        print(f"Generating MIMIC, {problem_sizes[i]}")
        run_mimic_algorithm(problem=problems[i],
                            keep_percent_list=keep_percent_list, problem_name=f"{problem_name}-{problem_sizes[i]}",
                            max_attempts=MAX_ATTEMPTS, iterations=ITERATIONS, population_size=250)


if __name__ == "__main__":
    print(f"START")
    start_time = time.time()
    run_ga_all()
    run_sa_all()
    run_rhc_all()
    run_mimic_all()
    end_time = time.time()
    print(f"FINISH")
    print(f"Total time: {end_time - start_time}, seconds")
