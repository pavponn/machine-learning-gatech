import os
import time

import numpy as np
import pandas as pd

import mlrose_hiive as ml_h
import multiprocessing
from functools import partial
from constants import SEEDS


def run_ga_with_seed(seed_index, problem, mutation_rates, problem_name, iterations, max_attempts, population_size):
    ga = ml_h.GARunner(problem=problem,
                       experiment_name=f"ga-{problem_name}-seed-{seed_index}",
                       output_directory=f"outputs/{problem_name}",
                       seed=SEEDS[seed_index],
                       iteration_list=[iterations],
                       population_sizes=[population_size],
                       max_attempts=max_attempts,
                       mutation_rates=mutation_rates)
    ga.run()


def run_ga_algorithm(problem, mutation_rates, problem_name, iterations, max_attempts, population_size):
    pool = multiprocessing.Pool()

    partial_run_ga_with_seed = partial(run_ga_with_seed, problem=problem,
                                       mutation_rates=mutation_rates, problem_name=problem_name, iterations=iterations,
                                       max_attempts=max_attempts, population_size=population_size)

    pool.map(partial_run_ga_with_seed, range(len(SEEDS)))

    pool.close()
    pool.join()


def run_sa_with_seed(seed_index, problem, temperature_list, problem_name, iterations, max_attempts):
    sa = ml_h.SARunner(problem=problem,
                       experiment_name=f"sa-{problem_name}-seed-{seed_index}",
                       output_directory=f"outputs/{problem_name}",
                       seed=SEEDS[seed_index],
                       iteration_list=[iterations],
                       max_attempts=max_attempts,
                       temperature_list=temperature_list,
                       decay_list=[ml_h.GeomDecay])
    sa.run()


def run_sa_algorithm(problem, temperature_list, problem_name, iterations, max_attempts):
    pool = multiprocessing.Pool()

    partial_run_sa_with_seed = partial(run_sa_with_seed, problem=problem,
                                       temperature_list=temperature_list, problem_name=problem_name,
                                       iterations=iterations,
                                       max_attempts=max_attempts)

    pool.map(partial_run_sa_with_seed, range(len(SEEDS)))

    pool.close()
    pool.join()


def run_rhc_with_seed(seed_index, problem, restart_list, problem_name, iterations, max_attempts):
    sa = ml_h.RHCRunner(problem=problem,
                        experiment_name=f"rhc-{problem_name}-seed-{seed_index}",
                        output_directory=f"outputs/{problem_name}",
                        seed=SEEDS[seed_index],
                        iteration_list=[iterations],
                        max_attempts=max_attempts,
                        restart_list=restart_list)
    sa.run()


def run_rhc_algorithm(problem, restart_list, problem_name, iterations, max_attempts):
    pool = multiprocessing.Pool()

    partial_run_rhc_with_seed = partial(run_rhc_with_seed, problem=problem,
                                        restart_list=restart_list, problem_name=problem_name, iterations=iterations,
                                        max_attempts=max_attempts)

    pool.map(partial_run_rhc_with_seed, range(len(SEEDS)))

    pool.close()
    pool.join()


def run_mimic_with_seed(seed_index, problem, keep_percent_list, problem_name, iterations, max_attempts,
                        population_size):
    mimic = ml_h.MIMICRunner(problem=problem,
                             experiment_name=f"mimic-{problem_name}-seed-{seed_index}",
                             output_directory=f"outputs/{problem_name}",
                             seed=SEEDS[seed_index],
                             iteration_list=[iterations],
                             population_sizes=[population_size],
                             max_attempts=max_attempts,
                             use_fast_mimic=True,
                             keep_percent_list=keep_percent_list)
    mimic.run()


def run_mimic_algorithm(problem, keep_percent_list, problem_name, iterations, max_attempts, population_size):
    problem.set_mimic_fast_mode(True)
    pool = multiprocessing.Pool()

    partial_run_rhc_with_seed = partial(run_mimic_with_seed, problem=problem,
                                        keep_percent_list=keep_percent_list, problem_name=problem_name,
                                        iterations=iterations,
                                        max_attempts=max_attempts, population_size=population_size)

    pool.map(partial_run_rhc_with_seed, range(len(SEEDS)))

    pool.close()
    pool.join()