# 这是一个用于演示如何处理约束的演化算法的示例，并非是最优的解决方案。

import numpy as np
import matplotlib.pyplot as plt

def objective_function(x):
    return np.square(x)

def is_feasible(x):
    g = x - 5
    return g <= 0

def create_population(size, lower_bound, upper_bound):
    return np.random.uniform(lower_bound, upper_bound, size)

def tournament_selection(population, fitness, tournament_size):
    indices = np.random.choice(len(population), tournament_size)
    best_index = indices[np.argmin(fitness[indices])]
    return population[best_index]

def crossover(parent1, parent2):
    return (parent1 + parent2) / 2

def mutation(offspring, mutation_rate, lower_bound, upper_bound):
    if np.random.random() < mutation_rate:
        offspring = np.random.uniform(lower_bound, upper_bound)
    return offspring

def run_constrained_genetic_algorithm(population_size, generations, lower_bound, upper_bound,
                                      mutation_rate, tournament_size, plot_generation=10):
    population = create_population(population_size, lower_bound, upper_bound)

    for generation in range(generations):
        fitness = objective_function(population)

        new_population = []
        for _ in range(population_size):
            parent1 = tournament_selection(population, fitness, tournament_size)
            parent2 = tournament_selection(population, fitness, tournament_size)

            offspring = crossover(parent1, parent2)
            offspring = mutation(offspring, mutation_rate, lower_bound, upper_bound)

            new_population.append(offspring)

        population = np.array(new_population)

        if generation % plot_generation == 0:
            plot_population(population, generation)

    return population

def plot_population(population, generation):
    x = np.linspace(lower_bound, upper_bound, 1000)
    y = objective_function(x)
    plt.plot(x, y, label="Objective function")

    feasible = [ind for ind in population if is_feasible(ind)]
    infeasible = [ind for ind in population if not is_feasible(ind)]

    plt.scatter(feasible, objective_function(feasible), color="green", label=f"Feasible (Gen {generation})")
    plt.scatter(infeasible, objective_function(infeasible), color="red", label=f"Infeasible (Gen {generation})")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

# GA parameters
population_size = 100
generations = 100
lower_bound = -5
upper_bound = 10
mutation_rate = 0.1
tournament_size = 5

population = run_constrained_genetic_algorithm(population_size, generations, lower_bound, upper_bound,
                                               mutation_rate, tournament_size, plot_generation=10)
