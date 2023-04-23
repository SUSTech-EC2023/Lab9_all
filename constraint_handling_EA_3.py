# 这是一个用于演示如何处理约束的演化算法的示例，并非是最优的解决方案。

import numpy as np
import matplotlib.pyplot as plt

# Define the objective function and constraints
def objective_function(x):
    return (x[0] - 2)**2 + (x[1] - 3)**2

# Constraints
def constraint1(x):
    return (x[0] - 1)**2 + (x[1] - 1)**2 - 1

def constraint2(x):
    return (x[0] - 4)**2 + (x[1] - 4)**2 - 1

def constraint3(x):
    return (x[0] - 1)**2 + (x[1] - 5)**2 - 1

def constraint_function(x):
    return [constraint1(x), constraint2(x), constraint3(x)]

def evaluate_population(population):
    fitness = np.array([objective_function(individual) for individual in population])
    constraint_violations = np.array([constraint_function(individual) for individual in population])
    is_feasible = np.any(constraint_violations <= 0, axis=1)
    return fitness, is_feasible, constraint_violations


# Define the main evolutionary algorithm functions
def create_initial_population(pop_size):
    return np.random.uniform(-5, 5, size=(pop_size, 2))

def crossover(parent1, parent2, crossover_rate):
    if np.random.rand() < crossover_rate:
        alpha = np.random.rand()
        offspring = alpha * parent1 + (1 - alpha) * parent2
    else:
        offspring = parent1.copy()

    return offspring

def mutation(individual, mutation_rate):
    if np.random.rand() < mutation_rate:
        individual = individual + np.random.normal(0, 0.5, size=2)

    return individual


def tournament_selection(population, fitness, is_feasible, constraint_violations, tournament_size):
    indices = np.random.choice(len(population), size=tournament_size)
    feasible_indices = np.where(is_feasible[indices])[0]
    infeasible_indices = np.where(~is_feasible[indices])[0]

    if len(feasible_indices) > 0:
        best_index = indices[feasible_indices[np.argmin(fitness[indices[feasible_indices]])]]
    else:
        constraint_violations = np.sum(np.where(constraint_violations > 0, constraint_violations, 0), axis=1)
        best_index = indices[infeasible_indices[np.argmin(constraint_violations[indices[infeasible_indices]])]]

    return population[best_index]


def plot_population(population, is_feasible):
    # Create a meshgrid that covers the entire search space
    x1, x2 = np.meshgrid(np.linspace(-5, 5, 200), np.linspace(-5, 5, 200))
    grid_points = np.c_[x1.ravel(), x2.ravel()]
    
    # Evaluate the constraints for each point in the meshgrid
    constraint_values = np.array([constraint_function(point) for point in grid_points])
    is_grid_feasible = np.any(constraint_values <= 0, axis=1)
    
    # Reshape the feasibility array to match the shape of the meshgrid
    is_grid_feasible = is_grid_feasible.reshape(x1.shape)

    # Plot the feasible and infeasible areas using a contour plot
    plt.contourf(x1, x2, is_grid_feasible, cmap='RdYlBu_r', alpha=0.5)

    # Plot the feasible and infeasible solutions in the population
    plt.scatter(population[is_feasible, 0], population[is_feasible, 1], color='blue', label='Feasible')
    plt.scatter(population[~is_feasible, 0], population[~is_feasible, 1], color='red', label='Infeasible')
    
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.title('Population')
    plt.show()


def plot_fitness_curve(best_fitness_history):
    plt.plot(best_fitness_history)
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.title('Fitness Curve')
    plt.show()

def run_evolutionary_algorithm(num_generations, pop_size, mutation_rate, crossover_rate, tournament_size):
    population = create_initial_population(pop_size)
    fitness, is_feasible, constraint_violations = evaluate_population(population)
    plot_population(population, is_feasible)

    best_fitness_history = []

    for generation in range(num_generations):
        new_population = []

        for _ in range(pop_size):
            parent1 = tournament_selection(population, fitness, is_feasible, constraint_violations, tournament_size)
            parent2 = tournament_selection(population, fitness, is_feasible, constraint_violations, tournament_size)

            offspring = crossover(parent1, parent2, crossover_rate)
            offspring = mutation(offspring, mutation_rate)

            new_population.append(offspring)

        population = np.array(new_population)
        fitness, is_feasible, constraint_violations = evaluate_population(population)

        best_fitness_history.append(np.min(fitness[is_feasible]))

        if generation>0 and generation % 20 == 0:
            plot_population(population, is_feasible)
            plot_fitness_curve(best_fitness_history)

    plot_population(population, is_feasible)
    plot_fitness_curve(best_fitness_history)

    return population

# Parameters
num_generations = 100
pop_size = 100
mutation_rate = 0.1
crossover_rate = 0.8
tournament_size = 3

# Run the evolutionary algorithm
final_population = run_evolutionary_algorithm(num_generations, pop_size, mutation_rate, crossover_rate, tournament_size)

# Print the final population
print("Final population:")
print(final_population)

