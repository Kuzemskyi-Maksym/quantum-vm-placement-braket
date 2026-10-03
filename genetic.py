"""
Генетичний алгоритм (метод #4, новий, за вимогою Жарікова) — працює
безпосередньо на повному CQM-інстансі (6 ВМ, 3 ФС), без QUBO-конверсії.

Представлення хромосоми: список довжини N, ген j = індекс ФС, куди
призначена ВМ j (пряме кодування assignment, без x/y бінарних змінних —
природне представлення для ГА, еквівалентне тому самому CQM).

Фітнес: active_pms(assignment) + великий штраф за недопустимість (щоб
допустимі рішення завжди були кращі за недопустимі), сумісний з тим самим
обмеженням місткості T*cap, що й в інших методах.
"""

import random
import time

from config import VM_LOADS, PM_CAPACITY, N, M, EFFECTIVE_CAPACITY, is_feasible, active_pms, pm_loads

PENALTY = 1000.0


def overload_amount(assignment):
    loads = [0.0] * M
    for j, i in enumerate(assignment):
        loads[i] += VM_LOADS[j]
    return sum(max(0.0, loads[i] - EFFECTIVE_CAPACITY[i]) for i in range(M))


def fitness(assignment):
    """Менше — краще. Допустимі рішення завжди кращі за недопустимі (великий штраф)."""
    overload = overload_amount(assignment)
    return active_pms(assignment) + PENALTY * overload


def random_individual(rng):
    return [rng.randrange(M) for _ in range(N)]


def tournament_select(population, fitnesses, rng, k=3):
    idxs = rng.sample(range(len(population)), k)
    best = min(idxs, key=lambda i: fitnesses[i])
    return population[best]


def crossover(p1, p2, rng):
    point = rng.randrange(1, N)
    return p1[:point] + p2[point:]


def mutate(ind, rng, rate=0.2):
    return [rng.randrange(M) if rng.random() < rate else gene for gene in ind]


def run_genetic(pop_size=60, generations=150, crossover_rate=0.8, mutation_rate=0.2, seed=42):
    start = time.perf_counter()
    rng = random.Random(seed)

    population = [random_individual(rng) for _ in range(pop_size)]
    best_ever, best_ever_fit = None, float("inf")
    history = []

    for gen in range(generations):
        fitnesses = [fitness(ind) for ind in population]

        gen_best_idx = min(range(pop_size), key=lambda i: fitnesses[i])
        if fitnesses[gen_best_idx] < best_ever_fit:
            best_ever = population[gen_best_idx][:]
            best_ever_fit = fitnesses[gen_best_idx]
        history.append(best_ever_fit)

        new_population = [best_ever[:]]  # елітизм: завжди зберігаємо найкращого
        while len(new_population) < pop_size:
            p1 = tournament_select(population, fitnesses, rng)
            p2 = tournament_select(population, fitnesses, rng)
            child = crossover(p1, p2, rng) if rng.random() < crossover_rate else p1[:]
            child = mutate(child, rng, rate=mutation_rate)
            new_population.append(child)
        population = new_population

    elapsed = time.perf_counter() - start

    return {
        "method": "Генетичний алгоритм",
        "assignment": best_ever,
        "active_pms": active_pms(best_ever),
        "feasible": is_feasible(best_ever),
        "pm_loads": pm_loads(best_ever),
        "runtime_s": elapsed,
        "generations": generations,
        "pop_size": pop_size,
        "best_fitness": best_ever_fit,
    }


if __name__ == "__main__":
    result = run_genetic()
    print(result)
