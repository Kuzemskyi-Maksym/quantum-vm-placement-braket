"""
BRKGA — Biased Random-Key Genetic Algorithm (ГА зі зміщеними випадковими ключами).
Сучасна заміна базового ГА (genetic.py) для задачі упаковки ВМ на сервери.

Ідея (Gonçalves & Resende, 2013, Int. J. Production Economics):
  - особина = список з N випадкових чисел ("ключів") у [0, 1], по одному на ВМ;
  - окремий ДЕКОДЕР перетворює ключі на розклад: ВМ розглядаються в порядку зростання
    ключа, кожна ставиться на ПЕРШИЙ активний сервер, де ще влазить (First Fit), інакше
    вмикається наступний сервер. Тому ліміти місткості виконуються самі, штрафи не потрібні;
  - еволюція: еліта (найкращі) копіюється, частина популяції — нові випадкові "мутанти",
    решта — діти від схрещування еліти з не-елітою, де кожен ключ береремо від еліти з
    імовірністю RHO (зміщене, "biased" схрещування);
  - у початкову популяцію додано одну особину з порядком First Fit Decreasing
    (найбільші ВМ першими), щоб алгоритм стартував не гірше за MBFD-подібну евристику.
"""

import random
import time

from config import VM_LOADS, N, M, EFFECTIVE_CAPACITY, is_feasible, active_pms, pm_loads

PENALTY = 1000.0


def decode(keys):
    """Ключі -> розклад (assignment[j] = індекс сервера). First Fit у порядку ключів."""
    order = sorted(range(N), key=lambda j: keys[j])
    loads = [0.0] * M
    assignment = [None] * N
    for j in order:
        target = None
        for i in range(M):                       # перший сервер, де ВМ влазить (порожні — теж "перший вільний")
            if loads[i] + VM_LOADS[j] <= EFFECTIVE_CAPACITY[i] + 1e-9:
                target = i
                break
        if target is None:                       # нікуди не влазить: найменш завантажений (недопустимо)
            target = min(range(M), key=lambda i: loads[i] - EFFECTIVE_CAPACITY[i])
        assignment[j] = target
        loads[target] += VM_LOADS[j]
    return assignment


def fitness(assignment):
    """Менше — краще: кількість активних серверів + штраф за перевантаження
    − невеликий бонус за «щільність» (сума квадратів завантажень), що розрізняє розклади
    з однаковою кількістю серверів і веде пошук до «повних» серверів."""
    loads = pm_loads(assignment)
    overload = sum(max(0.0, loads[i] - EFFECTIVE_CAPACITY[i]) for i in range(M))
    total = sum(loads) or 1.0
    density = sum(l * l for l in loads) / (total * total)   # у (0, 1]
    return active_pms(assignment) + PENALTY * overload - 0.5 * density


def run_brkga(pop_size=60, generations=100, elite_frac=0.2, mutant_frac=0.15, rho=0.7, seed=42, seed_ffd=True):
    start = time.perf_counter()
    rng = random.Random(seed)
    n_elite = max(1, int(pop_size * elite_frac))
    n_mut = max(1, int(pop_size * mutant_frac))

    population = [[rng.random() for _ in range(N)] for _ in range(pop_size)]
    ffd_order = sorted(range(N), key=lambda j: -VM_LOADS[j])
    ffd_keys = [0.0] * N
    for rank, j in enumerate(ffd_order):
        ffd_keys[j] = rank / N
    if seed_ffd:
        population[0] = ffd_keys

    def evaluate(pop):
        scored = []
        for keys in pop:
            a = decode(keys)
            scored.append((fitness(a), keys, a))
        scored.sort(key=lambda t: t[0])
        return scored

    scored = evaluate(population)
    best_fit, _, best_assign = scored[0]

    for _ in range(generations):
        elite = [s[1] for s in scored[:n_elite]]
        non_elite = [s[1] for s in scored[n_elite:]]
        new_pop = [k[:] for k in elite]
        new_pop += [[rng.random() for _ in range(N)] for _ in range(n_mut)]
        while len(new_pop) < pop_size:
            e = rng.choice(elite)
            o = rng.choice(non_elite) if non_elite else rng.choice(elite)
            new_pop.append([e[g] if rng.random() < rho else o[g] for g in range(N)])
        scored = evaluate(new_pop)
        if scored[0][0] < best_fit:
            best_fit, _, best_assign = scored[0]

    elapsed = time.perf_counter() - start
    return {
        "method": "BRKGA (ГА зі зміщеними випадковими ключами)",
        "assignment": best_assign,
        "active_pms": active_pms(best_assign),
        "feasible": is_feasible(best_assign),
        "pm_loads": pm_loads(best_assign),
        "runtime_s": elapsed,
        "generations": generations,
        "pop_size": pop_size,
        "best_fitness": best_fit,
    }


if __name__ == "__main__":
    print(run_brkga())
