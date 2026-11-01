"""
BRKGA — Biased Random-Key Genetic Algorithm (ГА зі зміщеними випадковими ключами).
Сучасна заміна базового ГА (genetic.py) для задачі упаковки ВМ на сервери.

Ідея (Gonçalves & Resende, 2013, Int. J. Production Economics):
  - особина = список з N випадкових чисел ("ключів") у [0, 1], по одному на ВМ;
  - окремий ДЕКОДЕР перетворює ключі на розклад: ВМ розглядаються в порядку зростання
    ключа, кожна ставиться на ПЕРШИЙ активний сервер, де ще влазить (First Fit), інакше
    вмикається наступний сервер. Тому ліміти місткості виконуються самі, штрафи не потрібні;
  - еволюція: еліта (найкращі) копіюється, частина популяції — нові випадкові "мутанти",
    решта — діти від схрещування еліти з не-елітою, де кожен ключ беремо від еліти з
    імовірністю RHO (зміщене, "biased" схрещування);
  - у початкову популяцію додано одну особину з порядком First Fit Decreasing
    (найбільші ВМ першими), щоб алгоритм стартував не гірше за MBFD-подібну евристику.

Режим з міграціями (X0 задано, рішення Жарікова 2026-11-01):
  - ціль F = активні ФС + MU · Σ s_j·[ВМ j переїхала] / Σ s_j (див. config.objective);
  - особина має 2N ключів: перші N — порядок, другі N — прапорець «залишитись»:
    якщо ключ_stay[j] >= 0.5, ВМ j спершу пробує СВІЙ поточний сервер x0[j] (якщо влазить),
    інакше — звичайний First Fit. Так еволюція сама обирає, яких ВМ рухати, а яких ні;
  - у початкову популяцію додано особину «нічого не рухати» (= x0) і особину FFD («повна переукладка»).
"""

import random
import time

from config import (VM_LOADS, N, M, EFFECTIVE_CAPACITY, X0, MU,
                    is_feasible, active_pms, pm_loads, n_migrations, migration_cost)

PENALTY = 1000.0


def decode(keys, x0=None):
    """Ключі -> розклад (assignment[j] = індекс сервера). First Fit у порядку ключів.
    Якщо x0 задано і len(keys) == 2N, ключі N..2N-1 — прапорці «спершу спробувати поточний сервер»."""
    order = sorted(range(N), key=lambda j: keys[j])
    stay = keys[N:] if (x0 is not None and len(keys) >= 2 * N) else None
    loads = [0.0] * M
    assignment = [None] * N
    for j in order:
        target = None
        if stay is not None and stay[j] >= 0.5:
            i = x0[j]
            if loads[i] + VM_LOADS[j] <= EFFECTIVE_CAPACITY[i] + 1e-9:
                target = i
        if target is None:
            for i in range(M):                   # перший сервер, де ВМ влазить (порожні — теж "перший вільний")
                if loads[i] + VM_LOADS[j] <= EFFECTIVE_CAPACITY[i] + 1e-9:
                    target = i
                    break
        if target is None:                       # нікуди не влазить: найменш завантажений (недопустимо)
            target = min(range(M), key=lambda i: loads[i] - EFFECTIVE_CAPACITY[i])
        assignment[j] = target
        loads[target] += VM_LOADS[j]
    return assignment


def fitness(assignment, x0=None, mu=0.0):
    """Менше — краще: активні ФС + mu·зважені міграції + штраф за перевантаження
    − невеликий бонус за «щільність» (сума квадратів завантажень), що розрізняє розклади
    з однаковою ціллю і веде пошук до «повних» серверів. У режимі міграцій бонус зменшено
    (0.01 замість 0.5), щоб він лишався лише tie-breaker-ом і не перекривав член міграцій."""
    loads = pm_loads(assignment)
    overload = sum(max(0.0, loads[i] - EFFECTIVE_CAPACITY[i]) for i in range(M))
    total = sum(loads) or 1.0
    density = sum(l * l for l in loads) / (total * total)   # у (0, 1]
    base = active_pms(assignment) + PENALTY * overload
    if x0 is None:
        return base - 0.5 * density
    return base + mu * migration_cost(assignment, x0) - 0.01 * density


def run_brkga(pop_size=60, generations=100, elite_frac=0.2, mutant_frac=0.15, rho=0.7, seed=42,
              seed_ffd=True, x0=X0, mu=MU):
    start = time.perf_counter()
    rng = random.Random(seed)
    n_elite = max(1, int(pop_size * elite_frac))
    n_mut = max(1, int(pop_size * mutant_frac))
    use_mig = x0 is not None and mu > 0
    L = 2 * N if use_mig else N                      # довжина особини

    population = [[rng.random() for _ in range(L)] for _ in range(pop_size)]
    ffd_order = sorted(range(N), key=lambda j: -VM_LOADS[j])
    ffd_keys = [0.0] * N
    for rank, j in enumerate(ffd_order):
        ffd_keys[j] = rank / N
    if seed_ffd:
        population[0] = ffd_keys + ([0.0] * N if use_mig else [])        # FFD, нічого не «тримати»
    if use_mig:
        population[1] = ffd_keys + [1.0] * N                              # «нічого не рухати» = x0

    x0d = x0 if use_mig else None

    def evaluate(pop):
        scored = []
        for keys in pop:
            a = decode(keys, x0d)
            scored.append((fitness(a, x0d, mu), keys, a))
        scored.sort(key=lambda t: t[0])
        return scored

    scored = evaluate(population)
    best_fit, _, best_assign = scored[0]

    for _ in range(generations):
        elite = [s[1] for s in scored[:n_elite]]
        non_elite = [s[1] for s in scored[n_elite:]]
        new_pop = [k[:] for k in elite]
        new_pop += [[rng.random() for _ in range(L)] for _ in range(n_mut)]
        while len(new_pop) < pop_size:
            e = rng.choice(elite)
            o = rng.choice(non_elite) if non_elite else rng.choice(elite)
            new_pop.append([e[g] if rng.random() < rho else o[g] for g in range(L)])
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
        "migrations": n_migrations(best_assign, x0) if use_mig else None,
        "migration_cost": migration_cost(best_assign, x0) if use_mig else None,
        "objective": active_pms(best_assign) + (mu * migration_cost(best_assign, x0) if use_mig else 0.0),
        "mu": mu if use_mig else 0.0,
        "runtime_s": elapsed,
        "generations": generations,
        "pop_size": pop_size,
        "best_fitness": best_fit,
    }


if __name__ == "__main__":
    print(run_brkga())
