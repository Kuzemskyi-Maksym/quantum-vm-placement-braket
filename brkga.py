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

Режим з міграціями (x0 задано, рішення Жарікова 2026-11-01):
  - ціль F = активні ФС + mu · Σ s_j·[ВМ j переїхала] / Σ s_j (див. config.objective);
  - особина має 2N ключів: перші N — порядок, другі N — прапорець «залишитись»:
    якщо ключ_stay[j] >= 0.5, ВМ j спершу пробує СВІЙ поточний сервер x0[j] (якщо влазить),
    інакше — First Fit спершу по вже непорожніх серверах, потім по порожніх (без цього переїзд на активний
    ФС при порожньому ФС з меншим індексом був недосяжний — баг, виправлено 2026-11-17).
    Так еволюція сама обирає, яких ВМ рухати, а яких ні;
  - у початкову популяцію додано особину «нічого не рухати» (= x0) і особину FFD («повна переукладка»).

Інстанс за замовчуванням — з config (VMP_INSTANCE); `run_brkga(loads=..., caps=...)` розв'язує довільний
(потрібно для симуляції життєвого циклу, lifecycle_sim.py).
"""

import random
import time

import config
from config import T

PENALTY = 1000.0


def _loads_of(assignment, loads, m):
    out = [0.0] * m
    for j, i in enumerate(assignment):
        out[i] += loads[j]
    return out


def _mig_cost(assignment, x0, loads):
    tot = float(sum(loads)) or 1.0
    return sum(loads[j] for j in range(len(loads)) if assignment[j] != x0[j]) / tot


def decode(keys, x0=None, loads=None, eff_caps=None):
    """Ключі -> розклад (assignment[j] = індекс сервера). First Fit у порядку ключів.
    Якщо x0 задано і len(keys) == 2N, ключі N..2N-1 — прапорці «спершу спробувати поточний сервер»."""
    loads = loads if loads is not None else config.VM_LOADS
    eff_caps = eff_caps if eff_caps is not None else config.EFFECTIVE_CAPACITY
    n, m = len(loads), len(eff_caps)
    order = sorted(range(n), key=lambda j: keys[j])
    stay = keys[n:] if (x0 is not None and len(keys) >= 2 * n) else None
    cur = [0.0] * m
    assignment = [None] * n
    for j in order:
        target = None
        if stay is not None and stay[j] >= 0.5:
            i = x0[j]
            if cur[i] + loads[j] <= eff_caps[i] + 1e-9:
                target = i
        if target is None:
            # без x0 — First Fit за індексом (порожні — теж "перший вільний"), як було;
            # з x0 — спершу вже непорожні на цьому кроці декодування сервери, потім порожні (виправлення 2026-11-17:
            # інакше переїзд на активний ФС при порожньому ФС з меншим індексом недосяжний, напр. [1,2,1,0] → [1,2,1,1])
            cand = range(m) if x0 is None else \
                [i for i in range(m) if cur[i] > 0] + [i for i in range(m) if cur[i] <= 0]
            for i in cand:
                if cur[i] + loads[j] <= eff_caps[i] + 1e-9:
                    target = i
                    break
        if target is None:                       # нікуди не влазить: найменш завантажений (недопустимо)
            target = min(range(m), key=lambda i: cur[i] - eff_caps[i])
        assignment[j] = target
        cur[target] += loads[j]
    return assignment


def fitness(assignment, x0=None, mu=0.0, loads=None, eff_caps=None):
    """Менше — краще: активні ФС + mu·зважені міграції + штраф за перевантаження
    − невеликий бонус за «щільність» (сума квадратів завантажень), що розрізняє розклади
    з однаковою ціллю і веде пошук до «повних» серверів. У режимі міграцій бонус зменшено
    (0.01 замість 0.5), щоб він лишався лише tie-breaker-ом і не перекривав член міграцій."""
    loads = loads if loads is not None else config.VM_LOADS
    eff_caps = eff_caps if eff_caps is not None else config.EFFECTIVE_CAPACITY
    m = len(eff_caps)
    cur = _loads_of(assignment, loads, m)
    overload = sum(max(0.0, cur[i] - eff_caps[i]) for i in range(m))
    total = sum(cur) or 1.0
    density = sum(l * l for l in cur) / (total * total)   # у (0, 1]
    base = len(set(assignment)) + PENALTY * overload
    if x0 is None:
        return base - 0.5 * density
    return base + mu * _mig_cost(assignment, x0, loads) - 0.01 * density


def run_brkga(pop_size=60, generations=100, elite_frac=0.2, mutant_frac=0.15, rho=0.7, seed=42,
              seed_ffd=True, x0="config", mu="config", loads=None, caps=None):
    """x0/mu за замовчуванням — з config (X0, MU); передайте x0=None, щоб вимкнути міграції.
    loads/caps — довільний інстанс (caps — номінальні місткості, поріг T застосовується тут)."""
    if x0 == "config":
        x0 = config.X0
    if mu == "config":
        mu = config.MU
    loads = list(loads) if loads is not None else list(config.VM_LOADS)
    eff_caps = [T * c for c in caps] if caps is not None else list(config.EFFECTIVE_CAPACITY)
    n = len(loads)
    start = time.perf_counter()
    rng = random.Random(seed)
    n_elite = max(1, int(pop_size * elite_frac))
    n_mut = max(1, int(pop_size * mutant_frac))
    use_mig = x0 is not None and mu > 0
    L = 2 * n if use_mig else n                      # довжина особини

    population = [[rng.random() for _ in range(L)] for _ in range(pop_size)]
    ffd_order = sorted(range(n), key=lambda j: -loads[j])
    ffd_keys = [0.0] * n
    for rank, j in enumerate(ffd_order):
        ffd_keys[j] = rank / n
    if seed_ffd:
        population[0] = ffd_keys + ([0.0] * n if use_mig else [])        # FFD, нічого не «тримати»
    if use_mig:
        population[1] = ffd_keys + [1.0] * n                              # «нічого не рухати» = x0

    x0d = x0 if use_mig else None

    def evaluate(pop):
        scored = []
        for keys in pop:
            a = decode(keys, x0d, loads, eff_caps)
            scored.append((fitness(a, x0d, mu, loads, eff_caps), keys, a))
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
    cur = _loads_of(best_assign, loads, len(eff_caps))
    feasible = all(cur[i] <= eff_caps[i] + 1e-9 for i in range(len(eff_caps)))
    migc = _mig_cost(best_assign, x0, loads) if use_mig else 0.0
    return {
        "method": "BRKGA (ГА зі зміщеними випадковими ключами)",
        "assignment": best_assign,
        "active_pms": len(set(best_assign)),
        "feasible": feasible,
        "pm_loads": cur,
        "migrations": sum(1 for j in range(n) if best_assign[j] != x0[j]) if use_mig else None,
        "migration_cost": migc if use_mig else None,
        "objective": len(set(best_assign)) + (mu * migc if use_mig else 0.0),
        "mu": mu if use_mig else 0.0,
        "runtime_s": elapsed,
        "generations": generations,
        "pop_size": pop_size,
        "best_fitness": best_fit,
    }


if __name__ == "__main__":
    print(run_brkga())
