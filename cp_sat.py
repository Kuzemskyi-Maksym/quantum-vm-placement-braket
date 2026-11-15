"""
CP-SAT (Google OR-Tools) — точний класичний baseline.
Той самий інстанс, вирішений точно через CP-SAT: доведено оптимальне (в межах time_limit)
значення цілі F = активні ФС + MU · зважені міграції відносно поточного розподілу X0.
Якщо X0 = None (або mu = 0), ціль — лише кількість активних ФС, як раніше.
"""

import time

from ortools.sat.python import cp_model

from config import (VM_LOADS, PM_CAPACITY, N, M, T, X0, MU, TOTAL_LOAD,
                    is_feasible, active_pms, pm_loads, n_migrations, migration_cost, objective)

SCALE = 1000  # ціль цілочисельна: 1 сервер = SCALE одиниць, міграція ВМ j = round(mu·SCALE·s_j/Σs)


def run_cp_sat(time_limit_s=30.0, x0=X0, mu=MU):
    start = time.perf_counter()

    model = cp_model.CpModel()

    x = {(j, i): model.NewBoolVar(f"x_{j}_{i}") for j in range(N) for i in range(M)}
    y = {i: model.NewBoolVar(f"y_{i}") for i in range(M)}

    # кожна ВМ — рівно на одному ФС
    for j in range(N):
        model.Add(sum(x[j, i] for i in range(M)) == 1)

    # місткість з урахуванням порогу T (цілочисельно, масштаб 100): якщо y_i = 0, сума = 0; інакше <= T·C_i
    scale = 100
    for i in range(M):
        cap_scaled = int(round(T * PM_CAPACITY[i] * scale))
        model.Add(sum(VM_LOADS[j] * scale * x[j, i] for j in range(N)) <= cap_scaled * y[i])

    obj = SCALE * sum(y[i] for i in range(M))
    if x0 is not None and mu > 0:
        # вартість міграції ВМ j: платимо, якщо вона НЕ на своєму поточному сервері x0[j]
        obj = obj + sum(int(round(mu * SCALE * VM_LOADS[j] / TOTAL_LOAD)) * (1 - x[j, x0[j]]) for j in range(N))
    model.Minimize(obj)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    status = solver.Solve(model)

    elapsed = time.perf_counter() - start

    assignment = [None] * N
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for j in range(N):
            for i in range(M):
                if solver.Value(x[j, i]) == 1:
                    assignment[j] = i

    ok = None not in assignment
    return {
        "method": "CP-SAT",
        "status": solver.StatusName(status),
        "assignment": assignment,
        "active_pms": active_pms(assignment) if ok else None,
        "feasible": is_feasible(assignment) if ok else False,
        "pm_loads": pm_loads(assignment) if ok else None,
        "migrations": n_migrations(assignment, x0) if ok else None,
        "migration_cost": migration_cost(assignment, x0) if ok else None,
        "objective": objective(assignment, x0, mu) if ok else None,
        "mu": mu,
        "runtime_s": elapsed,
    }


def solve_cpsat(loads, caps, x0=None, mu=0.0, time_limit=10.0, t=T, workers=8):
    """Та сама модель, що й run_cp_sat, але для довільного інстансу (lifecycle_sim.py, алгоритм 1 як еталон):
    loads — навантаження ВМ, caps — номінальні місткості (поріг t застосовується тут), x0 — поточний розподіл
    (None або mu=0 → лише кількість ФС). Повертає assignment, status, objective (ФС + mu·зважені міграції)."""
    start = time.perf_counter()
    n, m = len(loads), len(caps)
    total = float(sum(loads)) or 1.0
    model = cp_model.CpModel()
    x = {(j, i): model.NewBoolVar(f"x_{j}_{i}") for j in range(n) for i in range(m)}
    y = {i: model.NewBoolVar(f"y_{i}") for i in range(m)}
    for j in range(n):
        model.Add(sum(x[j, i] for i in range(m)) == 1)
    scale = 100
    for i in range(m):
        model.Add(sum(int(round(loads[j] * scale)) * x[j, i] for j in range(n)) <= int(round(t * caps[i] * scale)) * y[i])
    obj = SCALE * sum(y.values())
    use_mig = x0 is not None and mu > 0
    if use_mig:
        obj = obj + sum(int(round(mu * SCALE * loads[j] / total)) * (1 - x[j, x0[j]]) for j in range(n))
        for j in range(n):                                   # підказка: поточний розподіл
            for i in range(m):
                model.AddHint(x[j, i], int(x0[j] == i))
    model.Minimize(obj)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = workers
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return dict(assignment=None, status=solver.StatusName(status), objective=None,
                    runtime_s=time.perf_counter() - start)
    a = [next(i for i in range(m) if solver.Value(x[j, i])) for j in range(n)]
    migc = sum(loads[j] for j in range(n) if a[j] != x0[j]) / total if use_mig else 0.0
    return dict(assignment=a, status=solver.StatusName(status), active_pms=len(set(a)),
                objective=len(set(a)) + (mu * migc if use_mig else 0.0), runtime_s=time.perf_counter() - start)


if __name__ == "__main__":
    result = run_cp_sat()
    print(result)
