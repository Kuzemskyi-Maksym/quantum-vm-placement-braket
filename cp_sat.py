 """
CP-SAT (Google OR-Tools) — точний класичний baseline.
Той самий QUBO-еквівалентний інстанс, вирішений точно через CP-SAT,
щоб мати оптимальне (або доведено оптимальне в межах time_limit) число активних ФС.
"""

import time

from ortools.sat.python import cp_model

from config import VM_LOADS, PM_CAPACITY, N, M, T, is_feasible, active_pms, pm_loads


def run_cp_sat(time_limit_s=30.0):
    start = time.perf_counter()

    model = cp_model.CpModel()

    x = {(j, i): model.NewBoolVar(f"x_{j}_{i}") for j in range(N) for i in range(M)}
    y = {i: model.NewBoolVar(f"y_{i}") for i in range(M)}

    # кожна ВМ — рівно на одному ФС
    for j in range(N):
        model.Add(sum(x[j, i] for i in range(M)) == 1)

    # місткість з урахуванням порогу T (цілочисельно: 100 * capacity * T як верхня межа *100)
    scale = 100
    for i in range(M):
        cap_scaled = int(round(T * PM_CAPACITY[i] * scale))
        model.Add(sum(VM_LOADS[j] * scale * x[j, i] for j in range(N)) <= cap_scaled * y[i] + cap_scaled * (1 - y[i]) * 0)
        # точніше: якщо y[i]==0, сума має бути 0; якщо y[i]==1, сума <= cap_scaled
        model.Add(sum(VM_LOADS[j] * scale * x[j, i] for j in range(N)) <= cap_scaled * y[i])

    model.Minimize(sum(y[i] for i in range(M)))

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

    return {
        "method": "CP-SAT",
        "status": solver.StatusName(status),
        "assignment": assignment,
        "active_pms": active_pms(assignment) if None not in assignment else None,
        "feasible": is_feasible(assignment) if None not in assignment else False,
        "pm_loads": pm_loads(assignment) if None not in assignment else None,
        "runtime_s": elapsed,
    }


if __name__ == "__main__":
    result = run_cp_sat()
    print(result)
