"""
Побудова CQM (Constrained Quadratic Model) для задачі розміщення ВМ
і конвертація в BQM/QUBO — спільна база для QAOA, SimulatedAnnealingSampler
та генетичного алгоритму (через розкодування бінарних x_{j,i}).

Змінні:
    x[j, i] = 1, якщо ВМ j розміщена на ФС i
    y[i]    = 1, якщо ФС i активний (використовується хоча б однією ВМ)

Мета: мінімізувати кількість активних ФС: sum(y[i])

Обмеження:
    1) кожна ВМ призначена рівно на один ФС:      sum_i x[j,i] == 1   для всіх j
    2) місткість (з урахуванням порогу T):        sum_j vm_loads[j]*x[j,i] - T*cap[i]*y[i] <= 0   для всіх i
"""

import dimod
from dimod import ConstrainedQuadraticModel, Binary, quicksum

from config import VM_LOADS, PM_CAPACITY, N, M, T, LAMBDA1


def build_cqm(vm_loads=None, pm_capacity=None, t=None):
    """Параметризовано, щоб можна було будувати і повний інстанс, і зменшений
    (для QAOA/реального QPU, де повний 21-змінний CQM дає ~33-кубітний BQM
    після slack-конверсії — більше за місткість IQM Garnet)."""
    vm_loads = vm_loads if vm_loads is not None else VM_LOADS
    pm_capacity = pm_capacity if pm_capacity is not None else PM_CAPACITY
    t = t if t is not None else T
    n, m = len(vm_loads), len(pm_capacity)

    x = {(j, i): Binary(f"x_{j}_{i}") for j in range(n) for i in range(m)}
    y = {i: Binary(f"y_{i}") for i in range(m)}

    cqm = ConstrainedQuadraticModel()
    cqm.set_objective(quicksum(y[i] for i in range(m)))

    for j in range(n):
        cqm.add_constraint(
            quicksum(x[j, i] for i in range(m)) == 1,
            label=f"assign_{j}",
        )

    for i in range(m):
        cqm.add_constraint(
            quicksum(vm_loads[j] * x[j, i] for j in range(n)) - t * pm_capacity[i] * y[i] <= 0,
            label=f"capacity_{i}",
        )

    return cqm, x, y


def build_bqm(lagrange_multiplier=LAMBDA1, vm_loads=None, pm_capacity=None, t=None):
    cqm, x, y = build_cqm(vm_loads=vm_loads, pm_capacity=pm_capacity, t=t)
    bqm, invert = dimod.cqm_to_bqm(cqm, lagrange_multiplier=lagrange_multiplier)
    return bqm, invert, cqm


def decode_sample_to_assignment(sample, invert=None, n=N, m=M):
    """
    sample: dict змінна -> 0/1 (з BQM-семплера, вже інвертований через `invert`, якщо задано)
    Повертає assignment[j] = i або None, якщо ВМ j не призначена рівно одному ФС (inf./infeasible).
    """
    if invert is not None:
        sample = invert(sample)
    assignment = []
    for j in range(n):
        assigned = [i for i in range(m) if sample.get(f"x_{j}_{i}", 0) == 1]
        assignment.append(assigned[0] if len(assigned) == 1 else None)
    return assignment


if __name__ == "__main__":
    cqm, x, y = build_cqm()
    print(f"CQM (повний інстанс): {len(cqm.variables)} змінних, {len(cqm.constraints)} обмежень")
    bqm, invert, _ = build_bqm()
    print(f"BQM (lagrange={LAMBDA1}): {len(bqm.variables)} змінних, {len(bqm.quadratic)} квадратичних членів")
