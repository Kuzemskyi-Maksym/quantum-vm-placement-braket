"""
MBFD (Modified Best Fit Decreasing) — класичний евристичний baseline.

Алгоритм:
1. Сортуємо ВМ за навантаженням у порядку зменшення.
2. Для кожної ВМ шукаємо серед ВЖЕ активних ФС той, що має найменший залишок
   місткості, у який ВМ ще вміщається (Best Fit).
3. Якщо жодний активний ФС не підходить — активуємо новий (найменший за індексом
   вільний) і розміщуємо ВМ туди.
"""

import time

from config import VM_LOADS, PM_CAPACITY, N, M, EFFECTIVE_CAPACITY, is_feasible, active_pms, pm_loads


def run_mbfd():
    start = time.perf_counter()

    order = sorted(range(N), key=lambda j: VM_LOADS[j], reverse=True)
    assignment = [None] * N
    loads = [0.0] * M
    active = [False] * M

    for j in order:
        best_i = None
        best_remaining = None
        for i in range(M):
            if not active[i]:
                continue
            remaining = EFFECTIVE_CAPACITY[i] - loads[i]
            if remaining >= VM_LOADS[j] and (best_remaining is None or remaining < best_remaining):
                best_i = i
                best_remaining = remaining
        if best_i is None:
            # активуємо новий найменший вільний ФС
            free = [i for i in range(M) if not active[i]]
            if not free:
                raise RuntimeError("Немає вільних ФС — інстанс нерозв'язний цим правилом")
            best_i = free[0]
            active[best_i] = True

        assignment[j] = best_i
        loads[best_i] += VM_LOADS[j]

    elapsed = time.perf_counter() - start

    return {
        "method": "MBFD",
        "assignment": assignment,
        "active_pms": active_pms(assignment),
        "feasible": is_feasible(assignment),
        "pm_loads": pm_loads(assignment),
        "runtime_s": elapsed,
    }


if __name__ == "__main__":
    result = run_mbfd()
    print(result)
