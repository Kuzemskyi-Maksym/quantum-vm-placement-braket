"""
Спільна тестова конфігурація для порівняння всіх методів.
Відтворює інстанс із першої статті (VM placement / consolidation).

ВАЖЛИВО: усі методи (MBFD, CP-SAT, QAOA, генетичний алгоритм, аналог відпалу)
повинні використовувати ЦІ ЖЕ значення, щоб результати були порівнювані.
"""

import os

# Вибір інстансу змінною середовища VMP_INSTANCE:
#   full   (за замовчуванням) — 6 ВМ / 3 ФС із першої статті (лише класичні методи; ~33 кубіти зі slack)
#   common — 4 ВМ / 2 ФС: ЄДИНИЙ інстанс для всіх 5 методів, 10 кубітів у slack-free кодуванні
_INSTANCES = {
    "full": ([3, 4, 2, 5, 3, 4], [12, 12, 12]),
    "common": ([6, 5, 2, 1], [12, 12]),
}
_name = os.environ.get("VMP_INSTANCE", "full")

# Навантаження віртуальних машин (ВМ), j = 0..N-1
# Місткість фізичних серверів (ФС), i = 0..M-1
VM_LOADS, PM_CAPACITY = (list(v) for v in _INSTANCES[_name])

N = len(VM_LOADS)   # кількість ВМ = 6
M = len(PM_CAPACITY)  # кількість ФС = 3

# Порогова утилізація (threshold) — ФС не може бути завантажений більш ніж T * capacity
T = 0.90

# Множники штрафу (Lagrange) при переведенні CQM -> BQM/QUBO
LAMBDA1 = 6.0  # штраф за порушення "рівно одне призначення" (assignment constraint)
LAMBDA2 = 6.0  # штраф за порушення місткості (capacity constraint)

# Для QAOA / генетичного алгоритму: ефективна місткість з урахуванням T
EFFECTIVE_CAPACITY = [T * c for c in PM_CAPACITY]


def is_feasible(assignment):
    """
    assignment: список довжини N, assignment[j] = i (індекс ФС, куди призначено ВМ j)
    Повертає True, якщо жодний ФС не перевантажений понад T * capacity.
    """
    loads = [0.0] * M
    for j, i in enumerate(assignment):
        loads[i] += VM_LOADS[j]
    return all(loads[i] <= EFFECTIVE_CAPACITY[i] + 1e-9 for i in range(M))


def active_pms(assignment):
    """Кількість фізично активних (непорожніх) ФС — цільова функція, яку мінімізуємо."""
    used = set(assignment)
    return len(used)


def pm_loads(assignment):
    loads = [0.0] * M
    for j, i in enumerate(assignment):
        loads[i] += VM_LOADS[j]
    return loads


if __name__ == "__main__":
    print(f"N (ВМ) = {N}, M (ФС) = {M}")
    print(f"VM_LOADS = {VM_LOADS}")
    print(f"PM_CAPACITY = {PM_CAPACITY}, T = {T}")
    print(f"EFFECTIVE_CAPACITY = {EFFECTIVE_CAPACITY}")
