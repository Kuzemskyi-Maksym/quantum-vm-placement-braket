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
    # "W" — інстанс для warm start / критерію міграцій: 4 ВМ, 3 ФС (15 кубітів у slack-free кодуванні),
    # оптимум 2 ФС; поточний розподіл X0 = [0, 1, 2, 2] (3 активні ФС, допустимий, не оптимальний).
    "W": ([6, 5, 2, 1], [12, 12, 12]),
    # "large" — великий інстанс ЛИШЕ для класичних методів (ГА, BRKGA, CP-SAT): 30 ВМ, 12 ФС.
    # Навантаження зафіксовано (згенеровано random.Random(7), цілі 2..9), щоб результат відтворювався.
    "large": (__import__("random").Random(7).choices(range(2, 10), k=30), [20] * 12),
    # "rand" — випадковий великий інстанс: VMP_SEED (за замовч. 1) і VMP_N (за замовч. 30) задають генерацію.
    "rand": (__import__("random").Random(int(os.environ.get("VMP_SEED", 1))).choices(
        range(3, 10), k=int(os.environ.get("VMP_N", 30))), [20] * int(os.environ.get("VMP_N", 30))),
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

# ---- Критерій міграцій (рішення Жарікова 2026-11-01) ----
# Ціль: F(x) = (кількість активних ФС) + MU * (Σ_j s_j·[ВМ j змінила сервер]) / Σ_j s_j.
# Вартість переміщення пропорційна розміру ВМ; нормування на Σ s_j робить другий член ∈ [0, 1],
# тобто переселити ВСІ ВМ коштує рівно MU. MU < 1: вимкнути один сервер завжди вигідніше за будь-які
# міграції; MU > 1: алгоритм триматиметься поточного розподілу.
# X0 — поточний розподіл (assignment[j] = сервер ВМ j). Змінна середовища VMP_X0="0,1,2,2";
# якщо X0 = None, міграції не рахуються (F = кількість ФС, як раніше).
_X0_DEFAULT = {"W": [0, 1, 2, 2]}
X0 = [int(v) for v in os.environ["VMP_X0"].split(",")] if os.environ.get("VMP_X0") else _X0_DEFAULT.get(_name)
MU = float(os.environ.get("VMP_MU", "1.0"))
TOTAL_LOAD = float(sum(VM_LOADS))


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


def n_migrations(assignment, x0=X0):
    """Кількість ВМ, що змінили сервер відносно x0 (0, якщо x0 не задано)."""
    if x0 is None:
        return 0
    return sum(1 for j in range(N) if assignment[j] != x0[j])


def migration_cost(assignment, x0=X0):
    """Зважена вартість міграцій: Σ_j s_j·[a_j ≠ x0_j] / Σ_j s_j ∈ [0, 1]. 0, якщо x0 не задано."""
    if x0 is None:
        return 0.0
    return sum(VM_LOADS[j] for j in range(N) if assignment[j] != x0[j]) / TOTAL_LOAD


def objective(assignment, x0=X0, mu=MU):
    """Повна ціль F = активні ФС + mu · зважені міграції (менше — краще)."""
    return active_pms(assignment) + mu * migration_cost(assignment, x0)


if __name__ == "__main__":
    print(f"N (ВМ) = {N}, M (ФС) = {M}")
    print(f"VM_LOADS = {VM_LOADS}")
    print(f"PM_CAPACITY = {PM_CAPACITY}, T = {T}")
    print(f"EFFECTIVE_CAPACITY = {EFFECTIVE_CAPACITY}")
    print(f"X0 = {X0}, MU = {MU}" + (f", F(x0) = {objective(X0):.3f}" if X0 else ""))
