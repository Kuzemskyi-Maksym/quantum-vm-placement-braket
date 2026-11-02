"""
АЛГОРИТМ 2 — розміщення НОВИХ ВМ на частково заповнені сервери (сценарій Жарікова 2026-11-01:
ЦОД працює довго, частина ВМ вимкнулась, прийшли нові; їх розміщує окремий алгоритм, без міграцій).

Вхід: залишкові місткості серверів (після наявних ВМ), які сервери вже увімкнені, пакет нових ВМ.
Ціль: мінімум НОВИХ увімкнених серверів (уже активні коштують 0); міграцій немає за означенням.

Методи:
  - Best Fit Decreasing на залишкових місткостях (класична онлайн-евристика; спершу активні сервери,
    новий сервер вмикається лише коли ВМ нікуди не влазить);
  - повний перебір (точний оптимум, для пакетів до ~6 ВМ);
  - CP-SAT (точний, якщо є OR-Tools);
  - QAOA ЛИШЕ над новими ВМ (точна симуляція, холодний старт): кубіти = (нові ВМ) × (ФС) + (порожні ФС).
    Старі ВМ зафіксовані і в схему не входять, тому кількість кубітів не залежить від розміру ЦОД —
    для 2 нових ВМ на 3 ФС це 6 + (кількість порожніх ФС) ≤ 9 кубітів, що влазить в IQM Garnet (20).
    Місткість кодується штрафами за мінімальні порушуючі підмножини розміру 1 (ВМ не влазить у залишок)
    і 2 (пара разом не влазить); підмножини ≥ 3 не підтримуються (assert, як у qaoa_noslack).

Запуск: python placement_new.py            (приклад: 3 ФС по 12, наявні ВМ [6,2 | 5 | —], нові [3, 4])
        python placement_new.py --new 3,4,2
"""
import itertools, sys, time
import numpy as np
from scipy.optimize import minimize
from qaoa_exact import qaoa_state

T = 0.9
LAM = 3.0


def residual(caps, existing, t=T):
    """existing: список списків навантажень на кожному сервері. Повертає (залишки, прапорці активності)."""
    res = [t * c - sum(v) for c, v in zip(caps, existing)]
    active = [len(v) > 0 for v in existing]
    return res, active


def new_servers(assign, active):
    return len({i for i in assign if not active[i]})


# ---------- класика ----------
def best_fit_decreasing(new_loads, res, active):
    """Best Fit Decreasing: ВМ за спаданням; серед серверів, де влазить, беремо активний з найменшим залишком;
    якщо жоден активний не підходить — порожній сервер з найменшим залишком, що вміщує ВМ."""
    res, active = list(res), list(active)
    order = sorted(range(len(new_loads)), key=lambda j: -new_loads[j])
    assign = [None] * len(new_loads)
    for j in order:
        fits = [i for i in range(len(res)) if res[i] >= new_loads[j] - 1e-9]
        if not fits:
            raise RuntimeError(f"ВМ {new_loads[j]} нікуди не влазить")
        act = [i for i in fits if active[i]]
        pool = act if act else fits
        i = min(pool, key=lambda k: res[k])
        assign[j] = i; res[i] -= new_loads[j]; active[i] = True
    return assign


def brute_force(new_loads, res, active):
    best, arg = None, None
    for a in itertools.product(range(len(res)), repeat=len(new_loads)):
        ld = [0.0] * len(res)
        for j, i in enumerate(a):
            ld[i] += new_loads[j]
        if any(ld[i] > res[i] + 1e-9 for i in range(len(res))):
            continue
        k = new_servers(a, active)
        if best is None or k < best:
            best, arg = k, list(a)
    return best, arg


def cp_sat(new_loads, res, active, time_limit_s=10.0):
    from ortools.sat.python import cp_model
    k, m = len(new_loads), len(res)
    mdl = cp_model.CpModel()
    x = {(j, i): mdl.NewBoolVar(f"x{j}_{i}") for j in range(k) for i in range(m)}
    y = {i: mdl.NewBoolVar(f"y{i}") for i in range(m) if not active[i]}
    for j in range(k):
        mdl.Add(sum(x[j, i] for i in range(m)) == 1)
    for i in range(m):
        cap = int(round(100 * res[i]))
        if active[i]:
            mdl.Add(sum(int(round(100 * new_loads[j])) * x[j, i] for j in range(k)) <= cap)
        else:
            mdl.Add(sum(int(round(100 * new_loads[j])) * x[j, i] for j in range(k)) <= cap * y[i])
    mdl.Minimize(sum(y.values()))
    s = cp_model.CpSolver(); s.parameters.max_time_in_seconds = time_limit_s
    st = s.Solve(mdl)
    a = [next(i for i in range(m) if s.Value(x[j, i])) for j in range(k)]
    return a, s.StatusName(st)


# ---------- QAOA лише над новими ВМ ----------
def build_new(new_loads, res, active, lam=LAM):
    """Енергія над станами: кубіти x_{j,i} (j — нова ВМ, i — сервер) + y_i для ПОРОЖНІХ серверів.
    H = Σ_{i порожній} y_i + lam·Σ_j (Σ_i x_ji − 1)² + lam·Σ_{j, i порожній} x_ji(1 − y_i)
        + lam·Σ_{j,i: s_j > res_i} x_ji + lam·Σ_{i, (j,k): s_j+s_k > res_i} x_ji x_ki."""
    k, m = len(new_loads), len(res)
    empty = [i for i in range(m) if not active[i]]
    yi = {i: k * m + t for t, i in enumerate(empty)}
    n = k * m + len(empty)
    xi = lambda j, i: j * m + i
    for r in range(3, k + 1):                                   # перевірка: порушуючі підмножини ≤ 2
        for S in itertools.combinations(range(k), r):
            for i in range(m):
                if sum(new_loads[j] for j in S) > res[i] + 1e-9:
                    assert any(new_loads[a] > res[i] + 1e-9 for a in S) or \
                           any(new_loads[a] + new_loads[b] > res[i] + 1e-9 for a, b in itertools.combinations(S, 2)), \
                        "мінімальна порушуюча підмножина ≥ 3 — парних штрафів недостатньо"
    St = np.arange(2 ** n)
    X = ((St[:, None] >> (n - 1 - np.arange(n))[None, :]) & 1).astype(np.int8)
    E = np.zeros(2 ** n)
    for i in empty:
        E += X[:, yi[i]]
    for j in range(k):
        E += lam * (sum(X[:, xi(j, i)] for i in range(m)) - 1) ** 2
        for i in empty:
            E += lam * X[:, xi(j, i)] * (1 - X[:, yi[i]])
        for i in range(m):
            if new_loads[j] > res[i] + 1e-9:
                E += lam * X[:, xi(j, i)]
    for i in range(m):
        for a, b in itertools.combinations(range(k), 2):
            if new_loads[a] + new_loads[b] > res[i] + 1e-9:
                E += lam * X[:, xi(a, i)] * X[:, xi(b, i)]
    xs = np.stack([np.stack([X[:, xi(j, i)] for i in range(m)], 1) for j in range(k)], 1)   # (S,k,m)
    onehot = (xs.sum(2) == 1).all(1)
    assign = xs.argmax(2)
    ld = np.stack([((assign == i) * np.array(new_loads, dtype=float)[None, :]).sum(1) for i in range(m)], 1)
    capok = (ld <= np.array(res)[None, :] + 1e-9).all(1)
    linked = np.ones(2 ** n, bool)
    for i in empty:
        linked &= (xs[:, :, i].sum(1) == 0) | (X[:, yi[i]] == 1)
    feasible = onehot & capok & linked
    obj = np.array([new_servers(a, active) for a in assign])
    best = obj[feasible].min()
    optimal = feasible & (obj == best)
    return n, E, onehot, feasible, optimal, best


def qaoa_new(new_loads, res, active, ps=(1, 2, 3), restarts=30, seed=0):
    n, E, onehot, feasible, optimal, best = build_new(new_loads, res, active)
    Ez = (E - E.min()) / (E.max() - E.min())
    rng = np.random.default_rng(seed)
    out = dict(n_qubits=n, optimum_new_servers=int(best), baseline_feasible=float(feasible.mean()),
               baseline_optimal=float(optimal.mean()), by_p={})
    for p in ps:
        bestr = None
        for _ in range(restarts):
            x0 = np.concatenate([rng.uniform(0, np.pi / 2, p), rng.uniform(0, 2 * np.pi, p)])
            r = minimize(lambda x: float((np.abs(qaoa_state(Ez, x, n)) ** 2) @ Ez), x0, method="L-BFGS-B")
            if bestr is None or r.fun < bestr.fun:
                bestr = r
        pr = np.abs(qaoa_state(Ez, bestr.x, n)) ** 2
        out["by_p"][p] = dict(P_feasible=float(pr[feasible].sum()), P_optimal=float(pr[optimal].sum()),
                              angles=bestr.x.tolist())
    return out


# ---------- приклад ----------
EXAMPLE = dict(caps=[12, 12, 12], existing=[[6, 2], [5], []], new=[3, 4])


def main(caps, existing, new):
    res, active = residual(caps, existing)
    print(f"ФС={caps}, наявні ВМ по серверах={existing}, залишки={[round(r, 1) for r in res]}, активні={active}")
    print(f"нові ВМ: {new}")
    kbf, abf = brute_force(new, res, active)
    print(f"перебір:  нових серверів = {kbf}, розміщення {abf}")
    a = best_fit_decreasing(new, res, active)
    print(f"Best Fit Decreasing: нових серверів = {new_servers(a, active)}, розміщення {a}")
    try:
        a, st = cp_sat(new, res, active)
        print(f"CP-SAT:   нових серверів = {new_servers(a, active)} ({st}), розміщення {a}")
    except ImportError:
        print("CP-SAT:   OR-Tools не встановлено, пропущено")
    t = time.time()
    q = qaoa_new(new, res, active)
    print(f"QAOA лише над новими ВМ: кубітів = {q['n_qubits']} (повна задача мала б "
          f"{(sum(len(v) for v in existing) + len(new)) * len(caps) + len(caps)}), оптимум = {q['optimum_new_servers']} новий сервер; "
          f"baseline допустимі {q['baseline_feasible']:.3f}, оптимальні {q['baseline_optimal']:.3f}")
    for p, r in q["by_p"].items():
        print(f"  p={p}: P(допустимі)={r['P_feasible']:.3f}  P(оптимум)={r['P_optimal']:.3f}")
    print(f"  [{time.time() - t:.0f} с]")


if __name__ == "__main__":
    ex = dict(EXAMPLE)
    if "--new" in sys.argv:
        ex["new"] = [float(v) for v in sys.argv[sys.argv.index("--new") + 1].split(",")]
    main(**ex)
