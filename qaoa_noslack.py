"""
QAOA БЕЗ slack-бітів (8 кубітів замість 16).

Ідея: нерівність місткості sum_j L_j x_ji <= T*C_i замінюється ТОЧНИМИ парними штрафами
за кожну пару ВМ, яка разом не влазить у ФС (працює, коли всі мінімальні порушуючі
підмножини — пари; це перевіряється assert-ом). Змінні: x_{j,i} (N*M) та y_i (M).

H = sum_i y_i                                   (кількість активних ФС)
  + lam * sum_j (sum_i x_ji - 1)^2              (кожна ВМ — на одній ФС)
  + lam * sum_{j,i} x_ji (1 - y_i)              (ВМ лише на активній ФС)
  + lam * sum_i sum_{(j,k) не влазять} x_ji x_ki (місткість)

Два інстанси:
  A  — оригінальний зменшений [3,4,2] / [12,12]: місткість НЕ обмежує (9 <= 10.8) -> штрафів місткості нема;
  B  — "binding" [6,5,2] / [12,12], T=0.9 (ліміт 10.8): пара 6+5=11 не влазить -> оптимум = 2 ФС.
"""
import itertools, sys, time
import numpy as np
from scipy.optimize import minimize
from qaoa_exact import apply_mixer, qaoa_state

T = 0.9
LAM = 3.0
INSTANCES = {"A": ([3, 4, 2], [12, 12]), "B": ([6, 5, 2], [12, 12])}


def build(loads, caps, lam=LAM):
    N, M = len(loads), len(caps)
    cap = [T * c for c in caps]
    # змінні: x_{j,i} -> j*M+i ; y_i -> N*M+i
    n = N * M + M
    xi = lambda j, i: j * M + i
    yi = lambda i: N * M + i
    # перевірка: мінімальні порушуючі підмножини — пари
    for r in range(3, N + 1):
        for S in itertools.combinations(range(N), r):
            for i in range(M):
                if sum(loads[j] for j in S) > cap[i]:
                    assert any(loads[a] + loads[b] > cap[i] for a, b in itertools.combinations(S, 2)), \
                        "є мінімальна порушуюча підмножина розміру >=3 — парні штрафи недостатні"
    S = np.arange(2 ** n)
    X = ((S[:, None] >> (n - 1 - np.arange(n))[None, :]) & 1).astype(np.int8)
    E = np.zeros(2 ** n)
    for i in range(M):
        E += X[:, yi(i)]
    for j in range(N):
        E += lam * (sum(X[:, xi(j, i)] for i in range(M)) - 1) ** 2
        for i in range(M):
            E += lam * X[:, xi(j, i)] * (1 - X[:, yi(i)])
    for i in range(M):
        for a, b in itertools.combinations(range(N), 2):
            if loads[a] + loads[b] > cap[i]:
                E += lam * X[:, xi(a, i)] * X[:, xi(b, i)]
    # класифікація (за "істинною" семантикою, не за енергією)
    xs = np.stack([np.stack([X[:, xi(j, i)] for i in range(M)], 1) for j in range(N)], 1)
    onehot = (xs.sum(2) == 1).all(1)
    assign = xs.argmax(2)
    ld = np.stack([((assign == i) * np.array(loads)[None, :]).sum(1) for i in range(M)], 1)
    capok = (ld <= np.array(cap)[None, :] + 1e-9).all(1)
    linked = np.all(np.stack([(xs[:, :, i].sum(1) == 0) | (X[:, yi(i)] == 1) for i in range(M)], 1), 1)
    feasible = onehot & capok & linked
    obj = X[:, N * M:].sum(1)
    best = obj[feasible].min()
    optimal = feasible & (obj == best)
    return n, E, onehot, feasible, optimal, best


def run(name, ps=(1, 2, 3, 4, 5), restarts=30, seed=0):
    loads, caps = INSTANCES[name]
    n, E, onehot, feasible, optimal, best = build(loads, caps)
    gs = E.min(); gm = np.isclose(E, gs)
    print(f"\n=== Інстанс {name}: VM={loads}, PM={caps}, кубітів={n}, оптимум={best} ФС ===")
    print(f"baseline uniform: one-hot {onehot.mean():.3f} feasible {feasible.mean():.3f} optimal {optimal.mean():.4f}")
    print(f"ground energy {gs:.2f}, виродженість {gm.sum()}, ground=optimal: {optimal[gm].all()}")
    Ez = (E - E.min()) / (E.max() - E.min())
    rng = np.random.default_rng(seed)
    for p in ps:
        bestres = None
        for _ in range(restarts):
            x0 = np.concatenate([rng.uniform(0, np.pi / 2, p), rng.uniform(0, 2 * np.pi, p)])
            f = lambda x: float((np.abs(qaoa_state(Ez, x, n)) ** 2) @ Ez)
            r = minimize(f, x0, method="L-BFGS-B")
            if bestres is None or r.fun < bestres.fun:
                bestres = r
        pr = np.abs(qaoa_state(Ez, bestres.x, n)) ** 2
        print(f"p={p}: P(one-hot)={pr[onehot].sum():.3f} P(feasible)={pr[feasible].sum():.3f} "
              f"P(optimal)={pr[optimal].sum():.3f} P(ground)={pr[gm].sum():.3f}")
        sys.stdout.flush()


if __name__ == "__main__":
    for name in (sys.argv[1:] or ["A", "B"]):
        t = time.time(); run(name); print(f"[{time.time()-t:.0f}s]")
