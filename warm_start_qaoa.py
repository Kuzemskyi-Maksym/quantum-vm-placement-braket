"""
Прототип ідеї Жарікова: QAOA зі СТАРТОМ з відомого розподілу (warm start), точна симуляція.

Старт: |x0> — поточний розподіл (ВМ розкинуті по трьох ФС, активних 3).
Схема: M(b0) -> [P(g_l) -> M(b_l)] x p, де P — фаза exp(-i g Ez) (енергія задачі),
M — змішувач RX(2b) на кожному кубіті. Перший змішувач потрібен, бо фаза на чистому
базисному стані — лише глобальна фаза (без нього перший шар P нічого б не дав).
Порівнюємо з холодним стартом (|+>^n) і з класичним випадковим пошуком сусідів.
Інстанс W (config, VMP_INSTANCE=W): ВМ [6,5,2,1], ФС [12,12,12] (T=0.9, ліміт 10.8), 15 кубітів,
оптимум = 2 ФС, x0 = [0,1,2,2].

Критерій міграцій (рішення Жарікова 2026-11-01): F(x) = активні ФС + MU · Σ s_j·[ВМ j переїхала] / Σ s_j.
У гамільтоніан член міграцій входить як ЛІНІЙНІ поля на кубітах x_{j, x0[j]} — нових кубітів не потрібно.
«Кращий за x0» означає F(x) < F(x0) серед допустимих. MU = 0 відтворює стару постановку (лише ФС).

Запуск:
  python warm_start_qaoa.py                 # стара таблиця (середня енергія), MU з config/VMP_MU
  python warm_start_qaoa.py pbetter         # ціль «максимум P(кращий за x0)»
  python warm_start_qaoa.py pbetter --mu 1  # те саме з явним MU
  python warm_start_qaoa.py pbetter --mu 1 --mixer size --alpha 2   # змішувач, зважений за розміром ВМ

Змішувач за розміром ВМ (2026-11-02): кубіти ВМ j обертаються на кут β·(s_min/s_j)^alpha, тож малі ВМ
«рухливіші». На W це подвоює P(кращий за x0) (6% → 12% при p=2) і дає оптимум з двома дешевими міграціями
замість однієї дорогої, яку обирає звичайний RX.
"""
import os, sys, time
os.environ.setdefault("VMP_INSTANCE", "W")
import numpy as np
from scipy.optimize import minimize
from qaoa_noslack import build
from qaoa_exact import apply_mixer, qaoa_state
from config import VM_LOADS as LOADS, PM_CAPACITY as CAPS, X0 as X0_ASSIGN, MU, TOTAL_LOAD

N, M = len(LOADS), len(CAPS)
assert X0_ASSIGN is not None, "для warm start потрібен поточний розподіл X0 (VMP_INSTANCE=W або VMP_X0=...)"


def basis_index(assign, n):
    bits = [0] * n
    for j, i in enumerate(assign):
        bits[j * M + i] = 1
    for i in set(assign):
        bits[N * M + i] = 1
    idx = 0
    for b in bits:
        idx = (idx << 1) | b
    return idx


def decode_states(n):
    S = np.arange(2 ** n)
    X = ((S[:, None] >> (n - 1 - np.arange(n))[None, :]) & 1).astype(np.int8)
    xs = np.stack([X[:, j * M:(j + 1) * M] for j in range(N)], 1)
    assign = xs.argmax(2)
    active = X[:, N * M:].sum(1)
    return X, assign, active


def size_weights(alpha=1.0):
    """Ваги змішувача за розміром ВМ: кубіт x_{j,i} обертається на кут β·(s_min/s_j)^alpha — малі ВМ
    «рухливіші», великі майже не рухаються; кубіти y_i (сервери) — вага 1. alpha=0 → звичайний RX."""
    smin = min(LOADS)
    w = np.ones(N * M + M)
    for j in range(N):
        w[j * M:(j + 1) * M] = (smin / LOADS[j]) ** alpha
    return w


def apply_mixer_w(psi, beta, w, n):
    """RX(2·β·w_k) на кожному кубіті k (поелементно різні кути)."""
    psi = psi.reshape((2,) * n)
    for ax in range(n):
        c, s = np.cos(beta * w[ax]), -1j * np.sin(beta * w[ax])
        a0 = np.take(psi, 0, axis=ax)
        a1 = np.take(psi, 1, axis=ax)
        psi = np.stack([c * a0 + s * a1, s * a0 + c * a1], axis=ax)
    return psi.reshape(-1)


def warm_state(Ez, params, n, idx0, w=None):
    """Схема теплого старту: |x0> → M(β0) → [P(γ_l) → M(β_l)] × p. w — ваги змішувача (None = звичайний RX)."""
    p = (len(params) - 1) // 2
    b0, gs, bs = params[0], params[1:1 + p], params[1 + p:]
    mix = (lambda ps, b: apply_mixer(ps, b, n)) if w is None else (lambda ps, b: apply_mixer_w(ps, b, w, n))
    psi = np.zeros(2 ** n, dtype=complex)
    psi[idx0] = 1.0
    psi = mix(psi, b0)
    for l in range(p):
        psi = psi * np.exp(-1j * gs[l] * Ez)
        psi = mix(psi, bs[l])
    return psi


_BUILD_CACHE = {}


def problem(mu, x0_assign=None):
    """Інстанс (config) з членом міграцій. Повертає все потрібне для оцінки станів.
    x0_assign — інший поточний розподіл (compare_migration --random); None → X0 з config."""
    x0_assign = X0_ASSIGN if x0_assign is None else list(x0_assign)
    if "b" not in _BUILD_CACHE:                     # build/decode не залежать від x0 і mu — кешуємо
        n_ = N * M + M
        _BUILD_CACHE["b"] = build(LOADS, CAPS), decode_states(n_)
    (n, E, onehot, feasible, optimal, best), (X, assign, active) = _BUILD_CACHE["b"]
    x0 = np.array(x0_assign)
    idx0 = basis_index(x0_assign, n)
    moved = assign != x0[None, :]
    mig = moved.sum(1)                                                     # кількість міграцій
    migc = (moved * np.array(LOADS, dtype=float)[None, :]).sum(1) / TOTAL_LOAD   # зважена вартість ∈ [0,1]
    # член міграцій як лінійні поля: mu·s_j/Σs · (1 − x_{j,x0[j]})
    Emig = sum(mu * LOADS[j] / TOTAL_LOAD * (1 - X[:, j * M + x0[j]]) for j in range(N))
    Etot = E + Emig
    F = active + mu * migc                                                 # «істинна» ціль для допустимих станів
    F0 = F[idx0]
    better = feasible & (F < F0 - 1e-9)
    Ez = (Etot - Etot.min()) / (Etot.max() - Etot.min())
    return dict(n=n, Ez=Ez, E=Etot, onehot=onehot, feasible=feasible, optimal=optimal, best=best,
                assign=assign, active=active, mig=mig, migc=migc, F=F, F0=F0, better=better, idx0=idx0)


def report(tag, pr, P):
    ok = P["onehot"] & P["feasible"]
    em = (pr[ok] * P["mig"][ok]).sum() / max(pr[ok].sum(), 1e-12)
    bt = P["better"]
    emb = (pr[bt] * P["migc"][bt]).sum() / max(pr[bt].sum(), 1e-12)
    print(f"{tag:34s} one-hot {pr[P['onehot']].sum():.3f} | допуст. {pr[P['feasible']].sum():.3f} | "
          f"кращі за x0 {pr[bt].sum():.3f} | оптимум(ФС) {pr[P['optimal']].sum():.4f} | "
          f"сер. міграцій (допуст.) {em:.2f} | сер. варт. мігр. (кращі) {emb:.2f}")


def best_seen(pr, P, shots=1000):
    """Найкращий за F допустимий стан серед тих, що з'явилися б за `shots` вимірювань (pr ≥ 1/shots)."""
    cand = P["feasible"] & (pr >= 1.0 / shots)
    if not cand.any():
        return None
    i = np.flatnonzero(cand)[np.argmin(P["F"][cand])]
    return dict(assignment=P["assign"][i].tolist(), active_pms=int(P["active"][i]),
                migrations=int(P["mig"][i]), migration_cost=float(P["migc"][i]), objective=float(P["F"][i]),
                prob=float(pr[i]))


GMAX = 20 * np.pi   # межа для γ: енергія нормована в [0,1], тому різниці енергій сусідніх станів ~1/20,
                    # і корисні γ — десятки; межа 2π (стара) відсікала їх (γ упиралось у 6.28)


def optimize_warm(P, p, bmax, restarts, rng, target="pbetter", w=None, fast=False):
    """Підбір кутів теплого старту. target: 'pbetter' — максимум P(кращий за x0); 'energy' — мінімум <E>.
    w — ваги змішувача (size_weights) або None. fast=True — numba-ядро qaoa_fast.warm_state (той самий стан)."""
    n, Ez, idx0 = P["n"], P["Ez"], P["idx0"]
    bounds = [(0, bmax)] + [(0, GMAX)] * p + [(0, bmax)] * p
    if fast:
        import qaoa_fast as qf
        Ezc = np.ascontiguousarray(Ez)
        state = lambda x: qf.warm_state(Ezc, x, n, idx0, w)
    else:
        state = lambda x: warm_state(Ez, x, n, idx0, w)
    if target == "pbetter":
        f = lambda x: -float((np.abs(state(x)) ** 2)[P["better"]].sum())
    else:
        f = lambda x: float((np.abs(state(x)) ** 2) @ Ez)
    best = None
    for _ in range(restarts):
        x = np.concatenate([[rng.uniform(0, bmax)], rng.uniform(0, GMAX, p), rng.uniform(0, bmax, p)])
        r = minimize(f, x, method="L-BFGS-B", bounds=bounds)
        if best is None or r.fun < best.fun:
            best = r
    return best


def run_warm(mu=MU, p=2, restarts=25, seed=1, bmax=np.pi, shots=1000, mixer="rx", alpha=1.0, x0=None, fast=False):
    """Для порівняльної таблиці: теплий старт з ціллю P(кращий за x0) при заданому mu.
    mixer: 'rx' — звичайний RX на всіх кубітах; 'size' — кути, зважені за розміром ВМ (size_weights(alpha)).
    x0 — інший поточний розподіл (None → config.X0); fast — numba-ядро."""
    P = problem(mu, x0)
    w = size_weights(alpha) if mixer == "size" else None
    rng = np.random.default_rng(seed)
    t = time.time()
    r = optimize_warm(P, p, bmax, restarts, rng, w=w, fast=fast)
    pr = np.abs(warm_state(P["Ez"], r.x, P["n"], P["idx0"], w)) ** 2
    bt = P["better"]
    return dict(mu=mu, p=p, mixer=mixer, F0=float(P["F0"]), p_better=float(pr[bt].sum()),
                p_feasible=float(pr[P["feasible"]].sum()),
                mean_migc_better=float((pr[bt] * P["migc"][bt]).sum() / max(pr[bt].sum(), 1e-12)),
                best_seen=best_seen(pr, P, shots), angles=r.x.tolist(), runtime_s=time.time() - t)


def random_neighbor(mu, d, x0=None):
    """Класичний контроль: рівноімовірний допустимий розподіл на відстані d міграцій від x0."""
    P = problem(mu, x0)
    sel = P["onehot"] & P["feasible"] & (P["mig"] == d)
    pr = np.zeros(2 ** P["n"]); pr[sel] = 1 / max(sel.sum(), 1)
    bt = P["better"]
    return dict(mu=mu, d=d, p_better=float(pr[bt].sum()),
                mean_migc_better=float((pr[bt] * P["migc"][bt]).sum() / max(pr[bt].sum(), 1e-12)))


def main(ps=(1, 2, 3), restarts=25, seed=0, mu=MU):
    t0 = time.time()
    P = problem(mu)
    n, Ez, idx0 = P["n"], P["Ez"], P["idx0"]
    print(f"Інстанс W: ВМ={LOADS}, ФС={CAPS}, кубітів={n}, оптимум={P['best']} ФС; старт x0={X0_ASSIGN} "
          f"(3 активні ФС), MU={mu}, F(x0)={P['F0']:.3f}; feasible x0: {bool(P['feasible'][idx0])}")
    uni = np.full(2 ** n, 1 / 2 ** n)
    report("випадково (рівномірно, всі стани)", uni, P)
    for d in (1, 2):
        sel = P["onehot"] & P["feasible"] & (P["mig"] == d)
        pr = np.zeros(2 ** n); pr[sel] = 1 / sel.sum()
        report(f"класично: випадковий допуст. d={d}", pr, P)

    rng = np.random.default_rng(seed)
    for p in ps:
        best_c = None
        for _ in range(restarts):
            x = np.concatenate([rng.uniform(0, np.pi / 2, p), rng.uniform(0, 2 * np.pi, p)])
            r = minimize(lambda x: float((np.abs(qaoa_state(Ez, x, n)) ** 2) @ Ez), x, method="L-BFGS-B")
            if best_c is None or r.fun < best_c.fun: best_c = r
        report(f"QAOA холодний старт p={p}", np.abs(qaoa_state(Ez, best_c.x, n)) ** 2, P)
        for tag, bmax in (("вільні кути", np.pi), ("β ≤ 0.35 (близько до x0)", 0.35)):
            r = optimize_warm(P, p, bmax, restarts, rng, target="energy")
            report(f"QAOA теплий старт p={p}, {tag}", np.abs(warm_state(Ez, r.x, n, idx0)) ** 2, P)
        print(f"  [{time.time()-t0:.0f}s]")


def main_pbetter(ps=(1, 2), restarts=25, seed=1, mu=MU, mixer="rx", alpha=1.0):
    """Теплий старт з ЦІЛЛЮ «максимум P(кращий за x0)» замість мінімуму середньої енергії:
    середня енергія карає будь-який відхід від x0 (більшість сусідів — штрафні стани), тому
    оптимізатор залишається в x0 (β=0). Для циклу «виміряв -> перевірив -> прийняв» важить
    саме ймовірність отримати кращий розподіл за shot."""
    t0 = time.time()
    P = problem(mu)
    w = size_weights(alpha) if mixer == "size" else None
    rng = np.random.default_rng(seed)
    print(f"\n--- мета: максимум P(кращий за x0), MU={mu}, F(x0)={P['F0']:.3f}, змішувач={mixer}"
          + (f" (alpha={alpha})" if w is not None else "") + " ---")
    for p in ps:
        for tag, bmax in (("вільні кути", np.pi), ("β ≤ 0.35", 0.35)):
            r = optimize_warm(P, p, bmax, restarts, rng, w=w)
            pr = np.abs(warm_state(P["Ez"], r.x, P["n"], P["idx0"], w)) ** 2
            report(f"теплий старт p={p}, {tag}", pr, P)
            print(f"     кути: β0={r.x[0]:.3f}, γ={np.round(r.x[1:1+p],2)}, β={np.round(r.x[1+p:],3)}")
            bs = best_seen(pr, P)
            if bs:
                print(f"     найкращий стан за 1000 shots: {bs['assignment']} → ФС={bs['active_pms']}, "
                      f"міграцій={bs['migrations']} (варт. {bs['migration_cost']:.2f}), F={bs['objective']:.3f}")
        print(f"  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    arg = lambda k, d: type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
    mu, mixer, alpha = arg("--mu", MU), arg("--mixer", "rx"), arg("--alpha", 1.0)
    main_pbetter(mu=mu, mixer=mixer, alpha=alpha) if "pbetter" in sys.argv else main(mu=mu)
