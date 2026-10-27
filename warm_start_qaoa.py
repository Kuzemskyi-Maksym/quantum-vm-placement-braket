"""
Прототип ідеї Жарікова: QAOA зі СТАРТОМ з відомого розподілу (warm start), точна симуляція.

Старт: |x0> — поточний розподіл (ВМ розкинуті по трьох ФС, активних 3).
Схема: M(b0) -> [P(g_l) -> M(b_l)] x p, де P — фаза exp(-i g Ez) (енергія задачі),
M — змішувач RX(2b) на кожному кубіті. Перший змішувач потрібен, бо фаза на чистому
базисному стані — лише глобальна фаза (без нього перший шар P нічого б не дав).
Порівнюємо з холодним стартом (|+>^n) і з класичним випадковим пошуком сусідів.
Інстанс W: ВМ [6,5,2,1], ФС [12,12,12] (T=0.9, ліміт 10.8), 15 кубітів, оптимум = 2 ФС.
Запуск: python warm_start_qaoa.py
"""
import itertools, time
import numpy as np
from scipy.optimize import minimize
from qaoa_noslack import build
from qaoa_exact import apply_mixer, qaoa_state

LOADS, CAPS = [6, 5, 2, 1], [12, 12, 12]
N, M = len(LOADS), len(CAPS)
X0_ASSIGN = [0, 1, 2, 2]          # поточний розподіл: ВМ0->ФС0, ВМ1->ФС1, ВМ2,ВМ3->ФС2 (3 активні ФС)


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
    return assign, active


def warm_state(Ez, params, n, idx0):
    p = (len(params) - 1) // 2
    b0, gs, bs = params[0], params[1:1 + p], params[1 + p:]
    psi = np.zeros(2 ** n, dtype=complex)
    psi[idx0] = 1.0
    psi = apply_mixer(psi, b0, n)
    for l in range(p):
        psi = psi * np.exp(-1j * gs[l] * Ez)
        psi = apply_mixer(psi, bs[l], n)
    return psi


def report(tag, pr, onehot, feasible, optimal, better, mig):
    ok = onehot & feasible
    em = (pr[ok] * mig[ok]).sum() / max(pr[ok].sum(), 1e-12)
    print(f"{tag:34s} one-hot {pr[onehot].sum():.3f} | допуст. {pr[feasible].sum():.3f} | "
          f"кращі за x0 {pr[better].sum():.3f} | оптимум {pr[optimal].sum():.4f} | "
          f"сер. міграцій (допуст.) {em:.2f}")


def main(ps=(1, 2, 3), restarts=25, seed=0):
    t0 = time.time()
    n, E, onehot, feasible, optimal, best = build(LOADS, CAPS)
    assign, active = decode_states(n)
    idx0 = basis_index(X0_ASSIGN, n)
    mig = (assign != np.array(X0_ASSIGN)[None, :]).sum(1)
    better = feasible & (active < 3)
    Ez = (E - E.min()) / (E.max() - E.min())
    print(f"Інстанс W: ВМ={LOADS}, ФС={CAPS}, кубітів={n}, оптимум={best} ФС; старт x0={X0_ASSIGN} (3 активні ФС, "
          f"енергія x0 = {E[idx0]:.2f}, мінімум = {E.min():.2f}); feasible x0: {bool(feasible[idx0])}")
    uni = np.full(2 ** n, 1 / 2 ** n)
    report("випадково (рівномірно, всі стани)", uni, onehot, feasible, optimal, better, mig)

    # класичний випадковий сусід: усі допустимі розподіли на відстані d від x0, рівноімовірно
    for d in (1, 2):
        sel = onehot & feasible & (mig == d)
        pr = np.zeros(2 ** n); pr[sel] = 1 / sel.sum()
        report(f"класично: випадковий допуст. d={d}", pr, onehot, feasible, optimal, better, mig)

    rng = np.random.default_rng(seed)
    for p in ps:
        # холодний старт, та сама глибина
        best_c = None
        for _ in range(restarts):
            x = np.concatenate([rng.uniform(0, np.pi / 2, p), rng.uniform(0, 2 * np.pi, p)])
            r = minimize(lambda x: float((np.abs(qaoa_state(Ez, x, n)) ** 2) @ Ez), x, method="L-BFGS-B")
            if best_c is None or r.fun < best_c.fun: best_c = r
        pr = np.abs(qaoa_state(Ez, best_c.x, n)) ** 2
        report(f"QAOA холодний старт p={p}", pr, onehot, feasible, optimal, better, mig)

        # теплий старт: вільні кути, і з обмеженням малого β (близько до x0)
        for tag, bmax in (("вільні кути", np.pi), ("β ≤ 0.35 (близько до x0)", 0.35)):
            bounds = [(0, bmax)] + [(0, 2 * np.pi)] * p + [(0, bmax)] * p
            best_w = None
            for _ in range(restarts):
                x = np.concatenate([[rng.uniform(0, bmax)], rng.uniform(0, 2 * np.pi, p), rng.uniform(0, bmax, p)])
                f = lambda x: float((np.abs(warm_state(Ez, x, n, idx0)) ** 2) @ Ez)
                r = minimize(f, x, method="L-BFGS-B", bounds=bounds)
                if best_w is None or r.fun < best_w.fun: best_w = r
            pr = np.abs(warm_state(Ez, best_w.x, n, idx0)) ** 2
            report(f"QAOA теплий старт p={p}, {tag}", pr, onehot, feasible, optimal, better, mig)
        print(f"  [{time.time()-t0:.0f}s]")


def main_pbetter(ps=(1, 2), restarts=25, seed=1):
    """Теплий старт з ЦІЛЛЮ «максимум P(кращий за x0)» замість мінімуму середньої енергії:
    середня енергія карає будь-який відхід від x0 (більшість сусідів — штрафні стани), тому
    оптимізатор залишається в x0 (β=0). Для циклу «виміряв -> перевірив -> прийняв» важить
    саме ймовірність отримати кращий розподіл за shot."""
    t0 = time.time()
    n, E, onehot, feasible, optimal, best = build(LOADS, CAPS)
    assign, active = decode_states(n)
    idx0 = basis_index(X0_ASSIGN, n)
    mig = (assign != np.array(X0_ASSIGN)[None, :]).sum(1)
    better = feasible & (active < 3)
    Ez = (E - E.min()) / (E.max() - E.min())
    rng = np.random.default_rng(seed)
    print("\n--- мета: максимум P(кращий за x0) ---")
    for p in ps:
        for tag, bmax in (("вільні кути", np.pi), ("β ≤ 0.35", 0.35)):
            bounds = [(0, bmax)] + [(0, 2 * np.pi)] * p + [(0, bmax)] * p
            best_w = None
            for _ in range(restarts):
                x = np.concatenate([[rng.uniform(0, bmax)], rng.uniform(0, 2 * np.pi, p), rng.uniform(0, bmax, p)])
                f = lambda x: -float((np.abs(warm_state(Ez, x, n, idx0)) ** 2)[better].sum())
                r = minimize(f, x, method="L-BFGS-B", bounds=bounds)
                if best_w is None or r.fun < best_w.fun: best_w = r
            pr = np.abs(warm_state(Ez, best_w.x, n, idx0)) ** 2
            report(f"теплий старт p={p}, {tag}", pr, onehot, feasible, optimal, better, mig)
            print(f"     кути: β0={best_w.x[0]:.3f}, γ={np.round(best_w.x[1:1+p],2)}, β={np.round(best_w.x[1+p:],3)}")
        print(f"  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    import sys
    main_pbetter() if "pbetter" in sys.argv else main()
