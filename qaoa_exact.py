"""
Діагностика і коректне навчання QAOA на зменшеному інстансі (3 ВМ, 2 ФС, 16 кубітів).

Що робить:
  1. Перебором усіх 2^16 станів знаходить основний стан BQM і перевіряє, чи він є
     допустимим розв'язком (чи достатні штрафи λ).
  2. Точно (без shot-шуму) симулює QAOA на numpy і оптимізує кути ПО ТОЧНІЙ ОЧІКУВАНІЙ
     ЕНЕРГІЇ, з нормуванням гамільтоніана (кути γ тоді одного порядку) і багатьма рестартами.
  3. Міряє те, що важливо: ймовірність допустимих / оптимальних розстановок і
     співвідношення апроксимації — проти рівномірного baseline (12.5% / 3.1%).

Конвенція бітів: стан базису |b_0 ... b_15>, біт = значення бінарної змінної BQM.
"""

import sys
import time
import numpy as np
from scipy.optimize import minimize

from cqm_model import build_bqm
from config import T, LAMBDA1
from qaoa import REDUCED_VM_LOADS, REDUCED_PM_CAPACITY, REDUCED_N, REDUCED_M

LAMBDA = LAMBDA1


def build_problem(lagrange=LAMBDA):
    bqm, _, _ = build_bqm(lagrange_multiplier=lagrange, vm_loads=REDUCED_VM_LOADS,
                          pm_capacity=REDUCED_PM_CAPACITY, t=T)
    vo = list(bqm.variables)
    n = len(vo)
    idx = np.arange(2 ** n)
    # bit k (змінна vo[k]) = (idx >> (n-1-k)) & 1  -> змінна 0 найстарший біт
    X = ((idx[:, None] >> (n - 1 - np.arange(n))[None, :]) & 1).astype(np.int8)
    E = np.asarray(bqm.energies((X, vo)), dtype=float)
    return bqm, vo, X, E


def classify(X, vo):
    """Для кожного стану: чи x-частина — коректна one-hot розстановка; чи допустима; скільки ФС."""
    real = {v: k for k, v in enumerate(vo)}
    N, M = REDUCED_N, REDUCED_M
    xs = np.stack([np.stack([X[:, real[f"x_{j}_{i}"]] for i in range(M)], 1) for j in range(N)], 1)  # (S,N,M)
    onehot = (xs.sum(2) == 1).all(1)
    assign = xs.argmax(2)  # (S,N)
    loads = np.zeros((X.shape[0], M))
    for j in range(N):
        loads[np.arange(X.shape[0]), :] += 0
    for i in range(M):
        loads[:, i] = ((assign == i) * np.array(REDUCED_VM_LOADS)[None, :]).sum(1)
    cap = np.array([T * c for c in REDUCED_PM_CAPACITY])
    feasible = onehot & (loads <= cap[None, :] + 1e-9).all(1)
    active = np.array([len(set(a)) for a in assign])
    optimal = feasible & (active == 1)
    return onehot, feasible, optimal


def apply_mixer(psi, beta, n):
    c, s = np.cos(beta), -1j * np.sin(beta)
    psi = psi.reshape((2,) * n)
    for ax in range(n):
        a0 = np.take(psi, 0, axis=ax)
        a1 = np.take(psi, 1, axis=ax)
        psi = np.stack([c * a0 + s * a1, s * a0 + c * a1], axis=ax)
    return psi.reshape(-1)


def qaoa_state(Ez, params, n):
    p = len(params) // 2
    betas, gammas = params[:p], params[p:]
    psi = np.full(2 ** n, 2 ** (-n / 2), dtype=complex)
    for l in range(p):
        psi = psi * np.exp(-1j * gammas[l] * Ez)
        psi = apply_mixer(psi, betas[l], n)
    return psi


def main(ps=(1, 2, 3, 4), restarts=12, seed=0):
    t0 = time.time()
    bqm, vo, X, E = build_problem()
    n = len(vo)
    onehot, feasible, optimal = classify(X, vo)
    print(f"qubits={n}  states={len(E)}")
    print(f"baseline uniform: one-hot {onehot.mean():.4f}  feasible {feasible.mean():.4f}  optimal {optimal.mean():.4f}")

    gs = E.min()
    gs_mask = np.isclose(E, gs)
    print(f"ground energy {gs:.3f}, degenerate states: {gs_mask.sum()}")
    print(f"  ground state(s) are: one-hot {onehot[gs_mask].all()}, feasible {feasible[gs_mask].all()}, optimal(1 PM) {optimal[gs_mask].all()}")
    print(f"  best energy among feasible: {E[feasible].min():.3f}, among optimal: {E[optimal].min():.3f}, among NON-feasible: {E[~feasible].min():.3f}")
    # скільки станів енергетично нижче за найкращий оптимальний
    below = (E < E[optimal].min() - 1e-9)
    print(f"  states with energy below best optimal: {below.sum()}  (of which feasible: {(below & feasible).sum()})")

    Emin, Emax = E.min(), E.max()
    Ez = (E - Emin) / (Emax - Emin)          # нормування в [0,1]
    mean_E = E.mean()
    rng = np.random.default_rng(seed)
    results = {}
    for p in ps:
        best = None
        for r in range(restarts):
            x0 = np.concatenate([rng.uniform(0, np.pi / 2, p), rng.uniform(0, 2 * np.pi, p)])
            f = lambda x: float((np.abs(qaoa_state(Ez, x, n)) ** 2) @ Ez)
            res = minimize(f, x0, method="L-BFGS-B")
            if best is None or res.fun < best.fun:
                best = res
        prob = np.abs(qaoa_state(Ez, best.x, n)) ** 2
        mean_energy = prob @ E
        ar = (E.max() - mean_energy) / (E.max() - gs)
        results[p] = dict(x=best.x, P_onehot=prob[onehot].sum(), P_feasible=prob[feasible].sum(),
                          P_optimal=prob[optimal].sum(), P_ground=prob[gs_mask].sum(), AR=ar)
        r_ = results[p]
        print(f"p={p}: P(one-hot)={r_['P_onehot']:.3f}  P(feasible)={r_['P_feasible']:.3f}  "
              f"P(optimal 1PM)={r_['P_optimal']:.3f}  P(ground)={r_['P_ground']:.4f}  AR={r_['AR']:.3f}   "
              f"[{time.time()-t0:.0f}s]")
        sys.stdout.flush()
    return bqm, vo, E, Ez, results


if __name__ == "__main__":
    main()
