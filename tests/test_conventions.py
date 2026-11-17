"""
Тести конвенцій (ціль з міграціями, декодер BRKGA, slack-free QUBO, порядок бітів у numpy/Qiskit/Aer).
Конвенцію бітів уже раз ламали: у numpy-станах кубіт 0 — СТАРШИЙ біт індексу; Qiskit-counts перевертаються
(int(bs[::-1], 2)), Braket-counts — ні. Повільні тести позначено `slow`:  pytest -m "not slow".
Запуск з кореня репозиторію:  python -m pytest tests -q
"""
import os, sys
os.environ["VMP_INSTANCE"] = "W"            # config читає інстанс під час імпорту
os.environ.pop("VMP_X0", None)
os.environ.pop("VMP_MU", None)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

import config


def test_objective_migrations_W():
    # (а) переїхали ВМ 2 і 3 (розміри 2 і 1 із Σ=14), активні ФС {0, 1}
    assert config.objective([0, 1, 0, 0], x0=[0, 1, 2, 2], mu=1) == pytest.approx(2 + 3 / 14)
    assert config.objective([0, 1, 2, 2], x0=[0, 1, 2, 2], mu=1) == pytest.approx(3)


def test_brkga_decode_ffd_W():
    # (б) ключі в порядку First Fit Decreasing → 2 ФС (оптимум W)
    import brkga
    loads = config.VM_LOADS
    order = sorted(range(len(loads)), key=lambda j: -loads[j])
    keys = [0.0] * len(loads)
    for r, j in enumerate(order):
        keys[j] = r / len(loads)
    a = brkga.decode(keys)
    assert config.is_feasible(a) and config.active_pms(a) == 2


def _decode_legacy(keys, loads, caps):
    # декодер до виправлення 2026-11-17 (без x0): First Fit за індексом сервера
    cur, a = [0.0] * len(caps), [None] * len(loads)
    for j in sorted(range(len(loads)), key=lambda j: keys[j]):
        t = next((i for i in range(len(caps)) if cur[i] + loads[j] <= caps[i] + 1e-9), None)
        if t is None:
            t = min(range(len(caps)), key=lambda i: cur[i] - caps[i])
        a[j] = t
        cur[t] += loads[j]
    return a


def test_brkga_decode_without_x0_unchanged():
    # без x0 декодер (а отже run_brkga(x0=None) і стратегія repack) відтворює старі результати 1:1
    import brkga
    rng = np.random.default_rng(7)
    for _ in range(300):
        n, m = int(rng.integers(3, 12)), int(rng.integers(2, 6))
        loads = rng.integers(1, 9, n).tolist()
        caps = [float(c) for c in rng.integers(8, 15, m)]
        keys = rng.random(2 * n).tolist()            # навіть з 2N ключами: без x0 прапорці ігноруються
        assert brkga.decode(keys, None, loads, caps) == _decode_legacy(keys, loads, caps)


def test_brkga_migration_reaches_active_server_W():
    # баг до 2026-11-17: x0=[1,2,1,0] → оптимум [1,2,1,1] (ВМ 3 на вже активний ФС 1) був недосяжний
    import brkga
    x0 = [1, 2, 1, 0]
    keys = [0.0, 0.1, 0.2, 0.3] + [1.0, 1.0, 1.0, 0.0]    # ВМ 0–2 лишаються, ВМ 3 рухається останньою
    assert brkga.decode(keys, x0) == [1, 2, 1, 1]
    runs = [brkga.run_brkga(seed=s, x0=x0, mu=1.0) for s in range(5)]
    assert all(r["objective"] == pytest.approx(config.objective([1, 2, 1, 1], x0=x0, mu=1.0)) for r in runs)


def test_noslack_build_C():
    # (в) інстанс C: оптимум 2 ФС, частка оптимальних станів 1/128
    from qaoa_noslack import build, INSTANCES
    n, E, onehot, feasible, optimal, best = build(*INSTANCES["C"])
    assert n == 10 and best == 2
    assert optimal.mean() == pytest.approx(1 / 128)
    assert optimal[np.isclose(E, E.min())].all()           # основний стан = оптимум


def test_basis_index_decode_W():
    # (г) basis_index ↔ decode_states узгоджені для x0 інстансу W
    import warm_start_qaoa as ws
    n = ws.N * ws.M + ws.M
    X, assign, active = ws.decode_states(n)
    idx = ws.basis_index(config.X0, n)
    assert assign[idx].tolist() == list(config.X0)
    assert active[idx] == len(set(config.X0))
    # кубіт 0 — старший біт: x_{0, x0[0]} = x_{0,0} → біт n−1 індексу
    assert (idx >> (n - 1)) & 1 == 1


def test_fast_kernels_match_numpy():
    # qaoa_fast (numba) = qaoa_exact / warm_start_qaoa
    import qaoa_fast as qf
    from qaoa_exact import qaoa_state
    from warm_start_qaoa import warm_state
    rng = np.random.default_rng(3)
    n = 11
    Ez = rng.random(2 ** n)
    x = rng.uniform(0, 3, 4)
    assert np.allclose(qf.qaoa_state(Ez, x, n), qaoa_state(Ez, x, n))
    w = rng.uniform(0.2, 1.0, n); x = rng.uniform(0, 3, 5)
    assert np.allclose(qf.warm_state(Ez, x, n, 37, w), warm_state(Ez, x, n, 37, w))


def test_noisy_scaling_conventions_C():
    # аналітичні Ising-коефіцієнти + Aer save_probabilities (після маршрутизації на решітці) = точний numpy
    os.environ["QAOA_INSTANCE"] = "C"
    import qaoa_noslack_noisy as nz
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    from qaoa_exact import qaoa_state
    S = nz.setup("C")
    x = np.array([0.4, 0.2, 7.0, 15.0])
    qc = nz.circuit(x, S["terms"], S["n"]); qc.remove_final_measurements()
    tqc = transpile(qc, basis_gates=["rz", "sx", "x", "cx"], coupling_map=nz.grid_map(*nz.grid_shape(S["n"])),
                    optimization_level=1, seed_transpiler=1)
    pr = np.asarray(AerSimulator(method="statevector").run(nz.with_saved_probs(tqc, S["n"]), shots=1)
                    .result().data()["probabilities"])
    assert np.allclose(pr, np.abs(qaoa_state(S["Ez"], x, S["n"])) ** 2, atol=1e-9)
    idx = np.arange(2 ** S["n"])
    fe, op = nz.classify_idx(idx, S["loads"], S["caps"], S["best"])
    assert (fe == S["feasible"]).all() and (op == S["optimal"]).all()


@pytest.mark.slow
def test_warm_noisy_circuit_matches_numpy():
    # (д) Qiskit-схема warm_start_noisy.circuit без шуму, 20000 shots: P(кращі) = numpy ± 0.01
    import warm_start_noisy as wn
    import warm_start_qaoa as ws
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    r = ws.run_warm(mu=wn.MU, p=2, restarts=5, mixer=wn.MIXER, alpha=wn.ALPHA, fast=True)   # p=2: P(кращі) ≈ 6%
    x = np.array(r["angles"])
    exact = np.abs(ws.warm_state(wn.Ez, x, wn.n, wn.idx0, None if wn.MIXER == "rx" else wn.w)) ** 2
    tqc = transpile(wn.circuit(x), basis_gates=["rz", "sx", "x", "cx"], optimization_level=1, seed_transpiler=1)
    shots = 20000
    pr = wn.probs_from_counts(AerSimulator(seed_simulator=5).run(tqc, shots=shots).result().get_counts(), shots)
    bt = wn.P["better"]
    assert exact[bt].sum() > 0.02                               # тест має сенс лише за ненульового сигналу
    assert abs(pr[bt].sum() - exact[bt].sum()) < 0.01
