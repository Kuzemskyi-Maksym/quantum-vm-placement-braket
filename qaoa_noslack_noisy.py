"""
Slack-free QAOA на Aer З ШУМОМ + топологія: решітка (як IQM Garnet/Emerald) або повна зв'язність (як IonQ Forte).
Кути беруться з точної noiseless-оптимізації (середня енергія). Шум — деполяризація (ІЛЮСТРАТИВНІ рівні,
не калібровка конкретного QPU) + readout-помилка. Показує, як P(допустимі)/P(оптимум) падають із ростом p і шуму.

Режими:
  $env:QAOA_INSTANCE="C"; python qaoa_noslack_noisy.py          # старий режим: p=1,2,3, решітка, 20000 shots
  python qaoa_noslack_noisy.py scaling                           # задача A: C, L18, L21, L28 × p∈{1,2} × обидві топології
  python qaoa_noslack_noisy.py scaling --inst C,L18 --p 1,2 --topology lattice|all|both --shots 4000 [--grid 8x7]

Режим scaling (2026-11-15, «Масштабування з шумом»):
  - сітка обирається автоматично: c = ⌈√n⌉, r = ⌈n/c⌉ (10 → 3×4, 18 → 4×5, 21 → 5×5, 28 → 5×6), або --grid RxC
    (для Emerald — 8x7); --topology all — coupling_map=None (без SWAP-ів, як на іонах), базис той самий
    rz,sx,x,cx — нам потрібна лише кількість 2-кубітних вентилів;
  - рівні шуму e2 = 0 / 0.3% / 0.5% / 1% / 2% (0.3% — орієнтир Forte/Emerald за паспортами 99.5–99.7%);
  - для n ≥ 18 Aer — statevector з траєкторіями (method="statevector" примусово), shots = 4000;
  - кути: точна оптимізація (qaoa_fast, L-BFGS-B, 30 рестартів), кеш у angles_<інстанс>_p<p>.json;
    для L28 (2^28 станів — точна симуляція не влазить у пам'ять) кути ПЕРЕНЕСЕНО з L21 (переносимість кутів —
    теж результат);
  - коефіцієнти Ising рахуються аналітично з QUBO (без 2^n×n² матриці lstsq), P(оптимум) — класифікацією лише
    виміряних станів, тому L28 не потребує векторів довжини 2^28 (крім самої Aer-симуляції);
  - результат — results_noisy_scaling.json і markdown-таблиця; критерій «є сигнал»: P(оптимум) при e2 = 1%
    ≥ 10 × baseline і ≥ 0.5%.
"""
import os, sys, itertools, time, json, math, datetime
import numpy as np
import numba as nb
from scipy.optimize import minimize
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
from qaoa_noslack import build, INSTANCES, LAM, T
from qaoa_exact import qaoa_state

EXACT_MAX_N = 22          # до 22 кубітів будуємо повні вектори 2^n (енергія, маски); більше — лише аналітично


# ---------- QUBO / Ising без векторів 2^n ----------
def qubo(loads, caps, lam=LAM):
    """Ті самі члени, що й qaoa_noslack.build, але як коефіцієнти: E(x) = c0 + Σ h_k x_k + Σ_{k<l} J_kl x_k x_l."""
    N, M = len(loads), len(caps)
    cap = [T * c for c in caps]
    n = N * M + M
    xi = lambda j, i: j * M + i
    yi = lambda i: N * M + i
    for r in range(3, N + 1):              # та сама перевірка, що в build: порушуючі підмножини — пари
        for S in itertools.combinations(range(N), r):
            for i in range(M):
                if sum(loads[j] for j in S) > cap[i]:
                    assert any(loads[a] + loads[b] > cap[i] for a, b in itertools.combinations(S, 2)), \
                        "є мінімальна порушуюча підмножина розміру >=3 — парні штрафи недостатні"
    c0, h, J = 0.0, np.zeros(n), np.zeros((n, n))

    def add(a, b, v):
        J[min(a, b), max(a, b)] += v
    for i in range(M):
        h[yi(i)] += 1                                          # Σ y_i
    for j in range(N):                                         # lam·(Σ_i x_ji − 1)² = lam·(1 − Σx + 2Σ_{i<i'} x x)
        c0 += lam
        for i in range(M):
            h[xi(j, i)] -= lam
        for i, i2 in itertools.combinations(range(M), 2):
            add(xi(j, i), xi(j, i2), 2 * lam)
        for i in range(M):                                     # lam·x_ji(1 − y_i)
            h[xi(j, i)] += lam
            add(xi(j, i), yi(i), -lam)
    for i in range(M):                                         # місткість: пари, що не влазять
        for a, b in itertools.combinations(range(N), 2):
            if loads[a] + loads[b] > cap[i]:
                add(xi(a, i), xi(b, i), lam)
    return n, c0, h, J


def qubo_energy(idx, n, c0, h, J):
    X = ((np.asarray(idx, np.int64)[:, None] >> (n - 1 - np.arange(n))[None, :]) & 1).astype(float)
    return c0 + X @ h + ((X @ J) * X).sum(1)


@nb.njit(cache=True)
def _chunk_range(ch, c0, h, Js, n, low):
    """Мінімум/максимум E по 2^low станах з фіксованими старшими бітами ch (обхід кодом Грея)."""
    hi_bits = n - low
    x = np.zeros(n)
    for k in range(hi_bits):
        x[k] = (ch >> (hi_bits - 1 - k)) & 1
    e = c0
    for k in range(n):
        if x[k] > 0:
            e += h[k]
            for l in range(k + 1, n):
                if x[l] > 0:
                    e += Js[k, l]
    mn = e; mx = e
    for t in range(1, 1 << low):
        b = 0
        while not (t >> b) & 1:
            b += 1
        k = n - 1 - b                     # біт b від молодшого = кубіт n−1−b
        f = h[k]
        for l in range(n):
            f += Js[k, l] * x[l]
        if x[k] > 0:
            e -= f; x[k] = 0.0
        else:
            e += f; x[k] = 1.0
        mn = min(mn, e); mx = max(mx, e)
    return mn, mx


@nb.njit(parallel=True, cache=True)
def _energy_range(c0, h, Js, n, low):
    """Мінімум і максимум E(x) по всіх 2^n станах: паралельно по старших бітах, код Грея по молодших `low`."""
    nch = 1 << (n - low)
    mins = np.empty(nch); maxs = np.empty(nch)
    for ch in nb.prange(nch):
        mins[ch], maxs[ch] = _chunk_range(ch, c0, h, Js, n, low)
    return mins.min(), maxs.max()


def energy_range(n, c0, h, J):
    Js = J + J.T                               # симетрична, діагональ нульова
    return _energy_range(c0, h, Js, n, min(n, 18))


def ising_terms(n, h, J, Emin, Emax):
    """Розклад Ez = (E − Emin)/(Emax − Emin) на Z_k і Z_k Z_l (x = (1 − Z)/2); порядок — як у старому lstsq."""
    R = Emax - Emin
    Js = J + J.T
    terms = []
    for k in range(n):
        c = (-h[k] / 2 - Js[k].sum() / 4) / R
        if abs(c) > 1e-12: terms.append(((k,), c))
    for k, l in itertools.combinations(range(n), 2):
        c = J[k, l] / 4 / R
        if abs(c) > 1e-12: terms.append(((k, l), c))
    return terms


def assignment_stats(loads, caps):
    """Перебір M^N розподілів: оптимум (ФС) і частки допустимих/оптимальних СТАНІВ кубітів (baseline)."""
    N, M = len(loads), len(caps)
    cap = [T * c for c in caps]
    used = []
    for a in itertools.product(range(M), repeat=N):
        ld = [0.0] * M
        for j, i in enumerate(a): ld[i] += loads[j]
        if all(ld[i] <= cap[i] + 1e-9 for i in range(M)):
            used.append(len(set(a)))
    best = min(used)
    n = N * M + M
    n_feas = sum(2 ** (M - k) for k in used)          # y ⊇ використані сервери
    n_opt = sum(1 for k in used if k == best)         # y = використані, |y| = best
    return best, n_feas / 2 ** n, n_opt / 2 ** n


def classify_idx(idx, loads, caps, best):
    """(допустимий, оптимальний) для заданих індексів станів — та сама семантика, що в build()."""
    N, M = len(loads), len(caps)
    n = N * M + M
    idx = np.asarray(idx, np.int64)
    X = (idx[:, None] >> (n - 1 - np.arange(n))[None, :]) & 1
    xs = X[:, :N * M].reshape(-1, N, M)
    onehot = (xs.sum(2) == 1).all(1)
    assign = xs.argmax(2)
    ld = np.stack([((assign == i) * np.array(loads)[None, :]).sum(1) for i in range(M)], 1)
    capok = (ld <= np.array([T * c for c in caps])[None, :] + 1e-9).all(1)
    linked = np.all(np.stack([(xs[:, :, i].sum(1) == 0) | (X[:, N * M + i] == 1) for i in range(M)], 1), 1)
    feasible = onehot & capok & linked
    optimal = feasible & (X[:, N * M:].sum(1) == best)
    return feasible, optimal


def setup(name):
    """Усе для інстансу: n, terms, best, baselines; для n ≤ EXACT_MAX_N — ще E, Ez і маски (як раніше)."""
    loads, caps = INSTANCES[name]
    n, c0, h, J = qubo(loads, caps)
    S = dict(name=name, loads=loads, caps=caps, n=n, c0=c0, h=h, J=J)
    if n <= EXACT_MAX_N:
        n_, E, onehot, feasible, optimal, best = build(loads, caps)
        assert n_ == n
        Emin, Emax = E.min(), E.max()
        S.update(E=E, Ez=(E - Emin) / (Emax - Emin), onehot=onehot, feasible=feasible, optimal=optimal, best=best,
                 base_feas=float(feasible.mean()), base_opt=float(optimal.mean()))
        chk = np.random.default_rng(0).integers(0, 2 ** n, 2000)
        assert np.allclose(qubo_energy(chk, n, c0, h, J), E[chk]), "QUBO-коефіцієнти ≠ build()"
    else:
        Emin, Emax = energy_range(n, c0, h, J)
        best, bf, bo = assignment_stats(loads, caps)
        S.update(E=None, Ez=None, onehot=None, feasible=None, optimal=None, best=best, base_feas=bf, base_opt=bo)
    S.update(Emin=float(Emin), Emax=float(Emax), terms=ising_terms(n, h, J, Emin, Emax))
    return S


# ---------- модульні змінні (сумісність: qaoa_noslack_real_qpu.py, compare_common.py) ----------
NAME = os.environ.get("QAOA_INSTANCE", "B")
_S = setup(NAME)
loads, caps = _S["loads"], _S["caps"]
N, M = len(loads), len(caps)
n, E, onehot, feasible, optimal, best = _S["n"], _S["E"], _S["onehot"], _S["feasible"], _S["optimal"], _S["best"]
Ez, terms = _S["Ez"], _S["terms"]


def circuit(params, terms_=None, n_=None):
    terms_ = terms if terms_ is None else terms_
    n_ = n if n_ is None else n_
    p = len(params) // 2
    betas, gammas = params[:p], params[p:]
    qc = QuantumCircuit(n_)
    qc.h(range(n_))
    for l in range(p):
        for lab, c in terms_:
            if len(lab) == 1: qc.rz(2 * gammas[l] * c, lab[0])
            else:
                a, b = lab; qc.cx(a, b); qc.rz(2 * gammas[l] * c, b); qc.cx(a, b)
        for q in range(n_): qc.rx(2 * betas[l], q)
    qc.measure_all()
    return qc


def grid_shape(n_):
    """Найменша майже квадратна сітка r×c ≥ n: c = ⌈√n⌉, r = ⌈n/c⌉ (8 → 3×3, 10 → 3×4, 18 → 4×5, 21 → 5×5, 28 → 5×6)."""
    c = math.ceil(math.sqrt(n_))
    return math.ceil(n_ / c), c


def grid_map(r=3, c=3):
    edges = []
    for i in range(r):
        for j in range(c):
            v = i * c + j
            if j + 1 < c: edges += [[v, v + 1], [v + 1, v]]
            if i + 1 < r: edges += [[v, v + c], [v + c, v]]
    return edges


def noise_model(e1, e2, ro):
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(e1, 1), ["sx", "x"])
    nm.add_all_qubit_quantum_error(depolarizing_error(e2, 2), ["cx"])
    nm.add_all_qubit_readout_error(ReadoutError([[1 - ro, ro], [ro, 1 - ro]]))
    return nm


def probs_from_counts(counts, shots):
    pr = np.zeros(2 ** n)
    for bs, c in counts.items():
        pr[int(bs[::-1], 2)] += c          # Qiskit: clbit 0 праворуч → перевертаємо, кубіт 0 = старший біт
    return pr / shots


def main(ps=(1, 2, 3), levels=((0, 0, 0), (0.0005, 0.005, 0.01), (0.001, 0.01, 0.02), (0.002, 0.02, 0.03)), shots=20000):
    rng = np.random.default_rng(0)
    print(f"інстанс {NAME}: baseline допустимі {feasible.mean():.3f} оптимальні {optimal.mean():.4f}")
    for p in ps:
        bestr = None
        for _ in range(30):
            x0 = np.concatenate([rng.uniform(0, np.pi / 2, p), rng.uniform(0, 2 * np.pi, p)])
            r = minimize(lambda x: float((np.abs(qaoa_state(Ez, x, n)) ** 2) @ Ez), x0, method="L-BFGS-B")
            if bestr is None or r.fun < bestr.fun: bestr = r
        qc = circuit(bestr.x)
        tqc = transpile(qc, basis_gates=["rz", "sx", "x", "cx"], coupling_map=grid_map(*grid_shape(n)), optimization_level=1, seed_transpiler=1)
        ops = tqc.count_ops()
        print(f"\np={p}: cx={ops.get('cx',0)} глибина={tqc.depth()}")
        # mapping: після transpile вимір на фізичні кубіти → classical bit k = логічний k (measure_all зберігає clbit)
        for e1, e2, ro in levels:
            sim = AerSimulator(noise_model=noise_model(e1, e2, ro) if (e1 or e2 or ro) else None)
            counts = sim.run(tqc, shots=shots).result().get_counts()
            pr = probs_from_counts(counts, shots)
            print(f"  шум e1={e1} e2={e2} ro={ro}: P(допустимі)={pr[feasible].sum():.3f} P(оптимум)={pr[optimal].sum():.3f}")
            sys.stdout.flush()


# ---------- режим scaling (задача A) ----------
SCALING_LEVELS = ((0, 0, 0), (0.0003, 0.003, 0.006), (0.0005, 0.005, 0.01), (0.001, 0.01, 0.02), (0.002, 0.02, 0.03))
ANGLE_SOURCE = {"L28": "L21"}          # інстанси, для яких точна оптимізація неможлива: кути переносимо


def angles_for(S, p, restarts=30, seed=0):
    """Кути з точної оптимізації середньої енергії (кешуються в angles_<інстанс>_p<p>.json)."""
    fn = f"angles_{S['name']}_p{p}.json"
    if os.path.exists(fn):
        return json.load(open(fn, encoding="utf-8"))
    if S["Ez"] is None:
        src = ANGLE_SOURCE[S["name"]]
        a = dict(angles_for(setup(src), p, restarts, seed))
        a.update(instance=S["name"], source=f"перенесено з {src} (точна оптимізація там)", transferred_from=src)
        json.dump(a, open(fn, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        return a
    import qaoa_fast as qf
    Ezc, n_ = np.ascontiguousarray(S["Ez"]), S["n"]
    f = lambda x: float(qf.expect(qf.qaoa_state(Ezc, x, n_), Ezc))
    rng = np.random.default_rng(seed)
    for pp in range(1, p):          # той самий потік випадкових стартів, що в main() (p=1, 2, … підряд) —
        for _ in range(restarts):   # щоб кути збігалися зі старим режимом (C, p=2: README ≈17% при e2=1%)
            rng.uniform(0, np.pi / 2, pp); rng.uniform(0, 2 * np.pi, pp)
    t = time.time(); bestr = None
    for _ in range(restarts):
        x0 = np.concatenate([rng.uniform(0, np.pi / 2, p), rng.uniform(0, 2 * np.pi, p)])
        r = minimize(f, x0, method="L-BFGS-B")
        if bestr is None or r.fun < bestr.fun: bestr = r
    a = dict(instance=S["name"], p=p, angles=bestr.x.tolist(), mean_energy_norm=float(bestr.fun), restarts=restarts,
             seed=seed, source="точна оптимізація <Ez> (qaoa_fast, L-BFGS-B)", runtime_s=round(time.time() - t, 1),
             generated=datetime.date.today().isoformat())
    json.dump(a, open(fn, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    return a


def counts_metrics(counts, S):
    """P(оптимум)/P(допустимі) з Qiskit-counts — класифікуються лише виміряні стани (для будь-якого n)."""
    idx = np.array([int(bs.replace(" ", "")[::-1], 2) for bs in counts])     # Qiskit: перевертаємо рядок
    cnt = np.array(list(counts.values()), float)
    shots = cnt.sum()
    fe, op = classify_idx(idx, S["loads"], S["caps"], S["best"])
    po, pf = cnt[op].sum() / shots, cnt[fe].sum() / shots
    return dict(P_opt=float(po), P_feas=float(pf), se_opt=float(np.sqrt(po * (1 - po) / shots)), shots=int(shots))


def with_saved_probs(tqc, n_):
    """Замість вимірювання — SaveProbabilities на фізичних кубітах логічних 0..n−1 (після маршрутизації).
    Aer: перший кубіт у списку — МОЛОДШИЙ біт, тож передаємо їх у зворотному порядку → кубіт 0 = старший біт."""
    from qiskit_aer.library import SaveProbabilities
    fl = tqc.layout.final_index_layout() if tqc.layout is not None else list(range(n_))
    out = tqc.copy()
    out.append(SaveProbabilities(n_), [fl[k] for k in reversed(range(n_))])
    return out


def apply_readout(pr, n_, ro):
    """Незалежна readout-помилка (біт перевертається з імовірністю ro) — точно, на векторі ймовірностей."""
    if not ro:
        return pr
    v = pr.reshape((2,) * n_)
    for ax in range(n_):
        a0, a1 = np.take(v, 0, axis=ax), np.take(v, 1, axis=ax)
        v = np.stack([(1 - ro) * a0 + ro * a1, ro * a0 + (1 - ro) * a1], axis=ax)
    return v.reshape(-1)


def run_traj(tqc_p, level, traj, S, batches=4, seed=11):
    """Шумна симуляція траєкторіями (Aer statevector): кожна траєкторія дає ПОВНИЙ вектор ймовірностей
    (save_probabilities, усереднення по траєкторіях), тому дисперсія оцінки P(оптимум) значно менша, ніж
    у такої ж кількості shots. Похибка — std між `batches` незалежними партіями / √batches."""
    e1, e2, ro = level
    noisy = bool(e1 or e2)
    nm = noise_model(e1, e2, 0) if noisy else None          # readout додаємо точно, на векторі ймовірностей
    po, pf = [], []
    # > 22 фізичних кубітів (L21 на решітці 5×5 = 25): один вектор стану 512 МБ — траєкторії НЕ паралелимо,
    # інакше Aer тримає кілька станів одночасно і вичерпує пам'ять (прогон 2026-11-15 так і обірвався)
    kw = dict(max_parallel_shots=1) if tqc_p.num_qubits > 22 else {}
    for b in range(batches if noisy else 1):
        sim = AerSimulator(method="statevector", noise_model=nm, seed_simulator=seed + b, **kw)
        pr = np.asarray(sim.run(tqc_p, shots=max(1, traj // batches) if noisy else 1).result().data()["probabilities"])
        pr = apply_readout(pr, S["n"], ro)
        po.append(float(pr[S["optimal"]].sum())); pf.append(float(pr[S["feasible"]].sum()))
        del pr
    se = float(np.std(po, ddof=1) / np.sqrt(len(po))) if len(po) > 1 else 0.0
    return dict(P_opt=float(np.mean(po)), P_feas=float(np.mean(pf)), se_opt=se,
                trajectories=(traj // batches) * batches if noisy else 0)


def circuit_fidelity(ops, n_, level):
    """Імовірність, що в траєкторії не сталося жодної помилки: (1−e2)^CX · (1−e1)^(sx+x) · (1−ro)^n.
    Груба модель: P_шум(опт) ≈ F·P_точно(опт) + (1−F)·baseline."""
    e1, e2, ro = level
    return (1 - e2) ** ops.get("cx", 0) * (1 - e1) ** (ops.get("sx", 0) + ops.get("x", 0)) * (1 - ro) ** n_


def fmt_pct(x):
    return f"{100 * x:.2f}%" if x >= 0.0001 else f"{100 * x:.1e}%"


def markdown(rows):
    head = "| інстанс | n | p | топологія | CX | глибина | точно | " + " | ".join(
        f"e2={100 * l[1]:g}%" for l in SCALING_LEVELS) + " | baseline | сигнал (1%) |"
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for r in rows:
        ex = fmt_pct(r["exact"]["P_opt"]) if r.get("exact") else "—"
        lv = " | ".join(fmt_pct(x["P_opt"]) if x["P_opt"] is not None else f"F={x['fidelity']:.2g}"
                        for x in r["levels"])
        sig = {True: "так", False: "ні", None: "—"}[r["signal_1pct"]]
        lines.append(f"| {r['instance']} | {r['n']} | {r['p']} | {r['topology']} | {r['cx']} | {r['depth']} | {ex} | {lv} | "
                     f"{fmt_pct(r['baseline_opt'])} | {sig} |")
    return "\n".join(lines)


def git_hash():
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return None


def traj_default(n_, topology):
    """Кількість траєкторій: на 21 кубіті з решіткою (25 фізичних кубітів) траєкторія ≈ 1 с — менше."""
    if n_ <= 12:
        return 2000
    return 200 if (n_ > 20 and topology == "lattice") else 800


def scaling(names=("C", "L18", "L21", "L28"), ps=(1, 2), topologies=("lattice", "all"), traj=None, grid=None,
            out="results_noisy_scaling.json"):
    t_all = time.time()
    payload = dict(instance=list(names), generated=datetime.datetime.now().isoformat(timespec="seconds"), git=git_hash(),
                   levels=[dict(e1=a, e2=b, ro=c) for a, b, c in SCALING_LEVELS], signal_criterion=
                   "P(оптимум) при e2=1% ≥ 10×baseline і ≥ 0.5%",
                   method="Aer statevector, траєкторії з save_probabilities (усереднений вектор ймовірностей), "
                          "readout — точно на векторі; похибка — std між 4 партіями", rows=[])
    if os.path.exists(out):                               # дописуємо: уже пораховані рядки не переганяємо
        payload["rows"] = json.load(open(out, encoding="utf-8")).get("rows", [])
    done = {(r["instance"], r["p"], r["topology"]) for r in payload["rows"]}
    for name in names:
        S = setup(name)
        n_ = S["n"]
        print(f"\n=== {name}: n={n_}, оптимум {S['best']} ФС, baseline оптимум {S['base_opt']:.2e}, "
              f"допустимі {S['base_feas']:.2e}, ZZ-членів {sum(len(l) == 2 for l, _ in S['terms'])} ===")
        for p in ps:
            if all((name, p, tp) in done for tp in topologies):
                continue
            a = angles_for(S, p)
            x = np.array(a["angles"])
            exact = None
            if S["Ez"] is not None:
                import qaoa_fast as qf
                pr = np.abs(qf.qaoa_state(np.ascontiguousarray(S["Ez"]), x, n_)) ** 2
                exact = dict(P_opt=float(pr[S["optimal"]].sum()), P_feas=float(pr[S["feasible"]].sum()))
                del pr
            qc = circuit(x, S["terms"], n_)
            qc.remove_final_measurements()
            for tp in topologies:
                if (name, p, tp) in done:
                    continue
                t = time.time()
                rc = tuple(int(v) for v in grid.lower().split("x")) if grid else grid_shape(n_)
                cmap = grid_map(*rc) if tp == "lattice" else None
                tqc = transpile(qc, basis_gates=["rz", "sx", "x", "cx"], coupling_map=cmap, optimization_level=1,
                                seed_transpiler=1)
                ops = dict(tqc.count_ops())
                row = dict(instance=name, n=n_, p=p, topology=tp, grid=f"{rc[0]}x{rc[1]}" if tp == "lattice" else None,
                           cx=int(ops.get("cx", 0)), n1q=int(ops.get("sx", 0) + ops.get("x", 0)), depth=int(tqc.depth()),
                           physical_qubits=tqc.num_qubits, angles=a["angles"], angles_source=a.get("source"),
                           exact=exact, baseline_opt=S["base_opt"], baseline_feas=S["base_feas"], levels=[])
                print(f"p={p} {tp}{' ' + row['grid'] if row['grid'] else ''}: CX={row['cx']} глибина={row['depth']}"
                      + (f"; точно P(опт)={exact['P_opt']:.4f} P(доп)={exact['P_feas']:.4f}" if exact else ""))
                simulate = n_ <= EXACT_MAX_N
                tqc_p = with_saved_probs(tqc, n_) if simulate else None
                tr = traj or traj_default(n_, tp)
                for lev in SCALING_LEVELS:
                    tl = time.time()
                    F = circuit_fidelity(ops, n_, lev)
                    if simulate:
                        m = run_traj(tqc_p, lev, tr, S)
                    else:
                        m = dict(P_opt=None, P_feas=None, se_opt=None, trajectories=0)
                    m.update(e1=lev[0], e2=lev[1], ro=lev[2], fidelity=F, runtime_s=round(time.time() - tl, 1))
                    if exact:
                        m["model_P_opt"] = F * exact["P_opt"] + (1 - F) * S["base_opt"]
                    row["levels"].append(m)
                    print(f"   e2={lev[1]:<6} F={F:.3g}  " + (f"P(опт)={m['P_opt']:.4f} ± {m['se_opt']:.4f}  "
                          f"P(доп)={m['P_feas']:.4f}  [{m['runtime_s']:.0f} с]" if simulate else "симуляція неможлива (2^n)"))
                    sys.stdout.flush()
                l1 = next(x for x in row["levels"] if x["e2"] == 0.01)
                row["signal_1pct"] = (bool(l1["P_opt"] >= 10 * S["base_opt"] and l1["P_opt"] >= 0.005)
                                      if l1["P_opt"] is not None else None)
                if exact:
                    row["noiseless_vs_exact_pp"] = round(100 * (row["levels"][0]["P_opt"] - exact["P_opt"]), 4)
                row["runtime_s"] = round(time.time() - t, 1)
                payload["rows"].append(row)
                payload["generated"] = datetime.datetime.now().isoformat(timespec="seconds")
                json.dump(payload, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    order = {nm: k for k, nm in enumerate(names)}
    payload["rows"].sort(key=lambda r: (order.get(r["instance"], 99), r["p"], r["topology"] != "lattice"))
    json.dump(payload, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("\n" + markdown(payload["rows"]))
    print(f"\nЗбережено {out} [{time.time() - t_all:.0f} с]")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "scaling":
        arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
        topo = arg("--topology", "both")
        scaling(names=tuple(arg("--inst", "C,L18,L21,L28").split(",")),
                ps=tuple(int(v) for v in arg("--p", "1,2").split(",")),
                topologies=("lattice", "all") if topo == "both" else (topo,),
                traj=int(arg("--traj", 0)) or None, grid=arg("--grid", None))
    else:
        t = time.time(); main(); print(f"[{time.time()-t:.0f}s]")
