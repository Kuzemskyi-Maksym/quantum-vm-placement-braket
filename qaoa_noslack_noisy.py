"""
Slack-free QAOA (інстанс B, 8 кубітів) на Aer З ШУМОМ + реальна топологія (сітка 3x3 як у IQM Garnet).
Кути беруться з точної noiseless-оптимізації. Шум — деполяризація (ІЛЮСТРАТИВНІ рівні, не калібровка Garnet)
+ readout-помилка. Показує, як P(допустимі)/P(оптимум) падають із ростом p і шуму.
"""
import sys, itertools, time
import numpy as np
import dimod
from scipy.optimize import minimize
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
from qaoa_noslack import build, INSTANCES, LAM, T
from qaoa_exact import qaoa_state

NAME = "B"
loads, caps = INSTANCES[NAME]
N, M = len(loads), len(caps)
n, E, onehot, feasible, optimal, best = build(loads, caps)
Ez = (E - E.min()) / (E.max() - E.min())

# Ising у Z-базисі напряму з діагоналі: x_k=(1-Z_k)/2; розклад Ez на {1, Z_k, Z_kZ_l} (точний, бо H квадратичний)
S = np.arange(2 ** n)
X = ((S[:, None] >> (n - 1 - np.arange(n))[None, :]) & 1)
Zs = 1 - 2 * X                       # eigenvalue Z
cols, labels = [np.ones(2 ** n)], [()]
for k in range(n): cols.append(Zs[:, k]); labels.append((k,))
for k, l in itertools.combinations(range(n), 2): cols.append(Zs[:, k] * Zs[:, l]); labels.append((k, l))
coef = np.linalg.lstsq(np.stack(cols, 1), Ez, rcond=None)[0]
assert np.allclose(np.stack(cols, 1) @ coef, Ez, atol=1e-9), "H не квадратичний?"
terms = [(lab, c) for lab, c in zip(labels, coef) if lab and abs(c) > 1e-12]


def circuit(params):
    p = len(params) // 2
    betas, gammas = params[:p], params[p:]
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for l in range(p):
        for lab, c in terms:
            if len(lab) == 1: qc.rz(2 * gammas[l] * c, lab[0])
            else:
                a, b = lab; qc.cx(a, b); qc.rz(2 * gammas[l] * c, b); qc.cx(a, b)
        for q in range(n): qc.rx(2 * betas[l], q)
    qc.measure_all()
    return qc


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
        pr[int(bs.replace(" ", ""), 2) if False else int(bs[::-1], 2) ] += c   # qubit k = перший біт у нашій індексації
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
        tqc = transpile(qc, basis_gates=["rz", "sx", "x", "cx"], coupling_map=grid_map(), optimization_level=1, seed_transpiler=1)
        ops = tqc.count_ops()
        print(f"\np={p}: cx={ops.get('cx',0)} глибина={tqc.depth()}")
        # mapping: після transpile вимір на фізичні кубіти → classical bit k = логічний k (measure_all зберігає clbit)
        for e1, e2, ro in levels:
            sim = AerSimulator(noise_model=noise_model(e1, e2, ro) if (e1 or e2 or ro) else None)
            counts = sim.run(tqc, shots=shots).result().get_counts()
            pr = probs_from_counts(counts, shots)
            print(f"  шум e1={e1} e2={e2} ro={ro}: P(допустимі)={pr[feasible].sum():.3f} P(оптимум)={pr[optimal].sum():.3f}")
            sys.stdout.flush()

if __name__ == "__main__":
    t = time.time(); main(); print(f"[{time.time()-t:.0f}s]")
