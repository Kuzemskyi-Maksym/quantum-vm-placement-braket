"""
Warm-start QAOA (інстанс W, 15 кубітів) на Aer З ШУМОМ + топологія-сітка (як qaoa_noslack_noisy.py для C).
Відповідає на питання «чи є сенс нести теплий старт на Garnet»: якщо при помилці CX ≈ 1–2% P(кращий за x0)
падає до рівня випадкового — на залізо не йдемо (і це теж результат).

Схема: X на кубітах x0 → RX(2β0·w_k) → [фаза exp(−iγ_l·Ez) (з членом міграцій MU) → RX(2β_l·w_k)] × p.
Кути — з точної оптимізації (warm_start_qaoa.run_warm, ціль «максимум P(кращий за x0)»).
w_k — ваги змішувача: 1 (звичайний RX) або за розміром ВМ (--mixer size --alpha 2).
Шум — деполяризація (ІЛЮСТРАТИВНІ рівні, не калібровка Garnet) + readout; сітка 4×4 ≥ 15 кубітів.

Запуск (локально, потрібні qiskit, qiskit-aer):
  $env:VMP_INSTANCE="W"; python warm_start_noisy.py --mu 1 --mixer size --alpha 2
  python warm_start_noisy.py --mu 1                       # звичайний RX
Порівняти з класичним контролем: випадковий допустимий сусід d=1/2 дає 25% / 40% кращих за x0 (без шуму).
"""
import os, sys, itertools, time
os.environ.setdefault("VMP_INSTANCE", "W")
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
import warm_start_qaoa as ws

arg = lambda k, d: type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
MU, MIXER, ALPHA = arg("--mu", ws.MU), arg("--mixer", "rx"), arg("--alpha", 1.0)
SHOTS = arg("--shots", 20000)
LEVELS = ((0, 0, 0), (0.0005, 0.005, 0.01), (0.001, 0.01, 0.02), (0.002, 0.02, 0.03))

P = ws.problem(MU)
n, Ez, idx0 = P["n"], P["Ez"], P["idx0"]
w = ws.size_weights(ALPHA) if MIXER == "size" else np.ones(n)

# Розклад Ez на {1, Z_k, Z_k Z_l} (точний: H квадратичний; член міграцій — лінійний)
S = np.arange(2 ** n)
X = ((S[:, None] >> (n - 1 - np.arange(n))[None, :]) & 1)
Zs = 1 - 2 * X
cols, labels = [np.ones(2 ** n)], [()]
for k in range(n): cols.append(Zs[:, k]); labels.append((k,))
for k, l in itertools.combinations(range(n), 2): cols.append(Zs[:, k] * Zs[:, l]); labels.append((k, l))
coef = np.linalg.lstsq(np.stack(cols, 1), Ez, rcond=None)[0]
assert np.allclose(np.stack(cols, 1) @ coef, Ez, atol=1e-9), "H не квадратичний?"
terms = [(lab, c) for lab, c in zip(labels, coef) if lab and abs(c) > 1e-12]
x0_bits = [(idx0 >> (n - 1 - k)) & 1 for k in range(n)]


def circuit(params):
    p = (len(params) - 1) // 2
    b0, gs, bs = params[0], params[1:1 + p], params[1 + p:]
    qc = QuantumCircuit(n)
    for k in range(n):
        if x0_bits[k]: qc.x(k)
    for k in range(n): qc.rx(2 * b0 * w[k], k)
    for l in range(p):
        for lab, c in terms:
            if len(lab) == 1: qc.rz(2 * gs[l] * c, lab[0])
            else:
                a, b = lab; qc.cx(a, b); qc.rz(2 * gs[l] * c, b); qc.cx(a, b)
        for k in range(n): qc.rx(2 * bs[l] * w[k], k)
    qc.measure_all()
    return qc


def grid_map(r, c):
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
        pr[int(bs[::-1], 2)] += c          # кубіт 0 — найстарший біт у нашій індексації
    return pr / shots


def main(ps=(1, 2)):
    bt, fe = P["better"], P["feasible"]
    print(f"W: MU={MU}, F(x0)={P['F0']:.3f}, змішувач={MIXER}" + (f" alpha={ALPHA}" if MIXER == "size" else "")
          + f"; baseline (рівномірно): кращі за x0 {bt.mean():.4f}, допустимі {fe.mean():.4f}")
    for p in ps:
        r = ws.run_warm(mu=MU, p=p, mixer=MIXER, alpha=ALPHA)
        pr_exact = np.abs(ws.warm_state(Ez, np.array(r["angles"]), n, idx0, None if MIXER == "rx" else w)) ** 2
        qc = circuit(np.array(r["angles"]))
        tqc = transpile(qc, basis_gates=["rz", "sx", "x", "cx"], coupling_map=grid_map(4, 4),
                        optimization_level=1, seed_transpiler=1)
        ops = tqc.count_ops()
        print(f"\np={p}: cx={ops.get('cx', 0)} глибина={tqc.depth()}; точно (numpy): кращі {pr_exact[bt].sum():.3f}, допустимі {pr_exact[fe].sum():.3f}")
        for e1, e2, ro in LEVELS:
            sim = AerSimulator(noise_model=noise_model(e1, e2, ro) if (e1 or e2 or ro) else None)
            counts = sim.run(tqc, shots=SHOTS).result().get_counts()
            pr = probs_from_counts(counts, SHOTS)
            em = (pr[bt] * P["migc"][bt]).sum() / max(pr[bt].sum(), 1e-12)
            print(f"  шум e1={e1} e2={e2} ro={ro}: P(кращі за x0)={pr[bt].sum():.3f}  P(допустимі)={pr[fe].sum():.3f}  "
                  f"сер. варт. мігр. (кращі)={em:.2f}")
            sys.stdout.flush()


if __name__ == "__main__":
    t = time.time(); main(); print(f"[{time.time() - t:.0f}s]")
