"""
Warm-start QAOA (інстанс W, 15 кубітів, критерій з міграціями MU) на реальному QPU IQM Garnet (eu-north-1).
Схема та сама, що в warm_start_noisy.py: X на кубітах x0 → RX(2β0·w) → [фаза → RX(2β·w)] × p.
За шумною симуляцією (2026-11-02) при e2≈2% p=1 (91 CX) дає ≈1.2% кращих за x0 проти 0.8% для p=2 (216 CX),
тому для заліза рекомендовано p=1 зі змішувачем за розміром ВМ.

Режими (параметри --mu/--mixer/--alpha як у warm_start_noisy.py; за замовч. MU з config, mixer=rx):
  python warm_start_real_qpu.py local [P]                 # безкоштовно: локальний симулятор Braket
  python warm_start_real_qpu.py submit [P] [SHOTS]        # ПЛАТНО: P=1, SHOTS=1000 (≈$1.75)
  python warm_start_real_qpu.py control [P] [SHOTS]       # ПЛАТНО: та сама схема, ВИПАДКОВІ кути (700 ≈ $1.32)
  python warm_start_real_qpu.py control_local [P]
  python warm_start_real_qpu.py fetch <task-arn>          # безкоштовно: забрати результат (повний ARN!)
Приклад: $env:VMP_INSTANCE="W"; python warm_start_real_qpu.py submit 1 1000 --mu 1 --mixer size --alpha 2
"""
import json, sys
import numpy as np
import boto3
from braket.aws import AwsDevice, AwsQuantumTask, AwsSession
from braket.circuits import Circuit
from braket.devices import LocalSimulator

import warm_start_noisy as wn      # P, terms, x0_bits, w, MU, MIXER, ALPHA (читає --mu/--mixer/--alpha)
import warm_start_qaoa as ws

DEVICE_ARN = "arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet"
REGION = "eu-north-1"
TAG = f"mu{wn.MU:g}_{wn.MIXER}" + (f"{wn.ALPHA:g}" if wn.MIXER == "size" else "")
n, P, w = wn.n, wn.P, wn.w


def braket_circuit(params):
    p = (len(params) - 1) // 2
    b0, gs, bs = params[0], params[1:1 + p], params[1 + p:]
    c = Circuit()
    for k in range(n):
        if wn.x0_bits[k]:
            c.x(k)
    for k in range(n):
        c.rx(k, 2 * b0 * w[k])
    for l in range(p):
        for lab, coef in wn.terms:
            if len(lab) == 1:
                c.rz(lab[0], 2 * gs[l] * coef)
            else:
                a, b = lab
                c.cnot(a, b); c.rz(b, 2 * gs[l] * coef); c.cnot(a, b)
        for k in range(n):
            c.rx(k, 2 * bs[l] * w[k])
    return c


def analyze(counts, label):
    shots = sum(counts.values())
    pr = np.zeros(2 ** n)
    for bs, c in counts.items():
        pr[int(bs, 2)] += c          # кубіт 0 = старший біт, x_k = виміряний біт
    pr = pr / shots
    bt, fe = P["better"], P["feasible"]
    pb = float(pr[bt].sum())
    seen = fe & (pr > 0)
    best = None
    if seen.any():
        i = np.flatnonzero(seen)[np.argmin(P["F"][seen])]
        best = dict(assignment=P["assign"][i].tolist(), active_pms=int(P["active"][i]), migrations=int(P["mig"][i]),
                    migration_cost=float(P["migc"][i]), objective=float(P["F"][i]), count=int(round(pr[i] * shots)))
    out = {
        "label": label, "shots": shots, "mu": wn.MU, "mixer": wn.MIXER, "alpha": wn.ALPHA, "F_x0": float(P["F0"]),
        "P_better": pb, "std_err_better": float(np.sqrt(pb * (1 - pb) / shots)),
        "P_feasible": float(pr[fe].sum()), "P_onehot": float(pr[P["onehot"]].sum()),
        "mean_migc_better": float((pr[bt] * P["migc"][bt]).sum() / max(pb, 1e-12)),
        "baseline_better": float(bt.mean()), "baseline_feasible": float(fe.mean()),
        "best_seen": best,
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return out


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "local"
    pos, skip = [], False                      # позиційні аргументи без значень прапорців --key value
    for a in sys.argv[2:]:
        if skip: skip = False; continue
        if a.startswith("--"): skip = True; continue
        pos.append(a)
    if mode == "fetch":
        sess = AwsSession(boto3.Session(region_name=REGION))
        task = AwsQuantumTask(pos[0], aws_session=sess)
        st = task.state(); print("Статус:", st)
        if st in ("QUEUED", "CREATED"):
            try: print("Позиція в черзі:", task.queue_position().queue_position)
            except Exception as e: print("позиція недоступна:", e)
            return
        if st != "COMPLETED":
            print("Результату немає"); return
        out = analyze(task.result().measurement_counts, "IQM Garnet")
        json.dump(out, open(f"warm_qpu_result_{TAG}.json", "w"), indent=2, ensure_ascii=False)
        return

    p = int(pos[0]) if len(pos) > 0 else 1
    is_control = mode in ("control", "control_local")
    shots = int(pos[1]) if len(pos) > 1 else (700 if is_control else 1000)
    if is_control:
        rng = np.random.default_rng(123)
        params = np.concatenate([[rng.uniform(0, np.pi)], rng.uniform(0, ws.GMAX, p), rng.uniform(0, np.pi, p)])
    else:
        params = np.array(ws.run_warm(mu=wn.MU, p=p, mixer=wn.MIXER, alpha=wn.ALPHA)["angles"])
    circ = braket_circuit(params)
    ideal = np.abs(ws.warm_state(P["Ez"], params, n, P["idx0"], None if wn.MIXER == "rx" else w)) ** 2
    print(f"p={p}, {TAG}: ідеальні P(кращі за x0)={ideal[P['better']].sum():.4f}, P(допустимі)={ideal[P['feasible']].sum():.3f}, "
          f"baseline кращих {P['better'].mean():.4f}; кути {np.round(params, 3).tolist()}")

    if mode in ("local", "control_local"):
        res = LocalSimulator().run(circ, shots=20000).result()
        analyze(res.measurement_counts, "Braket local simulator (noiseless)")
    elif mode in ("submit", "control"):
        cost = 0.30 + shots * 0.00145
        print(f"Відправка на Garnet: {shots} shots, орієнтовна вартість ≈ ${cost:.2f}")
        sess = AwsSession(boto3.Session(region_name=REGION))
        dev = AwsDevice(DEVICE_ARN, aws_session=sess)
        task = dev.run(circ, shots=shots)
        print("Task ID:", task.id)
        open(("warm_control_task_" if is_control else "warm_task_") + TAG + ".txt", "w").write(task.id)
        print("Task ID збережено. Результат: python warm_start_real_qpu.py fetch <повний ARN> (з тими самими --mu/--mixer/--alpha)")


if __name__ == "__main__":
    main()
