"""
Slack-free QAOA (інстанс B, 8 кубітів) на реальному QPU IQM Garnet (eu-north-1).

Режими:
  python qaoa_noslack_real_qpu.py local            # безкоштовно: локальний симулятор Braket, перевірка схеми
  python qaoa_noslack_real_qpu.py submit [P] [SHOTS]  # ПЛАТНО: P=2, SHOTS=1000 за замовчуванням (~$1.75)
  python qaoa_noslack_real_qpu.py fetch <task-arn> # безкоштовно: забрати результат за Task ID

submit НЕ чекає на результат: одразу друкує Task ID і зберігає у noslack_task.txt
(задача живе на AWS, QPU працює лише у вікнах доступності — будні дні).
"""
import json, sys
import numpy as np
from scipy.optimize import minimize
import boto3
from braket.aws import AwsDevice, AwsQuantumTask, AwsSession
from braket.circuits import Circuit
from braket.devices import LocalSimulator

from qaoa_exact import qaoa_state
import qaoa_noslack_noisy as nz   # будує E, Ez, terms, feasible, optimal для інстансу B

DEVICE_ARN = "arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet"
REGION = "eu-north-1"


def best_angles(p, restarts=30, seed=0):
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(restarts):
        x0 = np.concatenate([rng.uniform(0, np.pi / 2, p), rng.uniform(0, 2 * np.pi, p)])
        r = minimize(lambda x: float((np.abs(qaoa_state(nz.Ez, x, nz.n)) ** 2) @ nz.Ez), x0, method="L-BFGS-B")
        if best is None or r.fun < best.fun:
            best = r
    return best.x


def braket_circuit(params):
    p = len(params) // 2
    betas, gammas = params[:p], params[p:]
    c = Circuit()
    for q in range(nz.n):
        c.h(q)
    for l in range(p):
        for lab, coef in nz.terms:
            if len(lab) == 1:
                c.rz(lab[0], 2 * gammas[l] * coef)
            else:
                a, b = lab
                c.cnot(a, b); c.rz(b, 2 * gammas[l] * coef); c.cnot(a, b)
        for q in range(nz.n):
            c.rx(q, 2 * betas[l])
    return c


def analyze(counts, label):
    shots = sum(counts.values())
    pr = np.zeros(2 ** nz.n)
    for bs, c in counts.items():
        pr[int(bs, 2)] += c          # qubit 0 = старший біт, x_k = виміряний біт
    pr = pr / shots
    out = {
        "label": label, "shots": shots,
        "P_feasible": float(pr[nz.feasible].sum()), "P_optimal": float(pr[nz.optimal].sum()),
        "P_onehot": float(pr[nz.onehot].sum()),
        "baseline_feasible": float(nz.feasible.mean()), "baseline_optimal": float(nz.optimal.mean()),
        "std_err_optimal": float(np.sqrt(pr[nz.optimal].sum() * (1 - pr[nz.optimal].sum()) / shots)),
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return out


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "local"
    if mode == "fetch":
        sess = AwsSession(boto3.Session(region_name=REGION))
        task = AwsQuantumTask(sys.argv[2], aws_session=sess)
        st = task.state(); print("Статус:", st)
        if st in ("QUEUED", "CREATED"):
            try: print("Позиція в черзі:", task.queue_position().queue_position)
            except Exception as e: print("позиція недоступна:", e)
            return
        if st != "COMPLETED":
            print("Результату немає"); return
        out = analyze(task.result().measurement_counts, "IQM Garnet")
        json.dump(out, open("noslack_qpu_result.json", "w"), indent=2, ensure_ascii=False)
        return

    p = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    shots = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    params = best_angles(p)
    circ = braket_circuit(params)
    ideal = np.abs(qaoa_state(nz.Ez, params, nz.n)) ** 2
    print(f"p={p}, ідеальні P(допустимі)={ideal[nz.feasible].sum():.3f}, baseline {nz.feasible.mean():.3f}")

    if mode == "local":
        res = LocalSimulator().run(circ, shots=20000).result()
        analyze(res.measurement_counts, "Braket local simulator (noiseless)")
    elif mode == "submit":
        cost = 0.30 + shots * 0.00145
        print(f"Відправка на Garnet: {shots} shots, орієнтовна вартість ≈ ${cost:.2f}")
        sess = AwsSession(boto3.Session(region_name=REGION))
        dev = AwsDevice(DEVICE_ARN, aws_session=sess)
        task = dev.run(circ, shots=shots)
        print("Task ID:", task.id)
        open("noslack_task.txt", "w").write(task.id)
        print("Збережено у noslack_task.txt. Результат: python qaoa_noslack_real_qpu.py fetch <Task ID>")


if __name__ == "__main__":
    main()
