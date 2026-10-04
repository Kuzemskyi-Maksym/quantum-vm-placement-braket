"""
Запуск QAOA на реальному QPU — IQM Garnet (eu-north-1), через Amazon Braket.

Використовує ті самі ОПТИМАЛЬНІ кути (betas, gammas), що вже знайдені локально
на Aer-симуляторі (збережені в qaoa_ising_params.json) — кути тут НЕ
оптимізуються заново (це окремий, дорожчий сценарій "варіаційний цикл на
реальному залізі"; тут перевіряється лише, чи тримається якість рішення на
шумному залізі з уже готовими кутами).

Працює на ЗМЕНШЕНОМУ під-інстансі (3 ВМ, 2 ФС, 16 кубітів) — той самий, що й
у qaoa.py — повний інстанс (6 ВМ, 3 ФС) не влазить у QPU (~33 кубіти > 20
в IQM Garnet).

ВАЖЛИВО: це ПЛАТНИЙ запуск на реальному залізі (списує кошти з AWS-кредитів
дослідницького акаунту corezoid-kpi-braket). SHOTS=100 навмисно невеликий
для першого тестового прогону.
"""

import json
import time

import boto3
from braket.aws import AwsDevice, AwsSession
from braket.circuits import Circuit

from cqm_model import build_bqm, decode_sample_to_assignment
from config import T, LAMBDA1
from qaoa import (
    REDUCED_VM_LOADS,
    REDUCED_PM_CAPACITY,
    REDUCED_N,
    REDUCED_M,
    reduced_is_feasible,
    reduced_active_pms,
    reduced_pm_loads,
)

DEVICE_ARN = "arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet"
REGION = "eu-north-1"
SHOTS = 100  # невелика кількість shots для першого тестового (платного) прогону


def build_ising():
    bqm, invert, cqm = build_bqm(
        lagrange_multiplier=LAMBDA1,
        vm_loads=REDUCED_VM_LOADS,
        pm_capacity=REDUCED_PM_CAPACITY,
        t=T,
    )
    ising = bqm.spin
    var_order = list(ising.variables)
    return ising, var_order, invert


def build_circuit(ising, var_order, betas, gammas):
    n = len(var_order)
    idx = {v: k for k, v in enumerate(var_order)}
    circ = Circuit()
    for q in range(n):
        circ.h(q)

    p = len(betas)
    for layer in range(p):
        gamma = gammas[layer]
        for v, coeff in ising.linear.items():
            if coeff == 0:
                continue
            circ.rz(idx[v], 2 * gamma * coeff)
        for (u, v), coeff in ising.quadratic.items():
            if coeff == 0:
                continue
            a, b = idx[u], idx[v]
            circ.cnot(a, b)
            circ.rz(b, 2 * gamma * coeff)
            circ.cnot(a, b)
        beta = betas[layer]
        for q in range(n):
            circ.rx(q, 2 * beta)
    return circ


def main():
    with open("qaoa_ising_params.json") as f:
        saved = json.load(f)
    opt_params = saved["optimal_params"]
    p = len(opt_params) // 2
    betas, gammas = opt_params[:p], opt_params[p:]

    # ПРИМІТКА: slack-змінні в dimod отримують випадкове UUID-ім'я при кожній
    # генерації BQM (cqm_to_bqm), тому var_order тут НЕ буде збігатись з тим,
    # що збережено в qaoa_ising_params.json — це нормально й не проблема:
    # оптимальні кути (betas, gammas) — це кути QAOA-шарів, не прив'язані до
    # конкретних імен змінних, тому застосовуються напряму до щойно
    # побудованого Ising-гамільтоніана (та сама структура задачі/та сама
    # кількість кубітів, лише інші внутрішні імена slack-змінних).
    ising, var_order, invert = build_ising()
    if len(var_order) != len(saved["var_order"]):
        raise RuntimeError(
            f"Кількість кубітів не збігається: щойно побудовано {len(var_order)}, "
            f"збережено {len(saved['var_order'])} — структура задачі змінилась, "
            f"перезапусти qaoa.py"
        )

    circ = build_circuit(ising, var_order, betas, gammas)

    session = AwsSession(boto3.Session(region_name=REGION))
    device = AwsDevice(DEVICE_ARN, aws_session=session)
    print(f"Пристрій: {device.name}, статус: {device.status}")

    start = time.perf_counter()
    task = device.run(circ, shots=SHOTS)
    print(f"Task ID: {task.id}")
    print("Очікування результату (реальний QPU — може зайняти від секунд до хвилин, залежно від черги)...")
    result = task.result()
    elapsed = time.perf_counter() - start

    counts = result.measurement_counts
    n = len(var_order)

    best_assignment, best_active, best_feasible_found = None, None, False
    for bitstring, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        sample = {var_order[k]: int(bitstring[k]) for k in range(n)}
        assignment = decode_sample_to_assignment(sample, invert=invert, n=REDUCED_N, m=REDUCED_M)
        if None not in assignment:
            feasible = reduced_is_feasible(assignment)
            if best_assignment is None or (feasible and not best_feasible_found) or (
                feasible and best_feasible_found and reduced_active_pms(assignment) < best_active
            ):
                best_assignment, best_active, best_feasible_found = assignment, reduced_active_pms(assignment), feasible
            if feasible:
                break

    output = {
        "method": "QAOA на реальному QPU (IQM Garnet, eu-north-1)",
        "task_id": str(task.id),
        "device": device.name,
        "shots": SHOTS,
        "assignment": best_assignment,
        "active_pms": reduced_active_pms(best_assignment) if best_assignment else None,
        "feasible": reduced_is_feasible(best_assignment) if best_assignment else False,
        "pm_loads": reduced_pm_loads(best_assignment) if best_assignment else None,
        "runtime_s": elapsed,
        "raw_counts_top5": sorted(counts.items(), key=lambda kv: -kv[1])[:5],
    }
    print(json.dumps(output, indent=2, ensure_ascii=False, default=str))

    with open("qaoa_real_qpu_result.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print("\nЗбережено у qaoa_real_qpu_result.json")


if __name__ == "__main__":
    main()
