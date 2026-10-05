"""
Забирає результат уже відправленої QAOA-задачі з реального QPU за Task ID
(НЕ відправляє нову задачу — безкоштовно, нічого не списує з кредитів).

Корисно, коли процес із qaoa_real_qpu.py помер (закрили ноут, обрив зв'язку),
а сама задача живе на AWS незалежно від локального процесу.

Використання:
    python qaoa_fetch_result.py                 # Task ID за замовчуванням (перший прогін)
    python qaoa_fetch_result.py <task-arn>      # інший Task ID
"""

import json
import sys

import boto3
from braket.aws import AwsQuantumTask, AwsSession

from cqm_model import decode_sample_to_assignment
from qaoa import (
    REDUCED_N,
    REDUCED_M,
    reduced_is_feasible,
    reduced_active_pms,
    reduced_pm_loads,
)
from qaoa_real_qpu import build_ising, REGION

DEFAULT_TASK = "arn:aws:braket:eu-north-1:573086082388:quantum-task/0cbf3ec1-71fe-42ba-b0ae-2a8836b42c76"


def main():
    task_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TASK

    session = AwsSession(boto3.Session(region_name=REGION))
    task = AwsQuantumTask(task_id, aws_session=session)

    state = task.state()
    print("Task ID:", task_id)
    print("Статус:", state)

    if state in ("QUEUED", "CREATED"):
        try:
            print("Позиція в черзі:", task.queue_position().queue_position)
        except Exception as exc:  # noqa: BLE001
            print("Позицію в черзі отримати не вдалось:", exc)
        print("Ще не виконано — спробуйте пізніше (QPU працює лише у вікнах доступності).")
        return
    if state != "COMPLETED":
        print("Задача не в стані COMPLETED — результату немає (FAILED/CANCELLED тощо).")
        return

    result = task.result()
    counts = result.measurement_counts
    total_shots = sum(counts.values())

    # Порядок non-slack змінних детермінований (y_*, x_*_*), slack-імена — випадкові UUID,
    # тому декодуємо лише по першим позиціям, які не slack.
    ising, var_order, _ = build_ising()
    n = len(var_order)
    real_idx = [k for k, v in enumerate(var_order) if not v.startswith("slack")]
    try:
        with open("qaoa_ising_params.json") as f:
            saved = json.load(f)
        saved_real = [v for v in saved["var_order"] if not v.startswith("slack")]
        assert [var_order[k] for k in real_idx] == saved_real, "порядок не-slack змінних не збігається"
    except FileNotFoundError:
        pass

    valid = feasible = 0
    active_hist = {}
    best = None
    best_active = None
    for bitstring, count in counts.items():
        sample = {var_order[k]: int(bitstring[k]) for k in real_idx}
        assignment = decode_sample_to_assignment(sample, invert=None, n=REDUCED_N, m=REDUCED_M)
        if None in assignment:
            continue
        valid += count
        if reduced_is_feasible(assignment):
            feasible += count
            a = reduced_active_pms(assignment)
            active_hist[a] = active_hist.get(a, 0) + count
            if best is None or a < best_active or (a == best_active and count > counts.get(best[1], 0)):
                best, best_active = (assignment, bitstring), a

    best_assignment = best[0] if best else None
    output = {
        "method": "QAOA на реальному QPU (IQM Garnet, eu-north-1), p=2, кути з Aer",
        "task_id": task_id,
        "shots": total_shots,
        "unique_bitstrings": len(counts),
        "valid_one_hot_shots": valid,
        "feasible_shots": feasible,
        "valid_fraction": round(valid / total_shots, 4),
        "feasible_fraction": round(feasible / total_shots, 4),
        "feasible_active_pms_histogram": active_hist,
        "best_assignment": best_assignment,
        "best_active_pms": best_active,
        "best_pm_loads": reduced_pm_loads(best_assignment) if best_assignment else None,
        "random_baseline_note": f"для порівняння: при рівномірно випадковому виході {n} кубітів "
                                f"частка допустимих була б дуже малою — див. Aer-результат",
    }
    try:
        meta = result.task_metadata
        output["created_at"] = str(getattr(meta, "createdAt", None))
        output["ended_at"] = str(getattr(meta, "endedAt", None))
        output["device_id"] = str(getattr(meta, "deviceId", None))
    except Exception:  # noqa: BLE001
        pass

    print(json.dumps(output, indent=2, ensure_ascii=False, default=str))
    with open("qaoa_real_qpu_result.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print("\nЗбережено у qaoa_real_qpu_result.json")


if __name__ == "__main__":
    main()
