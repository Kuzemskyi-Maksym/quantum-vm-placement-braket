"""
Запускає всі 5 методів на спільній тестовій конфігурації (config.py) і
виводить результати в узгодженому форматі:
    метод, активні ФС, час, допустимість, Task ID/вартість (для реальних
    хмарних запусків), апроксимаційне співвідношення r (де застосовно).

Примітка щодо QAOA: метод #3 через обмеження кубітів (33 > 20 для IQM
Garnet) запущено на зменшеному під-інстансі (3 ВМ, 2 ФС) — див. qaoa.py.
Інші 4 методи працюють на повному інстансі (6 ВМ, 3 ФС).
"""

import json

from mbfd import run_mbfd
from cp_sat import run_cp_sat
from genetic import run_genetic
from annealing import run_simulated_annealing
from qaoa import run_qaoa


def best_known_optimum():
    cp = run_cp_sat()
    return cp["active_pms"] if cp["status"] == "OPTIMAL" else None


def approx_ratio(active_pms, optimum):
    if active_pms is None or optimum is None or optimum == 0:
        return None
    return round(active_pms / optimum, 3)


def main():
    print("=" * 70)
    print("Порівняння методів розміщення ВМ — спільна тестова конфігурація")
    print("6 ВМ [3,4,2,5,3,4], 3 ФС [12,12,12], T=0.9, λ1=λ2=6.0")
    print("=" * 70)

    cp_result = run_cp_sat()
    optimum = cp_result["active_pms"] if cp_result["status"] == "OPTIMAL" else None

    results = []
    results.append({
        "method": "CP-SAT (точний, повний інстанс)",
        "active_pms": cp_result["active_pms"],
        "feasible": cp_result["feasible"],
        "runtime_s": round(cp_result["runtime_s"], 4),
        "approx_ratio_r": approx_ratio(cp_result["active_pms"], optimum),
        "task_id": None,
        "cost_usd": None,
    })

    mbfd_result = run_mbfd()
    results.append({
        "method": "MBFD (повний інстанс)",
        "active_pms": mbfd_result["active_pms"],
        "feasible": mbfd_result["feasible"],
        "runtime_s": round(mbfd_result["runtime_s"], 6),
        "approx_ratio_r": approx_ratio(mbfd_result["active_pms"], optimum),
        "task_id": None,
        "cost_usd": None,
    })

    ga_result = run_genetic()
    results.append({
        "method": "Генетичний алгоритм (повний інстанс)",
        "active_pms": ga_result["active_pms"],
        "feasible": ga_result["feasible"],
        "runtime_s": round(ga_result["runtime_s"], 4),
        "approx_ratio_r": approx_ratio(ga_result["active_pms"], optimum),
        "task_id": None,
        "cost_usd": None,
    })

    sa_result = run_simulated_annealing()
    results.append({
        "method": "SimulatedAnnealingSampler (повний інстанс, класична емуляція відпалу)",
        "active_pms": sa_result["active_pms"],
        "feasible": sa_result["feasible"],
        "runtime_s": round(sa_result["runtime_s"], 4),
        "approx_ratio_r": approx_ratio(sa_result["active_pms"], optimum),
        "task_id": None,
        "cost_usd": None,
    })

    qaoa_result, _ = run_qaoa()
    results.append({
        "method": "QAOA (p=2, Aer-симулятор, ЗМЕНШЕНИЙ під-інстанс 3ВМ/2ФС через ліміт кубітів)",
        "active_pms": qaoa_result["active_pms"],
        "feasible": qaoa_result["feasible"],
        "runtime_s": round(qaoa_result["runtime_s"], 4),
        "approx_ratio_r": None,  # інший інстанс, не порівнюється напряму з повним
        "task_id": None,
        "cost_usd": None,
    })

    print(f"\nВідомий оптимум (CP-SAT, повний інстанс): {optimum} активних ФС\n")
    for r in results:
        print(f"- {r['method']}")
        print(f"    active_pms={r['active_pms']}  feasible={r['feasible']}  "
              f"runtime_s={r['runtime_s']}  r={r['approx_ratio_r']}")

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nЗбережено у results.json")


if __name__ == "__main__":
    main()
