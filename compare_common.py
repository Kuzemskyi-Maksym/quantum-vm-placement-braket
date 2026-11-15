"""
ПОРІВНЯННЯ 5 методів на ЄДИНОМУ інстансі (VMP_INSTANCE=common: 4 ВМ [6,5,2,1], 2 ФС [12,12], T=0.9).
Запуск:  set VMP_INSTANCE=common  (PowerShell: $env:VMP_INSTANCE="common") ; python compare_common.py
Стохастичні методи (BRKGA, відпал) — 20 запусків з різними seed: частка запусків, що знайшли оптимум.
ГА в основній таблиці — BRKGA (рішення Жарікова 2026-11-01: лише BRKGA, без базового ГА / COFFGA); базовий ГА
лишається в genetic.py для compare_ga.py.
QAOA (slack-free, 10 кубітів) — окремо: $env:QAOA_INSTANCE="C"; python qaoa_noslack_noisy.py (ідеал і шумна симуляція).
"""
import os, json, time, statistics
os.environ.setdefault("VMP_INSTANCE", "common")
import numpy as np
from config import VM_LOADS, PM_CAPACITY, T
from mbfd import run_mbfd
from cp_sat import run_cp_sat
from brkga import run_brkga
from annealing import run_simulated_annealing

SEEDS = range(20)


def summarize(runs, optimum):
    ok = [r["feasible"] and r["active_pms"] == optimum for r in runs]
    return {"optimum_rate": sum(ok) / len(ok), "median_runtime_s": statistics.median(r["runtime_s"] for r in runs),
            "best_active_pms": min(r["active_pms"] for r in runs if r["feasible"])}


def main():
    out = {"instance": {"vm_loads": VM_LOADS, "pm_capacity": PM_CAPACITY, "T": T}}
    cp = run_cp_sat(); opt = cp["active_pms"]
    out["CP-SAT"] = {"active_pms": opt, "optimal_proved": cp.get("status") , "runtime_s": cp["runtime_s"]}
    mb = run_mbfd(); out["MBFD"] = {"active_pms": mb["active_pms"], "feasible": mb["feasible"], "runtime_s": mb["runtime_s"]}
    out["BRKGA"] = summarize([run_brkga(seed=s, x0=None) for s in SEEDS], opt)   # без міграцій: ціль — лише ФС
    out["Annealing"] = summarize([run_simulated_annealing(seed=s) for s in SEEDS], opt)
    # QAOA slack-free
    import qaoa_noslack as ns, qaoa_noslack_noisy as nz
    n, E, onehot, feas, optm, best = ns.build(VM_LOADS, PM_CAPACITY)
    assert best == opt, f"QAOA-інстанс дає оптимум {best}, CP-SAT {opt}"
    out["QAOA"] = {"n_qubits": n, "baseline_optimal": float(optm.mean())}
    print(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    from config import save_results
    save_results("common", out)

if __name__ == "__main__":
    main()
