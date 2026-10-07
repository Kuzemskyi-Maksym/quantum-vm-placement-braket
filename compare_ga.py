"""
Порівняння базового ГА (genetic.py) і BRKGA (brkga.py) із MBFD і CP-SAT.
Запуск:  $env:VMP_INSTANCE="large"; python compare_ga.py      (або common / full)
Для кожного стохастичного методу — 20 запусків із різними seed.
"""
import statistics
from config import VM_LOADS, PM_CAPACITY, T
from mbfd import run_mbfd
from cp_sat import run_cp_sat
from genetic import run_genetic
from brkga import run_brkga

SEEDS = range(20)


def summarize(name, runs, opt):
    act = [r["active_pms"] if r["feasible"] else 99 for r in runs]
    print(f"{name:<22} найкраще {min(act)}  середнє {statistics.mean(act):.2f}  "
          f"оптимум у {sum(a == opt for a in act)}/{len(act)}  час {statistics.median(r['runtime_s'] for r in runs):.2f} с")


def main():
    print(f"ВМ={len(VM_LOADS)}, ФС={len(PM_CAPACITY)}, сума навантажень={sum(VM_LOADS)}, "
          f"мін. серверів за обсягом ≥ {-(-sum(VM_LOADS) // (T * PM_CAPACITY[0])):.0f}")
    cp = run_cp_sat(time_limit_s=60)
    opt = cp["active_pms"]
    print(f"CP-SAT: {opt} серверів ({cp['status']}), {cp['runtime_s']:.1f} с")
    mb = run_mbfd()
    print(f"MBFD:   {mb['active_pms']} серверів")
    summarize("Базовий ГА", [run_genetic(seed=s) for s in SEEDS], opt)
    summarize("BRKGA", [run_brkga(seed=s) for s in SEEDS], opt)


if __name__ == "__main__":
    main()
