"""
ПОРІВНЯННЯ методів за ціллю з міграціями F = активні ФС + MU · зважені міграції (рішення Жарікова 2026-11-01).
Інстанс W (VMP_INSTANCE=W): ВМ [6,5,2,1], ФС [12,12,12], поточний розподіл x0 = [0,1,2,2] (3 ФС).
Для кожного MU ∈ {0, 0.5, 1, 2, 5} (при MU=5 оптимум — нічого не рухати):
  - точний оптимум F (повний перебір усіх M^N розподілів; для малих інстансів) і CP-SAT, якщо є OR-Tools;
  - «нічого не рухати» (x0) і MBFD як «повна переукладка» (міграцій не бачить);
  - BRKGA з ключами «залишитись», 20 запусків: частка запусків, що дійшли до оптимуму F;
  - класичний контроль: випадковий допустимий сусід на 1 і 2 міграції — P(кращий за x0);
  - warm-start QAOA p=1, 2 (точна симуляція, ціль «максимум P(кращий за x0)»): P(кращий), середня вартість
    міграцій серед кращих, найкращий стан, що з'явився б за 1000 shots.
Запуск:  $env:VMP_INSTANCE="W"; python compare_migration.py            (результат — results_migration.json)
         python compare_migration.py --no-qaoa                          (лише класичні методи, швидко)
"""
import os, sys, json, itertools, statistics, time
os.environ.setdefault("VMP_INSTANCE", "W")
from config import VM_LOADS, PM_CAPACITY, T, N, M, X0, is_feasible, active_pms, objective, n_migrations, migration_cost
from mbfd import run_mbfd
from brkga import run_brkga

MUS = [0.0, 0.5, 1.0, 2.0, 5.0]
SEEDS = range(20)


def brute_force(mu):
    """Точний оптимум F повним перебором (M^N розподілів)."""
    best, arg = None, None
    for a in itertools.product(range(M), repeat=N):
        if not is_feasible(a):
            continue
        f = objective(a, X0, mu)
        if best is None or f < best - 1e-12:
            best, arg = f, list(a)
    return best, arg


def describe(a, mu):
    return dict(assignment=list(a), active_pms=active_pms(a), migrations=n_migrations(a, X0),
                migration_cost=round(migration_cost(a, X0), 4), objective=round(objective(a, X0, mu), 4))


def main(with_qaoa=True):
    assert M ** N <= 2_000_000, "повний перебір лише для малих інстансів"
    out = {"instance": {"vm_loads": VM_LOADS, "pm_capacity": PM_CAPACITY, "T": T, "x0": X0}}
    print(f"Інстанс: ВМ={VM_LOADS}, ФС={PM_CAPACITY}, T={T}, x0={X0} (ФС={active_pms(X0)})")
    try:
        from cp_sat import run_cp_sat
    except ImportError:
        run_cp_sat = None
        print("OR-Tools не встановлено — CP-SAT пропущено, точний оптимум з повного перебору")
    if with_qaoa:
        import warm_start_qaoa as ws

    for mu in MUS:
        t = time.time()
        row = {}
        fb, ab = brute_force(mu)
        row["exact"] = describe(ab, mu)
        row["x0"] = describe(X0, mu)
        if run_cp_sat:
            cp = run_cp_sat(x0=X0, mu=mu)
            row["CP-SAT"] = dict(describe(cp["assignment"], mu), status=cp["status"], runtime_s=cp["runtime_s"])
            assert abs(cp["objective"] - fb) < 1e-6, f"CP-SAT {cp['objective']} ≠ перебір {fb}"
        mb = run_mbfd()
        row["MBFD (повна переукладка)"] = describe(mb["assignment"], mu)
        runs = [run_brkga(seed=s, x0=X0, mu=mu) for s in SEEDS]
        row["BRKGA"] = dict(optimum_rate=sum(abs(r["objective"] - fb) < 1e-6 for r in runs) / len(runs),
                            mean_objective=round(statistics.mean(r["objective"] for r in runs), 4),
                            median_runtime_s=statistics.median(r["runtime_s"] for r in runs),
                            best=describe(min(runs, key=lambda r: r["objective"])["assignment"], mu))
        if with_qaoa:
            row["random_neighbor"] = {f"d={d}": ws.random_neighbor(mu, d) for d in (1, 2)}
            row["warm_qaoa"] = {f"p={p}": ws.run_warm(mu=mu, p=p) for p in (1, 2)}
        out[f"mu={mu}"] = row

        e, x = row["exact"], row["x0"]
        print(f"\n=== MU = {mu}:  F(x0) = {x['objective']}  |  оптимум F = {e['objective']} "
              f"→ {e['assignment']}: ФС={e['active_pms']}, міграцій={e['migrations']} (варт. {e['migration_cost']})")
        m = row["MBFD (повна переукладка)"]
        print(f"  MBFD:   ФС={m['active_pms']}, міграцій={m['migrations']} (варт. {m['migration_cost']}), F={m['objective']}")
        b = row["BRKGA"]
        print(f"  BRKGA:  оптимум F у {b['optimum_rate']:.0%} запусків, сер. F={b['mean_objective']}, "
              f"найкраще → ФС={b['best']['active_pms']}, міграцій={b['best']['migrations']}, час {b['median_runtime_s']:.3f} с")
        if with_qaoa:
            for d in (1, 2):
                r = row["random_neighbor"][f"d={d}"]
                print(f"  випадковий сусід d={d}: P(кращий за x0)={r['p_better']:.3f}, сер. варт. мігр.={r['mean_migc_better']:.2f}")
            for p in (1, 2):
                r = row["warm_qaoa"][f"p={p}"]
                bs = r["best_seen"]
                print(f"  warm QAOA p={p}: P(кращий за x0)={r['p_better']:.3f}, допуст.={r['p_feasible']:.3f}, "
                      f"сер. варт. мігр. (кращі)={r['mean_migc_better']:.2f}; найкращий за 1000 shots: "
                      + (f"ФС={bs['active_pms']}, міграцій={bs['migrations']}, F={bs['objective']:.3f} (P={bs['prob']:.3f})" if bs else "—")
                      + f"  [{r['runtime_s']:.0f} с]")
        print(f"  [{time.time()-t:.0f} с]")
        sys.stdout.flush()

    json.dump(out, open("results_migration.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False, default=str)
    print("\nЗбережено results_migration.json")


if __name__ == "__main__":
    main(with_qaoa="--no-qaoa" not in sys.argv)
