"""
ПОРІВНЯННЯ методів за ціллю з міграціями F = активні ФС + MU · зважені міграції (рішення Жарікова 2026-11-01).
Інстанс W (VMP_INSTANCE=W): ВМ [6,5,2,1], ФС [12,12,12], поточний розподіл x0 = [0,1,2,2] (3 ФС).
Для кожного MU ∈ {0, 0.5, 1, 2, 5} (при MU=5 оптимум — нічого не рухати):
  - точний оптимум F (повний перебір усіх M^N розподілів; для малих інстансів) і CP-SAT, якщо є OR-Tools;
  - «нічого не рухати» (x0) і MBFD як «повна переукладка» (міграцій не бачить);
  - BRKGA з ключами «залишитись», 20 запусків: частка запусків, що дійшли до оптимуму F;
  - класичний контроль: випадковий допустимий сусід на 1 і 2 міграції — P(кращий за x0);
  - warm-start QAOA p=1, 2 (точна симуляція, ціль «максимум P(кращий за x0)»): P(кращий), середня вартість
    міграцій серед кращих, найкращий стан, що з'явився б за 1000 shots; окремо — p=2 зі змішувачем,
    зваженим за розміром ВМ (mixer=size, alpha=2).
Запуск:  $env:VMP_INSTANCE="W"; python compare_migration.py            (результат — results_migration.json)
         python compare_migration.py --no-qaoa                          (лише класичні методи, швидко)

Режим --random K --seed S (статистика по стартових розподілах, 2026-11-15):
  K випадкових допустимих x0 з 3 активними ФС для інстансу config (W або $env:VMP_INSTANCE="L18"); для кожного
  x0 і MU ∈ {0.5, 1, 2}: перебір (оптимум F), BRKGA ×20, випадковий сусід d=1/2, warm QAOA p=1 і p=2 зі
  змішувачем size α=2 і p=2 зі звичайним rx. Агрегати по x0: середнє ± std P(кращий за x0), частка запусків
  BRKGA з оптимумом, частка x0, де найкращий стан QAOA за 1000 shots = оптимум F.
  Результати (з проміжними після кожного x0) — results_migration_random.json, ключ = інстанс.
         $env:VMP_INSTANCE="W";   python compare_migration.py --random 10 --seed 1
         $env:VMP_INSTANCE="L18"; python compare_migration.py --random 10 --seed 1 --restarts 10   # 18 кубітів: менше рестартів
"""
import os, sys, json, itertools, statistics, time, datetime
os.environ.setdefault("VMP_INSTANCE", "W")
from config import VM_LOADS, PM_CAPACITY, T, N, M, X0, is_feasible, active_pms, objective, n_migrations, migration_cost
from mbfd import run_mbfd
from brkga import run_brkga

MUS = [0.0, 0.5, 1.0, 2.0, 5.0]
SEEDS = range(20)


def brute_force(mu, x0=None):
    """Точний оптимум F повним перебором (M^N розподілів). x0 — None → config.X0."""
    x0 = X0 if x0 is None else x0
    best, arg = None, None
    for a in itertools.product(range(M), repeat=N):
        if not is_feasible(a):
            continue
        f = objective(a, x0, mu)
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
            row["warm_qaoa"]["p=2, size-mixer a=2"] = ws.run_warm(mu=mu, p=2, mixer="size", alpha=2.0)
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
            for key, r in row["warm_qaoa"].items():
                bs = r["best_seen"]
                print(f"  warm QAOA {key}: P(кращий за x0)={r['p_better']:.3f}, допуст.={r['p_feasible']:.3f}, "
                      f"сер. варт. мігр. (кращі)={r['mean_migc_better']:.2f}; найкращий за 1000 shots: "
                      + (f"ФС={bs['active_pms']}, міграцій={bs['migrations']}, F={bs['objective']:.3f} (P={bs['prob']:.3f})" if bs else "—")
                      + f"  [{r['runtime_s']:.0f} с]")
        print(f"  [{time.time()-t:.0f} с]")
        sys.stdout.flush()

    from config import save_results
    save_results("migration", out)
    print("\nЗбережено results_migration.json")


# ---------- режим --random: статистика по випадкових x0 ----------
RANDOM_MUS = [0.5, 1.0, 2.0]
QAOA_VARIANTS = [("p=1, size a=2", dict(p=1, mixer="size", alpha=2.0)),
                 ("p=2, size a=2", dict(p=2, mixer="size", alpha=2.0)),
                 ("p=2, rx", dict(p=2, mixer="rx"))]


def random_x0s(k, seed, active=3):
    """k різних допустимих розподілів рівно з `active` активними ФС (рівноймовірно серед усіх таких)."""
    import random
    pool = [list(a) for a in itertools.product(range(M), repeat=N) if is_feasible(a) and active_pms(a) == active]
    rng = random.Random(seed)
    return rng.sample(pool, min(k, len(pool))), len(pool)


def _ms(v):
    v = [x for x in v if x is not None]
    if not v:
        return None
    return dict(mean=round(statistics.mean(v), 4), std=round(statistics.stdev(v), 4) if len(v) > 1 else 0.0, n=len(v))


def aggregate(per_x0):
    agg = {}
    for mu in RANDOM_MUS:
        key = f"mu={mu}"
        rows = [r[key] for r in per_x0]
        a = dict(improvable_x0=sum(r["exact"]["objective"] < r["x0"]["objective"] - 1e-9 for r in rows),
                 BRKGA_optimum_rate=_ms([r["BRKGA"]["optimum_rate"] for r in rows]))
        for d in (1, 2):
            a[f"random d={d} P_better"] = _ms([r["random_neighbor"][f"d={d}"]["p_better"] for r in rows])
        for tag, _ in QAOA_VARIANTS:
            q = [r["warm_qaoa"][tag] for r in rows]
            a[f"QAOA {tag} P_better"] = _ms([x["p_better"] for x in q])
            a[f"QAOA {tag} best1000=opt"] = round(statistics.mean(x["best_is_optimum"] for x in q), 3)
        agg[key] = a
    return agg


def markdown_random(inst, agg, k):
    pm = lambda m: f"{m['mean']:.3f} ± {m['std']:.3f}" if m else "—"
    lines = [f"**{inst}** (K = {k} випадкових x0 з 3 активними ФС)", "",
             "| MU | x0, які можна покращити | BRKGA: частка запусків з оптимумом | випадк. сусід d=1 | d=2 | "
             "QAOA p=1 size | p=2 size | p=2 rx | QAOA best@1000 = опт. (p=1 size / p=2 size / p=2 rx) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for mu in RANDOM_MUS:
        a = agg[f"mu={mu}"]
        lines.append(f"| {mu:g} | {a['improvable_x0']}/{k} | {pm(a['BRKGA_optimum_rate'])} | "
                     f"{pm(a['random d=1 P_better'])} | {pm(a['random d=2 P_better'])} | "
                     + " | ".join(pm(a[f'QAOA {t} P_better']) for t, _ in QAOA_VARIANTS) + " | "
                     + " / ".join(f"{a[f'QAOA {t} best1000=opt']:.0%}" for t, _ in QAOA_VARIANTS) + " |")
    return "\n".join(lines)


def main_random(k=10, seed=1, restarts=25, out="results_migration_random.json"):
    import warm_start_qaoa as ws
    inst = os.environ.get("VMP_INSTANCE", "W")
    x0s, pool = random_x0s(k, seed)
    print(f"Інстанс {inst}: ВМ={VM_LOADS}, ФС={PM_CAPACITY}; допустимих x0 з 3 ФС: {pool}, беремо {len(x0s)} (seed={seed})")
    store = json.load(open(out, encoding="utf-8")) if os.path.exists(out) else {}
    store = store.get("runs", store) if "runs" in store else {k: v for k, v in store.items()
                                                              if isinstance(v, dict) and "per_x0" in v}
    entry = store.get(inst) if store.get(inst, {}).get("seed") == seed else None
    if entry is None:
        entry = dict(instance=dict(vm_loads=VM_LOADS, pm_capacity=PM_CAPACITY, T=T), seed=seed, mus=RANDOM_MUS,
                     brkga_runs=len(SEEDS), shots_best_seen=1000, qaoa_restarts=restarts, per_x0=[])
    done = [r["x0"] for r in entry["per_x0"]]
    t_all = time.time()
    for x0 in x0s:
        if x0 in done:
            continue
        t = time.time()
        rec = {"x0": x0}
        for mu in RANDOM_MUS:
            fb, ab = brute_force(mu, x0)
            row = dict(exact=dict(assignment=ab, objective=round(fb, 4)),
                       x0=dict(objective=round(objective(x0, x0, mu), 4)))
            runs = [run_brkga(seed=s, x0=x0, mu=mu) for s in SEEDS]
            row["BRKGA"] = dict(optimum_rate=sum(abs(r["objective"] - fb) < 1e-6 for r in runs) / len(runs))
            row["random_neighbor"] = {f"d={d}": ws.random_neighbor(mu, d, x0=x0) for d in (1, 2)}
            row["warm_qaoa"] = {}
            for tag, kw in QAOA_VARIANTS:
                r = ws.run_warm(mu=mu, x0=x0, fast=True, restarts=restarts, **kw)
                bs = r["best_seen"]
                row["warm_qaoa"][tag] = dict(p_better=r["p_better"], p_feasible=r["p_feasible"],
                                             best_seen=bs, best_is_optimum=bool(bs and abs(bs["objective"] - fb) < 1e-6),
                                             angles=r["angles"], runtime_s=round(r["runtime_s"], 1))
            rec[f"mu={mu}"] = row
            q = row["warm_qaoa"]
            print(f"  x0={x0} MU={mu}: F(x0)={row['x0']['objective']} опт={row['exact']['objective']}; "
                  f"BRKGA {row['BRKGA']['optimum_rate']:.0%}; d=1 {row['random_neighbor']['d=1']['p_better']:.3f} "
                  f"d=2 {row['random_neighbor']['d=2']['p_better']:.3f}; QAOA "
                  + ", ".join(f"{t}: {v['p_better']:.3f}{'*' if v['best_is_optimum'] else ''}" for t, v in q.items()))
            sys.stdout.flush()
        rec["runtime_s"] = round(time.time() - t, 1)
        entry["per_x0"].append(rec)
        entry["aggregate"] = aggregate(entry["per_x0"])
        entry["generated"] = datetime.datetime.now().isoformat(timespec="seconds")
        store[inst] = entry
        from config import save_results                                                   # проміжний результат
        save_results(out, dict(runs=store), instance=sorted(store))
        print(f"  [x0 {len(entry['per_x0'])}/{len(x0s)}: {rec['runtime_s']:.0f} с]")
    print("\n" + markdown_random(inst, entry["aggregate"], len(entry["per_x0"])))
    print(f"\nЗбережено {out} [{time.time() - t_all:.0f} с]")


if __name__ == "__main__":
    if "--random" in sys.argv:
        arg = lambda k, d: type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
        main_random(k=arg("--random", 10), seed=arg("--seed", 1), restarts=arg("--restarts", 25))
    else:
        main(with_qaoa="--no-qaoa" not in sys.argv)

