"""
СИМУЛЯЦІЯ ЖИТТЄВОГО ЦИКЛУ ЦОД (сценарій Жарікова 2026-11-01), класичні методи.

Кожен крок: (1) приходить пакет нових ВМ → розміщує АЛГОРИТМ 2 (Best Fit Decreasing на залишкових
місткостях, без міграцій); (2) частина ВМ завершує роботу (час життя — геометричний); (3) АЛГОРИТМ 1
(перерозподіл) за обраною стратегією:
  - none    — ніколи не консолідувати (лише алгоритм 2);
  - repack  — повна переукладка BRKGA без урахування міграцій (мінімум ФС, міграції не рахуються);
  - mu=…    — BRKGA з ціллю F = ФС + mu · (зважені міграції) відносно поточного розподілу.
Усі стратегії бачать ОДНАКОВІ прибуття і часи життя (згенеровано наперед), тому порівнювані.

Метрики: середня кількість активних ФС за кроками (і нижня межа ⌈Σs / (T·C)⌉), сумарна кількість міграцій,
сумарне мігроване навантаження (Σ s_j по переміщених ВМ — «вартість пропорційна розміру»), час.

Запуск: python lifecycle_sim.py [--steps 20] [--servers 12] [--arrivals 5] [--life 4] [--seed 1]
Результат: results_lifecycle.json
"""
import json, math, sys, time, random
from brkga import run_brkga
from placement_new import best_fit_decreasing, residual

T = 0.9
CAP = 20
LOAD_RANGE = (2, 9)


def gen_scenario(steps, arrivals, mean_life, seed):
    """Наперед: для кожного кроку список (навантаження, час життя у кроках)."""
    rng = random.Random(seed)
    sc = []
    for _ in range(steps):
        batch = []
        for _ in range(arrivals):
            load = rng.randint(*LOAD_RANGE)
            life = 1 + int(rng.expovariate(1.0 / max(mean_life - 1, 1e-9))) if mean_life > 1 else 1
            batch.append((load, life))
        sc.append(batch)
    return sc


def simulate(strategy, scenario, m, mu=1.0, brkga_seed=0):
    caps = [CAP] * m
    vms = []                     # dict(load, server, ttl)
    hist = []
    t_start = time.perf_counter()
    for step, batch in enumerate(scenario):
        # (1) нові ВМ — алгоритм 2
        existing = [[v["load"] for v in vms if v["server"] == i] for i in range(m)]
        res, active = residual(caps, existing, T)
        new_loads = [b[0] for b in batch]
        try:
            placed = best_fit_decreasing(new_loads, res, active)
        except RuntimeError:
            hist.append(dict(step=step, overflow=True)); break
        for (load, life), srv in zip(batch, placed):
            vms.append(dict(load=load, server=srv, ttl=life))
        # (2) завершення
        for v in vms:
            v["ttl"] -= 1
        vms = [v for v in vms if v["ttl"] > 0]
        # (3) перерозподіл — алгоритм 1
        mig_n = mig_load = 0
        if vms and strategy != "none":
            loads = [v["load"] for v in vms]
            x0 = [v["server"] for v in vms]
            r = run_brkga(loads=loads, caps=caps, x0=(None if strategy == "repack" else x0),
                          mu=(0.0 if strategy == "repack" else mu), seed=brkga_seed + step)
            if r["feasible"]:
                for v, s_new in zip(vms, r["assignment"]):
                    if s_new != v["server"]:
                        mig_n += 1; mig_load += v["load"]
                        v["server"] = s_new
        total = sum(v["load"] for v in vms)
        hist.append(dict(step=step, n_vms=len(vms), total_load=total,
                         active=len({v["server"] for v in vms}),
                         lower_bound=math.ceil(total / (T * CAP) - 1e-9) if total else 0,
                         migrations=mig_n, migrated_load=mig_load))
    ok = [h for h in hist if "overflow" not in h]
    return dict(strategy=strategy if strategy != "mu" else f"mu={mu}",
                mean_active=sum(h["active"] for h in ok) / len(ok),
                mean_lower_bound=sum(h["lower_bound"] for h in ok) / len(ok),
                server_steps=sum(h["active"] for h in ok),
                total_migrations=sum(h["migrations"] for h in ok),
                total_migrated_load=sum(h["migrated_load"] for h in ok),
                total_load_steps=sum(h["total_load"] for h in ok),
                runtime_s=time.perf_counter() - t_start, history=hist)


def main(steps=20, servers=12, arrivals=5, mean_life=4, seed=1, mus=(0.5, 1.0, 2.0)):
    scenario = gen_scenario(steps, arrivals, mean_life, seed)
    print(f"ЦОД: {servers} ФС × {CAP} (T={T} → {T*CAP:.0f}), кроків {steps}, прибуття {arrivals} ВМ/крок "
          f"(навантаження {LOAD_RANGE}), середній час життя {mean_life} кроки, seed={seed}")
    runs = [simulate("none", scenario, servers), simulate("repack", scenario, servers)]
    runs += [simulate("mu", scenario, servers, mu=mu) for mu in mus]
    print(f"\n{'стратегія':<12} {'сер. ФС':>8} {'ниж. межа':>10} {'ФС·кроки':>9} {'міграцій':>9} {'мігр. навант.':>14} {'час, с':>7}")
    for r in runs:
        print(f"{r['strategy']:<12} {r['mean_active']:>8.2f} {r['mean_lower_bound']:>10.2f} {r['server_steps']:>9d} "
              f"{r['total_migrations']:>9d} {r['total_migrated_load']:>14d} {r['runtime_s']:>7.1f}")
    print("\nФС·кроки — сумарна кількість увімкнених серверів за всі кроки (енергія ∝ цьому);"
          " мігр. навант. — Σ розмірів переміщених ВМ (вартість міграцій, пропорційна розміру).")
    json.dump(dict(params=dict(steps=steps, servers=servers, cap=CAP, T=T, arrivals=arrivals, mean_life=mean_life,
                               seed=seed, load_range=LOAD_RANGE), runs=runs),
              open("results_lifecycle.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("Збережено results_lifecycle.json")


if __name__ == "__main__":
    arg = lambda k, d: type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
    main(steps=arg("--steps", 20), servers=arg("--servers", 12), arrivals=arg("--arrivals", 5),
         mean_life=arg("--life", 4), seed=arg("--seed", 1))
