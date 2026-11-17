"""
СИМУЛЯЦІЯ ЖИТТЄВОГО ЦИКЛУ ЦОД (сценарій Жарікова 2026-11-01), класичні методи.

Кожен крок: (1) приходить пакет нових ВМ → розміщує АЛГОРИТМ 2 (Best Fit Decreasing на залишкових
місткостях, без міграцій); (2) частина ВМ завершує роботу (час життя — геометричний); (3) АЛГОРИТМ 1
(перерозподіл) за обраною стратегією, якщо спрацював тригер:
  - none       — ніколи не консолідувати (лише алгоритм 2);
  - repack     — повна переукладка BRKGA без урахування міграцій (мінімум ФС, міграції не рахуються);
  - mu=…       — BRKGA з ціллю F = ФС + mu · (зважені міграції) відносно поточного розподілу;
  - cpsat mu=… — те саме, розв'язане ТОЧНО через CP-SAT (cp_sat.solve_cpsat, ліміт 10 с на крок) — еталон.
Тригер перерозподілу (--trigger): every — щокроку; threshold:0.6 — коли середня завантаженість активних ФС
Σ навантажень / (активні ФС · T·C) падає нижче порога; period:3 — кожні k кроків.
Усі стратегії бачать ОДНАКОВІ прибуття і часи життя (згенеровано наперед), тому порівнювані.

Метрики: середня кількість активних ФС за кроками (і нижня межа ⌈Σs / (T·C)⌉), ФС·кроки, сумарна кількість
міграцій, сумарне мігроване навантаження (Σ s_j по переміщених ВМ — «вартість пропорційна розміру»), час.
З --seeds K — середнє ± std по K сценаріях (seed, seed+1, …).

Запуск: python lifecycle_sim.py [--steps 20] [--servers 12] [--arrivals 5] [--life 4] [--seed 1]
        python lifecycle_sim.py --seeds 10 --steps 20                  # усереднення + порівняння тригерів
        python lifecycle_sim.py --trigger threshold:0.6                # тригер для основної таблиці
        python lifecycle_sim.py --no-cpsat                             # без CP-SAT (швидко)
Результат: results_lifecycle.json = {instance, generated, git, params, strategies: {назва: {mean, std, per_seed}},
           triggers_mu1: {…}}
"""
import collections, json, math, sys, time, random, statistics
from brkga import run_brkga
from placement_new import best_fit_decreasing, residual

T = 0.9
CAP = 20
LOAD_RANGE = (2, 9)
METRICS = ("mean_active", "server_steps", "total_migrations", "total_migrated_load")


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


def triggered(trigger, step, vms):
    """Чи запускати алгоритм 1 на цьому кроці."""
    if trigger == "every":
        return True
    kind, val = trigger.split(":")
    if kind == "period":
        k = int(val)
        return (step + 1) % k == 0
    if kind == "threshold":
        active = len({v["server"] for v in vms})
        util = sum(v["load"] for v in vms) / (active * T * CAP) if active else 1.0
        return util < float(val)
    raise ValueError(f"невідомий тригер {trigger}")


def simulate(strategy, scenario, m, mu=1.0, brkga_seed=0, trigger="every", cpsat_limit=10.0):
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
        # (3) перерозподіл — алгоритм 1 (якщо спрацював тригер)
        mig_n = mig_load = 0
        ran = False
        status = None
        if vms and strategy != "none" and triggered(trigger, step, vms):
            ran = True
            loads = [v["load"] for v in vms]
            x0 = [v["server"] for v in vms]
            if strategy == "cpsat":
                from cp_sat import solve_cpsat
                r = solve_cpsat(loads, caps, x0=x0, mu=mu, time_limit=cpsat_limit, t=T)
                new = r["assignment"]
                status = r["status"]                 # OPTIMAL / FEASIBLE (ліміт часу) / ...
            else:
                r = run_brkga(loads=loads, caps=caps, x0=(None if strategy == "repack" else x0),
                              mu=(0.0 if strategy == "repack" else mu), seed=brkga_seed + step)
                new = r["assignment"] if r["feasible"] else None
            if new is not None:
                for v, s_new in zip(vms, new):
                    if s_new != v["server"]:
                        mig_n += 1; mig_load += v["load"]
                        v["server"] = s_new
        total = sum(v["load"] for v in vms)
        hist.append(dict(step=step, n_vms=len(vms), total_load=total,
                         active=len({v["server"] for v in vms}),
                         lower_bound=math.ceil(total / (T * CAP) - 1e-9) if total else 0,
                         migrations=mig_n, migrated_load=mig_load, reconsolidated=ran,
                         **({"cpsat_status": status} if status else {})))
    ok = [h for h in hist if "overflow" not in h]
    name = {"mu": f"mu={mu}", "cpsat": f"cpsat mu={mu}"}.get(strategy, strategy)
    return dict(strategy=name + ("" if trigger == "every" else f" [{trigger}]"),
                mean_active=sum(h["active"] for h in ok) / len(ok),
                mean_lower_bound=sum(h["lower_bound"] for h in ok) / len(ok),
                server_steps=sum(h["active"] for h in ok),
                total_migrations=sum(h["migrations"] for h in ok),
                total_migrated_load=sum(h["migrated_load"] for h in ok),
                total_load_steps=sum(h["total_load"] for h in ok),
                reconsolidations=sum(h["reconsolidated"] for h in ok),
                runtime_s=time.perf_counter() - t_start, history=hist,
                **({"cpsat_status": dict(collections.Counter(h["cpsat_status"] for h in ok if "cpsat_status" in h))}
                   if strategy == "cpsat" else {}))


def _agg(runs):
    """{mean, std, per_seed} по seed-ах для METRICS (+ нижня межа і час)."""
    keys = METRICS + ("mean_lower_bound", "reconsolidations", "runtime_s")
    mean = {k: statistics.mean(r[k] for r in runs) for k in keys}
    std = {k: (statistics.stdev(r[k] for r in runs) if len(runs) > 1 else 0.0) for k in keys}
    per = [dict({k: r[k] for k in keys}, seed=r["seed"], **({"cpsat_status": r["cpsat_status"]}
                                                            if "cpsat_status" in r else {})) for r in runs]
    return dict(mean=mean, std=std, per_seed=per)


def run_set(specs, scenarios, servers, label="", cache=None):
    """specs: список (strategy, mu, trigger). Повертає {назва: агрегат}; cache — уже пораховані (не повторюємо)."""
    out = {}
    for strategy, mu, trig in specs:
        name = {"mu": f"mu={mu}", "cpsat": f"cpsat mu={mu}"}.get(strategy, strategy) + ("" if trig == "every" else f" [{trig}]")
        if cache and name in cache:
            out[name] = cache[name]
            continue
        runs = []
        for seed, sc in scenarios:
            r = simulate(strategy, sc, servers, mu=mu, trigger=trig)
            r["seed"] = seed
            runs.append(r)
        name = runs[0]["strategy"]
        out[name] = _agg(runs)
        a = out[name]["mean"]
        print(f"  {label}{name:<30} ФС {a['mean_active']:.2f}  міграцій {a['total_migrations']:.1f}  "
              f"[{sum(r['runtime_s'] for r in runs):.0f} с]")
        sys.stdout.flush()
    return out


def table(strats, title):
    pm = lambda s, k, f: f"{s['mean'][k]:{f}} ± {s['std'][k]:{f}}"
    lines = [title, "", "| стратегія | сер. ФС | ниж. межа | ФС·кроки | міграцій | мігр. навант. | перерозп. | час, с |",
             "|---|---|---|---|---|---|---|---|"]
    for name, s in strats.items():
        lines.append(f"| {name} | {pm(s, 'mean_active', '.2f')} | {s['mean']['mean_lower_bound']:.2f} | "
                     f"{pm(s, 'server_steps', '.1f')} | {pm(s, 'total_migrations', '.1f')} | "
                     f"{pm(s, 'total_migrated_load', '.1f')} | {s['mean']['reconsolidations']:.1f} | "
                     f"{s['mean']['runtime_s']:.1f} |")
    return "\n".join(lines)


def main(steps=20, servers=12, arrivals=5, mean_life=4, seed=1, seeds=1, mus=(0.5, 1.0, 2.0), trigger="every",
         with_cpsat=True, cpsat_mus=(1.0,), reuse_cpsat=None):
    """reuse_cpsat — шлях до попереднього results_lifecycle.json: рядки CP-SAT беруться звідти (не перераховуються;
    CP-SAT з 8 потоками недетермінований, а BRKGA після виправлення декодера треба перерахувати)."""
    from config import save_results
    cache = {}
    if reuse_cpsat:
        old = json.load(open(reuse_cpsat, encoding="utf-8"))
        assert old["params"]["seeds"] == [seed + k for k in range(seeds)] and old["params"]["steps"] == steps
        for sec in ("strategies", "triggers_mu1"):
            cache.update({k: v for k, v in old[sec].items() if k.startswith("cpsat")})
    scenarios = [(seed + k, gen_scenario(steps, arrivals, mean_life, seed + k)) for k in range(seeds)]
    print(f"ЦОД: {servers} ФС × {CAP} (T={T} → {T*CAP:.0f}), кроків {steps}, прибуття {arrivals} ВМ/крок "
          f"(навантаження {LOAD_RANGE}), середній час життя {mean_life} кроки, seeds {seed}..{seed + seeds - 1}, "
          f"тригер {trigger}")
    specs = [("none", 1.0, "every"), ("repack", 1.0, trigger)] + [("mu", mu, trigger) for mu in mus]
    if with_cpsat:
        specs += [("cpsat", mu, trigger) for mu in cpsat_mus]
    strategies = run_set(specs, scenarios, servers, cache=cache)
    trig_specs = [("mu", 1.0, tg) for tg in ("every", "threshold:0.6", "period:3")]
    if with_cpsat:
        trig_specs += [("cpsat", 1.0, tg) for tg in ("every", "threshold:0.6", "period:3")]
    print("\nПорівняння тригерів, μ = 1:")
    triggers = run_set(trig_specs, scenarios, servers, cache={**cache, **strategies})
    print("\n" + table(strategies, f"Стратегії (тригер {trigger}), середнє ± std по {seeds} seed"))
    print("\n" + table(triggers, "Тригери перерозподілу, μ = 1"))
    print("\nФС·кроки — сумарна кількість увімкнених серверів за всі кроки (енергія ∝ цьому);"
          " мігр. навант. — Σ розмірів переміщених ВМ (вартість міграцій, пропорційна розміру).")
    fn = save_results("lifecycle", dict(params=dict(steps=steps, servers=servers, cap=CAP, T=T, arrivals=arrivals,
                                                    mean_life=mean_life, seeds=[s for s, _ in scenarios],
                                                    load_range=LOAD_RANGE, trigger=trigger, cpsat_time_limit_s=10.0),
                                        strategies=strategies, triggers_mu1=triggers),
                      instance=dict(name="lifecycle", servers=servers, cap=CAP, T=T))
    print(f"Збережено {fn}")


if __name__ == "__main__":
    arg = lambda k, d: type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
    main(steps=arg("--steps", 20), servers=arg("--servers", 12), arrivals=arg("--arrivals", 5),
         mean_life=arg("--life", 4), seed=arg("--seed", 1), seeds=arg("--seeds", 1), trigger=arg("--trigger", "every"),
         with_cpsat="--no-cpsat" not in sys.argv, reuse_cpsat=arg("--reuse-cpsat", "") or None)
