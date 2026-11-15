"""
АЛГОРИТМ 2 — розміщення НОВИХ ВМ на частково заповнені сервери (сценарій Жарікова 2026-11-01:
ЦОД працює довго, частина ВМ вимкнулась, прийшли нові; їх розміщує окремий алгоритм, без міграцій).

Вхід: залишкові місткості серверів (після наявних ВМ), які сервери вже увімкнені, пакет нових ВМ.
Ціль: мінімум НОВИХ увімкнених серверів (уже активні коштують 0); міграцій немає за означенням.

Методи:
  - Best Fit Decreasing на залишкових місткостях (класична онлайн-евристика; спершу активні сервери,
    новий сервер вмикається лише коли ВМ нікуди не влазить);
  - повний перебір (точний оптимум, для пакетів до ~6 ВМ);
  - CP-SAT (точний, якщо є OR-Tools);
  - QAOA ЛИШЕ над новими ВМ (точна симуляція, холодний старт): кубіти = (нові ВМ) × (ФС) + (порожні ФС).
    Старі ВМ зафіксовані і в схему не входять, тому кількість кубітів не залежить від розміру ЦОД —
    для 2 нових ВМ на 3 ФС це 6 + (кількість порожніх ФС) ≤ 9 кубітів, що влазить в IQM Garnet (20).
    Місткість кодується штрафами за мінімальні порушуючі підмножини розміру 1 (ВМ не влазить у залишок)
    і 2 (пара разом не влазить); підмножини ≥ 3 не підтримуються (assert, як у qaoa_noslack).

Запуск: python placement_new.py            (приклад: 3 ФС по 12, наявні ВМ [6,2 | 5 | —], нові [3, 4])
        python placement_new.py --new 3,4,2

Залізо (2026-11-15, за зразком warm_start_real_qpu.py; пристрій BRAKET_DEVICE=garnet|emerald|forte, журнал
qpu_runs.json через devices.py). Схема — звичайний QAOA (холодний старт) над кубітами нових ВМ, кути з qaoa_new:
  python placement_new.py local [P] [SHOTS] [--new 3,4,2]     # безкоштовно: локальний симулятор Braket
  python placement_new.py control_local [P]                    # безкоштовно: випадкові кути
  python placement_new.py submit [P] [SHOTS] [--new 3,4,2]     # ПЛАТНО (за замовч. P=2, 1000 shots)
  python placement_new.py control [P] [SHOTS]                  # ПЛАТНО: та сама схема, ВИПАДКОВІ кути (700 shots)
  python placement_new.py fetch <task-arn> [--new 3,4,2]       # безкоштовно: забрати результат
Аналіз: P(оптимум), P(допустимі), baseline, найкращий стан, std.err. Приклад за замовч. — 7 кубітів, --new 3,4,2 — 10.
"""
import itertools, json, sys, time
import numpy as np
from scipy.optimize import minimize
from qaoa_exact import qaoa_state

T = 0.9
LAM = 3.0


def residual(caps, existing, t=T):
    """existing: список списків навантажень на кожному сервері. Повертає (залишки, прапорці активності)."""
    res = [t * c - sum(v) for c, v in zip(caps, existing)]
    active = [len(v) > 0 for v in existing]
    return res, active


def new_servers(assign, active):
    return len({i for i in assign if not active[i]})


# ---------- класика ----------
def best_fit_decreasing(new_loads, res, active):
    """Best Fit Decreasing: ВМ за спаданням; серед серверів, де влазить, беремо активний з найменшим залишком;
    якщо жоден активний не підходить — порожній сервер з найменшим залишком, що вміщує ВМ."""
    res, active = list(res), list(active)
    order = sorted(range(len(new_loads)), key=lambda j: -new_loads[j])
    assign = [None] * len(new_loads)
    for j in order:
        fits = [i for i in range(len(res)) if res[i] >= new_loads[j] - 1e-9]
        if not fits:
            raise RuntimeError(f"ВМ {new_loads[j]} нікуди не влазить")
        act = [i for i in fits if active[i]]
        pool = act if act else fits
        i = min(pool, key=lambda k: res[k])
        assign[j] = i; res[i] -= new_loads[j]; active[i] = True
    return assign


def brute_force(new_loads, res, active):
    best, arg = None, None
    for a in itertools.product(range(len(res)), repeat=len(new_loads)):
        ld = [0.0] * len(res)
        for j, i in enumerate(a):
            ld[i] += new_loads[j]
        if any(ld[i] > res[i] + 1e-9 for i in range(len(res))):
            continue
        k = new_servers(a, active)
        if best is None or k < best:
            best, arg = k, list(a)
    return best, arg


def cp_sat(new_loads, res, active, time_limit_s=10.0):
    from ortools.sat.python import cp_model
    k, m = len(new_loads), len(res)
    mdl = cp_model.CpModel()
    x = {(j, i): mdl.NewBoolVar(f"x{j}_{i}") for j in range(k) for i in range(m)}
    y = {i: mdl.NewBoolVar(f"y{i}") for i in range(m) if not active[i]}
    for j in range(k):
        mdl.Add(sum(x[j, i] for i in range(m)) == 1)
    for i in range(m):
        cap = int(round(100 * res[i]))
        if active[i]:
            mdl.Add(sum(int(round(100 * new_loads[j])) * x[j, i] for j in range(k)) <= cap)
        else:
            mdl.Add(sum(int(round(100 * new_loads[j])) * x[j, i] for j in range(k)) <= cap * y[i])
    mdl.Minimize(sum(y.values()))
    s = cp_model.CpSolver(); s.parameters.max_time_in_seconds = time_limit_s
    st = s.Solve(mdl)
    a = [next(i for i in range(m) if s.Value(x[j, i])) for j in range(k)]
    return a, s.StatusName(st)


# ---------- QAOA лише над новими ВМ ----------
def build_new(new_loads, res, active, lam=LAM):
    """Енергія над станами: кубіти x_{j,i} (j — нова ВМ, i — сервер) + y_i для ПОРОЖНІХ серверів.
    H = Σ_{i порожній} y_i + lam·Σ_j (Σ_i x_ji − 1)² + lam·Σ_{j, i порожній} x_ji(1 − y_i)
        + lam·Σ_{j,i: s_j > res_i} x_ji + lam·Σ_{i, (j,k): s_j+s_k > res_i} x_ji x_ki."""
    k, m = len(new_loads), len(res)
    empty = [i for i in range(m) if not active[i]]
    yi = {i: k * m + t for t, i in enumerate(empty)}
    n = k * m + len(empty)
    xi = lambda j, i: j * m + i
    for r in range(3, k + 1):                                   # перевірка: порушуючі підмножини ≤ 2
        for S in itertools.combinations(range(k), r):
            for i in range(m):
                if sum(new_loads[j] for j in S) > res[i] + 1e-9:
                    assert any(new_loads[a] > res[i] + 1e-9 for a in S) or \
                           any(new_loads[a] + new_loads[b] > res[i] + 1e-9 for a, b in itertools.combinations(S, 2)), \
                        "мінімальна порушуюча підмножина ≥ 3 — парних штрафів недостатньо"
    St = np.arange(2 ** n)
    X = ((St[:, None] >> (n - 1 - np.arange(n))[None, :]) & 1).astype(np.int8)
    E = np.zeros(2 ** n)
    for i in empty:
        E += X[:, yi[i]]
    for j in range(k):
        E += lam * (sum(X[:, xi(j, i)] for i in range(m)) - 1) ** 2
        for i in empty:
            E += lam * X[:, xi(j, i)] * (1 - X[:, yi[i]])
        for i in range(m):
            if new_loads[j] > res[i] + 1e-9:
                E += lam * X[:, xi(j, i)]
    for i in range(m):
        for a, b in itertools.combinations(range(k), 2):
            if new_loads[a] + new_loads[b] > res[i] + 1e-9:
                E += lam * X[:, xi(a, i)] * X[:, xi(b, i)]
    xs = np.stack([np.stack([X[:, xi(j, i)] for i in range(m)], 1) for j in range(k)], 1)   # (S,k,m)
    onehot = (xs.sum(2) == 1).all(1)
    assign = xs.argmax(2)
    ld = np.stack([((assign == i) * np.array(new_loads, dtype=float)[None, :]).sum(1) for i in range(m)], 1)
    capok = (ld <= np.array(res)[None, :] + 1e-9).all(1)
    linked = np.ones(2 ** n, bool)
    for i in empty:
        linked &= (xs[:, :, i].sum(1) == 0) | (X[:, yi[i]] == 1)
    feasible = onehot & capok & linked
    obj = np.array([new_servers(a, active) for a in assign])
    best = obj[feasible].min()
    optimal = feasible & (obj == best)
    return n, E, onehot, feasible, optimal, best


def qaoa_new(new_loads, res, active, ps=(1, 2, 3), restarts=30, seed=0):
    n, E, onehot, feasible, optimal, best = build_new(new_loads, res, active)
    Ez = (E - E.min()) / (E.max() - E.min())
    rng = np.random.default_rng(seed)
    out = dict(n_qubits=n, optimum_new_servers=int(best), baseline_feasible=float(feasible.mean()),
               baseline_optimal=float(optimal.mean()), by_p={})
    for p in ps:
        bestr = None
        for _ in range(restarts):
            x0 = np.concatenate([rng.uniform(0, np.pi / 2, p), rng.uniform(0, 2 * np.pi, p)])
            r = minimize(lambda x: float((np.abs(qaoa_state(Ez, x, n)) ** 2) @ Ez), x0, method="L-BFGS-B")
            if bestr is None or r.fun < bestr.fun:
                bestr = r
        pr = np.abs(qaoa_state(Ez, bestr.x, n)) ** 2
        out["by_p"][p] = dict(P_feasible=float(pr[feasible].sum()), P_optimal=float(pr[optimal].sum()),
                              angles=bestr.x.tolist())
    return out


# ---------- приклад ----------
EXAMPLE = dict(caps=[12, 12, 12], existing=[[6, 2], [5], []], new=[3, 4])


def main(caps, existing, new):
    res, active = residual(caps, existing)
    print(f"ФС={caps}, наявні ВМ по серверах={existing}, залишки={[round(r, 1) for r in res]}, активні={active}")
    print(f"нові ВМ: {new}")
    kbf, abf = brute_force(new, res, active)
    print(f"перебір:  нових серверів = {kbf}, розміщення {abf}")
    a = best_fit_decreasing(new, res, active)
    out = dict(caps=caps, existing=existing, new=new, residual=res, active=active,
               brute_force=dict(new_servers=kbf, assignment=abf),
               best_fit_decreasing=dict(new_servers=new_servers(a, active), assignment=a))
    print(f"Best Fit Decreasing: нових серверів = {new_servers(a, active)}, розміщення {a}")
    try:
        a, st = cp_sat(new, res, active)
        out["CP-SAT"] = dict(new_servers=new_servers(a, active), status=st, assignment=a)
        print(f"CP-SAT:   нових серверів = {new_servers(a, active)} ({st}), розміщення {a}")
    except ImportError:
        print("CP-SAT:   OR-Tools не встановлено, пропущено")
    t = time.time()
    q = qaoa_new(new, res, active)
    out["QAOA"] = q
    print(f"QAOA лише над новими ВМ: кубітів = {q['n_qubits']} (повна задача мала б "
          f"{(sum(len(v) for v in existing) + len(new)) * len(caps) + len(caps)}), оптимум = {q['optimum_new_servers']} новий сервер; "
          f"baseline допустимі {q['baseline_feasible']:.3f}, оптимальні {q['baseline_optimal']:.3f}")
    for p, r in q["by_p"].items():
        print(f"  p={p}: P(допустимі)={r['P_feasible']:.3f}  P(оптимум)={r['P_optimal']:.3f}")
    print(f"  [{time.time() - t:.0f} с]")
    from config import save_results
    print("Збережено", save_results("placement_new", out, instance=dict(name="placement_new", caps=caps,
                                                                         existing=existing, new=new)))


# ---------- залізо: схема QAOA над новими ВМ (Braket) ----------
def ising_terms(Ez, n):
    """Розклад Ez на Z_k і Z_k Z_l (точний: H квадратичний). Z-рядки ортогональні → коефіцієнт = <Ez, Z>/2^n."""
    S = np.arange(2 ** n)
    Zs = 1 - 2 * ((S[:, None] >> (n - 1 - np.arange(n))[None, :]) & 1)       # кубіт 0 = старший біт
    terms, rec = [], np.full(2 ** n, Ez.mean())
    for k in range(n):
        c = float(Ez @ Zs[:, k]) / 2 ** n
        if abs(c) > 1e-12: terms.append(((k,), c)); rec += c * Zs[:, k]
    for k, l in itertools.combinations(range(n), 2):
        c = float(Ez @ (Zs[:, k] * Zs[:, l])) / 2 ** n
        if abs(c) > 1e-12: terms.append(((k, l), c)); rec += c * Zs[:, k] * Zs[:, l]
    assert np.allclose(rec, Ez, atol=1e-9), "H не квадратичний?"
    return terms


def braket_circuit(params, terms, n):
    from braket.circuits import Circuit
    p = len(params) // 2
    betas, gammas = params[:p], params[p:]
    c = Circuit()
    for q in range(n):
        c.h(q)
    for l in range(p):
        for lab, coef in terms:
            if len(lab) == 1:
                c.rz(lab[0], 2 * gammas[l] * coef)
            else:
                a, b = lab
                c.cnot(a, b); c.rz(b, 2 * gammas[l] * coef); c.cnot(a, b)
        for q in range(n):
            c.rx(q, 2 * betas[l])
    return c


def analyze_counts(counts, prob, label):
    """Braket-counts: рядок НЕ перевертаємо (кубіт 0 = перший символ = старший біт індексу)."""
    n, onehot, feasible, optimal, assign, nsrv = (prob[k] for k in ("n", "onehot", "feasible", "optimal", "assign", "nsrv"))
    shots = sum(counts.values())
    pr = np.zeros(2 ** n)
    for bs, c in counts.items():
        pr[int(bs, 2)] += c
    pr = pr / shots
    po = float(pr[optimal].sum())
    seen = feasible & (pr > 0)
    best = None
    if seen.any():
        i = np.flatnonzero(seen)[np.argmin(nsrv[seen])]
        best = dict(assignment=assign[i].tolist(), new_servers=int(nsrv[i]), count=int(round(pr[i] * shots)))
    out = dict(label=label, shots=shots, n_qubits=n, P_optimal=po, std_err_optimal=float(np.sqrt(po * (1 - po) / shots)),
               P_feasible=float(pr[feasible].sum()), P_onehot=float(pr[onehot].sum()),
               baseline_optimal=float(optimal.mean()), baseline_feasible=float(feasible.mean()), best_seen=best)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return out


def hw_problem(new, caps=EXAMPLE["caps"], existing=EXAMPLE["existing"]):
    res, active = residual(caps, existing)
    n, E, onehot, feasible, optimal, best = build_new(new, res, active)
    Ez = (E - E.min()) / (E.max() - E.min())
    k, m = len(new), len(res)
    S = np.arange(2 ** n)
    X = (S[:, None] >> (n - 1 - np.arange(n))[None, :]) & 1
    assign = X[:, :k * m].reshape(-1, k, m).argmax(2)
    nsrv = np.array([new_servers(a, active) for a in assign])
    return dict(n=n, Ez=Ez, onehot=onehot, feasible=feasible, optimal=optimal, best=best, assign=assign, nsrv=nsrv,
                res=res, active=active, terms=ising_terms(Ez, n))


def hw_main(argv):
    import devices
    mode = argv[1]
    pos, skip = [], False                      # позиційні аргументи без значень прапорців --key value
    for a in argv[2:]:
        if skip: skip = False; continue
        if a.startswith("--"): skip = True; continue
        pos.append(a)
    new = [float(v) for v in argv[argv.index("--new") + 1].split(",")] if "--new" in argv else EXAMPLE["new"]
    tag = "new" + "-".join(f"{v:g}" for v in new)
    dev_key, dev = devices.selected()
    prob = hw_problem(new)
    n = prob["n"]
    if mode == "fetch":
        from braket.aws import AwsQuantumTask
        task = AwsQuantumTask(pos[0], aws_session=devices.session(dev))
        st = task.state(); print("Статус:", st)
        if st in ("QUEUED", "CREATED"):
            try: print("Позиція в черзі:", task.queue_position().queue_position)
            except Exception as e: print("позиція недоступна:", e)
            return
        if st != "COMPLETED":
            print("Результату немає"); return
        out = analyze_counts(task.result().measurement_counts, prob, dev["name"])
        kind = devices.kind_of(pos[0])
        fn = f"newvm_qpu_result_{tag}_{dev_key}_{kind}.json"
        json.dump(out, open(fn, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        devices.log_result(pos[0], out)
        print("збережено", fn)
        return

    p = int(pos[0]) if pos else 2
    is_control = mode in ("control", "control_local")
    shots = int(pos[1]) if len(pos) > 1 else (700 if is_control else 1000)
    if is_control:
        rng = np.random.default_rng(123)            # фіксований seed → відтворюваний контроль
        params = np.concatenate([rng.uniform(0, np.pi / 2, p), rng.uniform(0, 2 * np.pi, p)])
    else:
        params = np.array(qaoa_new(new, prob["res"], prob["active"], ps=(p,))["by_p"][p]["angles"])
    circ = braket_circuit(params, prob["terms"], n)
    ideal = np.abs(qaoa_state(prob["Ez"], params, n)) ** 2
    n2q = sum(len(l) == 2 for l, _ in prob["terms"]) * 2 * p
    print(f"нові ВМ {new}: кубітів {n}, p={p}, CNOT={n2q}; ідеальні P(оптимум)={ideal[prob['optimal']].sum():.4f}, "
          f"P(допустимі)={ideal[prob['feasible']].sum():.4f}; baseline оптимум {prob['optimal'].mean():.4f}; "
          f"кути {np.round(params, 3).tolist()}")
    if mode in ("local", "control_local"):
        from braket.devices import LocalSimulator
        out = analyze_counts(LocalSimulator().run(circ, shots=20000).result().measurement_counts, prob,
                             "Braket local simulator (noiseless)")
        if not is_control:                           # тест конвенції бітів: локальний симулятор = точний numpy
            po = ideal[prob["optimal"]].sum()
            assert abs(out["P_optimal"] - po) < 4 * np.sqrt(po * (1 - po) / 20000) + 1e-3, "конвенція бітів?"
            print(f"конвенція ок: локальний симулятор {out['P_optimal']:.4f} vs numpy {po:.4f}")
    elif mode in ("submit", "control"):
        sess = devices.session(dev)
        device = devices.open_device(dev, sess)
        devices.check_ready(device, n)
        est = devices.confirm(dev, shots)
        task = device.run(circ, shots=shots)
        print("Task ID:", task.id)
        open(("newvm_control_task_" if is_control else "newvm_task_") + f"{tag}_{dev_key}.txt", "w").write(task.id)
        devices.log_submit(task.id, dev_key, "placement_new.py", shots, est, instance=tag, new=new, p=p,
                           control=is_control, angles=np.round(params, 4).tolist())
        print(f"Task ID збережено і в qpu_runs.json. Результат: BRAKET_DEVICE={dev_key} python placement_new.py "
              f"fetch <повний ARN> --new {','.join(f'{v:g}' for v in new)}")
    else:
        sys.exit(f"невідомий режим {mode}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("local", "control_local", "submit", "control", "fetch"):
        hw_main(sys.argv)
    else:
        ex = dict(EXAMPLE)
        if "--new" in sys.argv:
            ex["new"] = [float(v) for v in sys.argv[sys.argv.index("--new") + 1].split(",")]
        main(**ex)
