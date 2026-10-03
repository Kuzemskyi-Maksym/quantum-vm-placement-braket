"""
Метод #5: аналог SimulatedAnnealingSampler.

SimulatedAnnealingSampler (dimod) сам по собі ЧИСТО класичний (локальна
емуляція відпалу, не апаратний квантовий відпал) — його запуск на реальному
залізі AWS Braket напряму неможливий: жодного пристрою "quantum annealer"
у Braket немає (D-Wave вибули з каталогу Braket; Braket пропонує лише
гейтові QPU — IQM, Rigetti, IonQ — і нейтрально-атомний QuEra Aquila).

Рішення (задокументоване і в Obsidian "Методи порівняння" як відкрите
питання, тут — фактична реалізація):
  (a) baseline: SimulatedAnnealingSampler (dimod) — чиста класична
      симуляція відпалу на тому самому BQM, що й QAOA/CQM-модель.
  (b) апаратний аналог: QuEra Aquila (Analog Hamiltonian Simulation,
      нейтральні атоми, us-east-1) природно реалізує Maximum Independent
      Set (MIS) через Rydberg blockade — це НЕ симуляція відпалу, а окремий
      analog-квантовий підхід, який справді існує як реальне залізо.
      Задачу розміщення ВМ можна переформулювати як MIS (граф конфліктів
      між парами (ВМ,ФС), де ребро = вони не можуть одночасно бути "1" без
      порушення обмежень) — це більший обсяг роботи (окрема QUBO->MIS
      редукція), тому в цьому проході реалізовано (a), а (b) лишається
      наступним кроком (вже є в Obsidian з дедлайном).
"""

import time

import dimod
from dwave.samplers import SimulatedAnnealingSampler

from cqm_model import build_bqm, decode_sample_to_assignment
from config import is_feasible, active_pms, pm_loads, N, M


def run_simulated_annealing(num_reads=1000, num_sweeps=5000, seed=42):
    start = time.perf_counter()

    bqm, invert, cqm = build_bqm()  # повний інстанс (6 ВМ, 3 ФС)
    sampler = SimulatedAnnealingSampler()
    sampleset = sampler.sample(bqm, num_reads=num_reads, num_sweeps=num_sweeps, seed=seed)

    best_assignment, best_active, best_feasible_found = None, None, False
    for sample, energy in zip(sampleset.samples(), sampleset.record.energy):
        assignment = decode_sample_to_assignment(dict(sample), invert=invert, n=N, m=M)
        if None in assignment:
            continue
        feasible = is_feasible(assignment)
        active = active_pms(assignment)
        if best_assignment is None or (feasible and not best_feasible_found) or (
            feasible and best_feasible_found and active < best_active
        ):
            best_assignment, best_active, best_feasible_found = assignment, active, feasible

    elapsed = time.perf_counter() - start

    return {
        "method": "SimulatedAnnealingSampler (dimod/neal) — чиста класична емуляція відпалу; "
                  "апаратного квантового відпалу в AWS Braket немає (D-Wave вибули з каталогу), "
                  "апаратний аналог-кандидат: QuEra Aquila через MIS-редукцію (наступний крок)",
        "assignment": best_assignment,
        "active_pms": active_pms(best_assignment) if best_assignment else None,
        "feasible": is_feasible(best_assignment) if best_assignment else False,
        "pm_loads": pm_loads(best_assignment) if best_assignment else None,
        "runtime_s": elapsed,
        "num_reads": num_reads,
    }


if __name__ == "__main__":
    result = run_simulated_annealing()
    print(result)
