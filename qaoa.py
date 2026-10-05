"""
QAOA для задачі розміщення ВМ.

ВАЖЛИВО про розмір задачі:
Повний інстанс (6 ВМ, 3 ФС, 21 CQM-змінна) після точної CQM->BQM конверсії
(з slack-бітами для нерівності місткості) розростається до ~33 кубітів —
це більше за statevector-симуляцію (2^33) і більше за місткість реальних
гейтових QPU (IQM Garnet ≈ 20 кубітів). Тому:
  - повний інстанс порівнюється лише класичними методами (MBFD, CP-SAT, ГА);
  - для QAOA/реального QPU використовується ЗМЕНШЕНИЙ під-інстанс
    (перші 3 ВМ, 2 ФС з того самого набору даних, той самий T і λ) —
    це стандартна практика при оцінці масштабованості QAOA на NISQ-залізі
    (SLR вже зафіксував цю прогалину: мало робіт тестують реальний QPU
    саме через обмеження кубітів).
Рішення і причина задокументовані в Obsidian ("Методи порівняння").
"""

import time
import json

import numpy as np
from scipy.optimize import minimize

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp

from cqm_model import build_bqm, decode_sample_to_assignment
from config import VM_LOADS, PM_CAPACITY, T, LAMBDA1, is_feasible as is_feasible_full

# зменшений під-інстанс для QAOA / реального QPU
REDUCED_VM_LOADS = VM_LOADS[:3]      # [3, 4, 2]
REDUCED_PM_CAPACITY = PM_CAPACITY[:2]  # [12, 12]
REDUCED_N = len(REDUCED_VM_LOADS)
REDUCED_M = len(REDUCED_PM_CAPACITY)
REDUCED_EFFECTIVE_CAPACITY = [T * c for c in REDUCED_PM_CAPACITY]


def reduced_is_feasible(assignment):
    loads = [0.0] * REDUCED_M
    for j, i in enumerate(assignment):
        loads[i] += REDUCED_VM_LOADS[j]
    return all(loads[i] <= REDUCED_EFFECTIVE_CAPACITY[i] + 1e-9 for i in range(REDUCED_M))


def reduced_active_pms(assignment):
    return len(set(assignment))


def reduced_pm_loads(assignment):
    loads = [0.0] * REDUCED_M
    for j, i in enumerate(assignment):
        loads[i] += REDUCED_VM_LOADS[j]
    return loads


def bqm_to_ising_hamiltonian(bqm):
    """Конвертує BQM (binary) у SparsePauliOp Ising-гамільтоніан (spin), повертає (H, var_order, offset)."""
    ising = bqm.spin
    h, J, offset = ising.linear, ising.quadratic, ising.offset
    var_order = list(ising.variables)
    idx = {v: k for k, v in enumerate(var_order)}
    n = len(var_order)

    pauli_list = []
    for v, coeff in h.items():
        if coeff == 0:
            continue
        label = ["I"] * n
        label[idx[v]] = "Z"
        pauli_list.append(("".join(label), coeff))

    for (u, v), coeff in J.items():
        if coeff == 0:
            continue
        label = ["I"] * n
        label[idx[u]] = "Z"
        label[idx[v]] = "Z"
        pauli_list.append(("".join(label), coeff))

    hamiltonian = SparsePauliOp.from_list(pauli_list) if pauli_list else SparsePauliOp.from_list([("I" * n, 0.0)])
    return hamiltonian, var_order, offset


def qaoa_circuit(hamiltonian, n_qubits, betas, gammas):
    p = len(betas)
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    pauli_terms = list(zip(hamiltonian.paulis.to_labels(), hamiltonian.coeffs.real))

    for layer in range(p):
        gamma = gammas[layer]
        for label, coeff in pauli_terms:
            z_qubits = [n_qubits - 1 - k for k, c in enumerate(label) if c == "Z"]
            if len(z_qubits) == 0:
                continue
            elif len(z_qubits) == 1:
                qc.rz(2 * gamma * coeff, z_qubits[0])
            elif len(z_qubits) == 2:
                qc.cx(z_qubits[0], z_qubits[1])
                qc.rz(2 * gamma * coeff, z_qubits[1])
                qc.cx(z_qubits[0], z_qubits[1])
        beta = betas[layer]
        for q in range(n_qubits):
            qc.rx(2 * beta, q)

    return qc


def expectation(hamiltonian, params, var_order, backend, shots=2000):
    p = len(params) // 2
    betas, gammas = params[:p], params[p:]
    n_qubits = len(var_order)
    qc = qaoa_circuit(hamiltonian, n_qubits, betas, gammas)
    qc.measure_all()
    job = backend.run(qc, shots=shots)
    counts = job.result().get_counts()

    pauli_terms = list(zip(hamiltonian.paulis.to_labels(), hamiltonian.coeffs.real))
    total = 0.0
    n_shots = sum(counts.values())
    for bitstring, count in counts.items():
        bits = bitstring.replace(" ", "")[::-1]
        spins = {var_order[k]: (1 if bits[k] == "0" else -1) for k in range(n_qubits)}
        energy = 0.0
        for label, coeff in pauli_terms:
            prod = 1
            for k, c in enumerate(label):
                if c == "Z":
                    prod *= spins[var_order[n_qubits - 1 - k]]
            energy += coeff * prod
        total += energy * count
    return total / n_shots, counts


def run_qaoa(p=2, shots=2000, maxiter=50, seed=42):
    start = time.perf_counter()
    rng = np.random.default_rng(seed)

    bqm, invert, cqm = build_bqm(
        lagrange_multiplier=LAMBDA1,
        vm_loads=REDUCED_VM_LOADS,
        pm_capacity=REDUCED_PM_CAPACITY,
        t=T,
    )
    hamiltonian, var_order, offset = bqm_to_ising_hamiltonian(bqm)
    n_qubits = len(var_order)
    backend = AerSimulator(method="statevector")

    init_params = rng.uniform(0, np.pi, size=2 * p)
    history = []

    def objective(params):
        val, _ = expectation(hamiltonian, params, var_order, backend, shots=shots)
        history.append(val)
        return val

    opt_result = minimize(objective, init_params, method="COBYLA", options={"maxiter": maxiter})
    _, final_counts = expectation(hamiltonian, opt_result.x, var_order, backend, shots=max(shots, 4000))

    best_assignment, best_active, best_feasible_found = None, None, False
    for bitstring, count in sorted(final_counts.items(), key=lambda kv: -kv[1]):
        bits = bitstring.replace(" ", "")[::-1]
        # гамільтоніан з bqm.spin: spin +1 <-> binary 1, а Z|0> = +1 => binary = 1 - bit
        sample = {var_order[k]: (0 if bits[k] == "1" else 1) for k in range(n_qubits)}
        assignment = decode_sample_to_assignment(sample, invert=invert, n=REDUCED_N, m=REDUCED_M)
        if None not in assignment:
            feasible = reduced_is_feasible(assignment)
            if best_assignment is None or (feasible and not best_feasible_found) or (
                feasible and best_feasible_found and reduced_active_pms(assignment) < best_active
            ):
                best_assignment, best_active, best_feasible_found = assignment, reduced_active_pms(assignment), feasible
            if feasible:
                break

    elapsed = time.perf_counter() - start

    result = {
        "method": f"QAOA (p={p}, зменшений під-інстанс 3 ВМ/2 ФС, Aer local simulator, {n_qubits} кубітів)",
        "assignment": best_assignment,
        "active_pms": reduced_active_pms(best_assignment) if best_assignment and None not in best_assignment else None,
        "feasible": reduced_is_feasible(best_assignment) if best_assignment and None not in best_assignment else False,
        "pm_loads": reduced_pm_loads(best_assignment) if best_assignment and None not in best_assignment else None,
        "runtime_s": elapsed,
        "optimizer_final_energy": opt_result.fun,
        "optimizer_iterations": len(history),
        "shots": shots,
        "n_qubits": n_qubits,
        "note": "Повний 6ВМ/3ФС-інстанс дає ~33 кубіти після slack-конверсії — "
                "запущено на зменшеному під-інстансі (перші 3 ВМ, 2 ФС)",
    }
    return result, (hamiltonian, var_order, offset, opt_result.x.tolist())


if __name__ == "__main__":
    result, ising_data = run_qaoa()
    print(json.dumps(result, indent=2, ensure_ascii=False))

    hamiltonian, var_order, offset, opt_params = ising_data
    with open("qaoa_ising_params.json", "w") as f:
        json.dump(
            {
                "var_order": var_order,
                "offset": offset,
                "optimal_params": opt_params,
                "pauli_list": [
                    [label, coeff.real] for label, coeff in zip(hamiltonian.paulis.to_labels(), hamiltonian.coeffs)
                ],
            },
            f,
            indent=2,
        )
    print("\nIsing-параметри збережено у qaoa_ising_params.json (для прогону на IQM Garnet)")
