import json, numpy as np
from qiskit_aer import AerSimulator
from qaoa import bqm_to_ising_hamiltonian, qaoa_circuit, REDUCED_VM_LOADS, REDUCED_PM_CAPACITY, REDUCED_N, REDUCED_M, reduced_is_feasible, reduced_active_pms
from cqm_model import build_bqm, decode_sample_to_assignment
from config import T, LAMBDA1
saved = json.load(open("qaoa_ising_params.json"))
opt = saved["optimal_params"]; p = len(opt)//2
bqm, invert, _ = build_bqm(lagrange_multiplier=LAMBDA1, vm_loads=REDUCED_VM_LOADS, pm_capacity=REDUCED_PM_CAPACITY, t=T)
H, vo, off = bqm_to_ising_hamiltonian(bqm)
real_idx = [k for k,v in enumerate(vo) if not v.startswith("slack")]
qc = qaoa_circuit(H, len(vo), opt[:p], opt[p:]); qc.measure_all()
def stats(counts):
    tot = sum(counts.values()); valid = one = 0
    for b,c in counts.items():
        bits = b.replace(" ","")[::-1]
        s = {vo[k]: int(bits[k]) for k in real_idx}
        a = decode_sample_to_assignment(s, invert=None, n=REDUCED_N, m=REDUCED_M)
        if None in a: continue
        valid += c
        if reduced_is_feasible(a) and reduced_active_pms(a)==1: one += c
    return valid/tot, one/tot, len(counts)
be = AerSimulator(method="statevector")
for shots in (100, 100000):
    r = stats(be.run(qc, shots=shots, seed_simulator=1).result().get_counts())
    print(f"Aer noiseless shots={shots}: valid={r[0]:.3f} optimal(1 PM)={r[1]:.3f} unique={r[2]}")
print("uniform random baseline: valid=0.125 optimal(1 PM)=0.031")

