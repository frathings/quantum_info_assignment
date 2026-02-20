import time
import numpy as np
import matplotlib.pyplot as plt
from math import gcd
from fractions import Fraction

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator


from modexp import ModExpTools, binary_to_decimal


def build_shor_circuit(N: int, a: int, n: int, nx: int) -> QuantumCircuit:
    x    = QuantumRegister(nx,     'x')
    z    = QuantumRegister(n,      'z')
    a_reg= QuantumRegister(n,      'a')
    b    = QuantumRegister(n + 1,  'b')
    c    = QuantumRegister(n,      'c')
    bN   = QuantumRegister(n,      'bN')
    t    = QuantumRegister(1,      't')
    cr   = ClassicalRegister(nx,   'c1')

    qc = QuantumCircuit(x, z, a_reg, b, c, bN, t, cr)

    qc.x(z[0])

    tempN = N
    for i in range(n):
        if tempN % 2 != 0:
            qc.x(bN[i])
        tempN //= 2

    for i in range(nx):
        qc.h(x[i])

    qc.barrier()

    modexp = ModExpTools.create_mod_exp(n, N, a, nx)
    qc.append(modexp, list(x)+list(z)+list(a_reg)+list(b)+list(c)+list(bN)+list(t))

    qc.barrier()

    qft_inv = ModExpTools.myqft(nx).inverse()
    qc.append(qft_inv, list(x))

    qc.barrier()
    qc.measure(x, cr)
    return qc


def extract_period(measurement: str, nx: int, N: int, a: int):
    measured_value = binary_to_decimal(measurement)
    if measured_value == 0:
        return None
    phase = measured_value / (2 ** nx)
    frac  = Fraction(phase).limit_denominator(N)
    r     = frac.denominator
    return r if pow(a, r, N) == 1 else None


def factor_from_period(N: int, a: int, r: int):
    if r % 2 != 0:
        return None
    x = pow(a, r // 2, N)
    if x == N - 1:
        return None
    p, q = gcd(x - 1, N), gcd(x + 1, N)
    if   p > 1 and p < N and q > 1 and q < N: return (p, q)
    elif p > 1 and p < N: return (p, N // p)
    elif q > 1 and q < N: return (q, N // q)
    return None


def run_shor(N, a, n, nx, shots=2048,
             backend_method="matrix_product_state") -> dict:
    """Build, transpile and simulate — returns timing + success info."""
    backend = AerSimulator(method=backend_method)

    t0 = time.time()
    qc = build_shor_circuit(N, a, n, nx)
    t_qc = transpile(qc, backend)
    job  = backend.run(t_qc, shots=shots)
    counts = job.result().get_counts()
    elapsed = time.time() - t0

    success = False
    for meas in sorted(counts, key=counts.get, reverse=True)[:10]:
        r = extract_period(meas, nx, N, a)
        if r is not None:
            factors = factor_from_period(N, a, r)
            if factors is not None:
                success = True
                break

    return {"elapsed": elapsed, "success": success}


# Benchmark configuration
TEST_CASES = [
    # label,  N,   a,   n,  nx,  shots,   bits
    ("4-bit",  15,  7,   5,  12, 2048,    4),
    ("6-bit",  55,  2,   6,  15, 4096,    6),
    ("7-bit",  65,  11,  7,  22, 4096,    7),
    ("8-bit", 133,  2,   8,  20, 4096,    8),
]

N_RUNS = 10   # independent repetitions per configuration



# Run the benchmark
print("=" * 65)
print("  Shor's Algorithm — Scaling Benchmark")
print(f"  {N_RUNS} runs × {len(TEST_CASES)} configurations")
print("=" * 65)

results = {}   # label -> list of elapsed times

for label, N, a, n, nx, shots, bits in TEST_CASES:
    times = []
    print(f"\n▶  {label}  (N={N}, a={a}, n={n}, nx={nx})")
    for run in range(1, N_RUNS + 1):
        r = run_shor(N, a, n, nx, shots)
        times.append(r["elapsed"])
        status = "✓" if r["success"] else "✗"
        print(f"   run {run:2d}/{N_RUNS}  {status}  {r['elapsed']:7.2f} s")
    results[label] = {"bits": bits, "N": N, "times": np.array(times)}

print("\n" + "=" * 65)
print("  Benchmark complete — generating plot …")
print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

labels    = [tc[0] for tc in TEST_CASES]
bit_vals  = np.array([results[l]["bits"]           for l in labels])
N_vals    = np.array([results[l]["N"]              for l in labels])
means     = np.array([results[l]["times"].mean()   for l in labels])
stderrs   = np.array([results[l]["times"].std(ddof=1) / np.sqrt(N_RUNS)
                       for l in labels])

# O(L^3) reference — fit constant k on the smallest two points
k = np.mean(means[:2] / bit_vals[:2] ** 3)
L_line  = np.linspace(bit_vals.min() - 0.5, bit_vals.max() + 0.7, 300)
t_theory = k * L_line ** 3

# figure 
fig, ax = plt.subplots(figsize=(10, 6))

# theoretical curve
ax.plot(L_line, t_theory, color="steelblue", linestyle="--", linewidth=1.8,
        alpha=0.8, label=r"Theoretical $O(L^3)$", zorder=1)

# scatter — individual runs (jittered slightly for visibility)
rng = np.random.default_rng(42)
for label in labels:
    bits   = results[label]["bits"]
    times  = results[label]["times"]
    jitter = rng.uniform(-0.06, 0.06, size=N_RUNS)
    ax.scatter(np.full(N_RUNS, bits) + jitter, times,
               alpha=0.45, s=28, color="tomato", zorder=3)

# error-bar means
ax.errorbar(bit_vals, means, yerr=stderrs,
            fmt="o", color="crimson", markersize=9,
            markeredgecolor="black", markeredgewidth=0.8,
            capsize=5, capthick=1.5, linewidth=1.8,
            zorder=5, label=f"Mean ± SEM  ({N_RUNS} runs)")

# annotations
for i, label in enumerate(labels):
    ax.annotate(f"N={N_vals[i]}\n({bit_vals[i]} bit)",
                xy=(bit_vals[i], means[i]),
                xytext=(0, 18), textcoords="offset points",
                ha="center", fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color="grey", lw=0.8))

ax.set_yscale("log")
ax.set_xlabel("Number of bits  $L = \\lceil\\log_2 N\\rceil$", fontsize=13)
ax.set_ylabel("Simulation time  (seconds)", fontsize=13)
ax.set_title("Shor's Algorithm — Scaling with Number Size\n"
             r"(MPS simulation, $O(L^3)$ theoretical reference)",
             fontsize=13)
ax.grid(True, which="both", linestyle=":", alpha=0.45)
ax.legend(frameon=True, shadow=True, fontsize=11)
ax.set_xticks(bit_vals)

plt.tight_layout()
outfile = "shor_scaling.png"
plt.savefig(outfile, dpi=150)
print(f"\n  Plot saved → {outfile}")
plt.show()