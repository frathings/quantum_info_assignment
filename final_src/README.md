## shor_algorithm_complete.ipynb

This notebook contains an implementation of **Shor’s factoring algorithm**. It integrates:

- the **classical preprocessing** step (random choice of $a$, $\gcd(a,N)$ check),
- **quantum order finding** via **Quantum Phase Estimation (QPE)** with controlled modular exponentiation,
- and the **classical post-processing** (continued fractions and $\gcd(a^{r/2}\pm 1, N)$) used to extract non-trivial factors.

The notebook also includes a **toy RSA application**, showing how factoring the RSA modulus $N=pq$ enables recovery of the private key and decryption of an encrypted message (for small, simulator-friendly parameters).

### Factoring test instances used in the notebook

The notebook validates the implementation on a set of **small composite moduli** $N=pq$ (products of two primes), chosen to be feasible in simulation while still exercising the full order-finding + continued-fractions workflow.  
For each instance, a coprime base $a$ is fixed and the QPE precision/simulation settings are tuned for stability.

| Case | $N$ (factorization) | Base $a$ | Counting qubits $n$ | Work register $n_x$ | Shots | Aer method |
|---|---:|---:|---:|---:|---:|---|
| 1 | $15 = 3\cdot 5$   | 7  | 5 | 12 | 2048 | `matrix_product_state` |
| 2 | $21 = 3\cdot 7$   | 2  | 6 | 15 | 2048 | `matrix_product_state` |
| 3 | $33 = 3\cdot 11$  | 7  | 6 | 15 | 4096 | `matrix_product_state` |
| 4 | $55 = 5\cdot 11$  | 2  | 6 | 15 | 4096 | `matrix_product_state` |
| 5 | $65 = 5\cdot 13$  | 11 | 7 | 22 | 4096 | `matrix_product_state` |
| 6 | $133 = 7\cdot 19$ | 2  | 8 | 20 | 4096 | `matrix_product_state` |


**Notes**
- $n$ controls the precision of phase estimation.  
- $n_x$ is the size of the work/auxiliary register used by the modular exponentiation block .
- The simulation backend method is selected to manage circuit size:
  - `matrix_product_state` is used for most cases as it scales better than full statevector for moderate qubit counts.
  - Running these circuits on actual quantum hardware is significantly more demanding than in simulation because Shor’s modular exponentiation requires deep, non-Clifford arithmetic.


---

