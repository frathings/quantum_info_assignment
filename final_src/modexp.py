from numpy import pi
from typing import Optional
from qiskit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from sympy import mod_inverse  


class ModExpTools:
    '''Collection of tools for building modular exponentiation circuits used in Shor's algorithm.'''
    @staticmethod
    def binary_to_decimal(binary_str: str) -> int:
        """
        Convert a binary string to its decimal representation.
        """
        decimal_value = 0
        for i, digit in enumerate(reversed(binary_str)):
            decimal_value += int(digit) * (2 ** i)
        return decimal_value

    ########################### arithmetic gates ##########################
    @staticmethod 
    def create_sum(name: Optional[str] = None):
        """
        3-qubit XOR gate for quantum addition: sum = a XOR b XOR carry_in.
        """
        qc = QuantumCircuit(3)
        qc.cx(1, 2)  # xor a into sum to accumulate the partial result
        qc.cx(0, 2)  # xor carry_in into sum to complete the full-adder logic
        return qc.to_gate(label=name or 'sum')

    @staticmethod
    def create_sum_inv(name: Optional[str] = None):
        """
        Inverse of the SUM gate. Needed to uncompute sum ancillas
        and restore reversibility in the ripple-carry adder.
        """
        qc = QuantumCircuit(3)
        qc.append(ModExpTools.create_sum().inverse(), range(3))
        return qc.to_gate(label=name or 'sum_inv')

    @staticmethod
    def create_carrier(name: Optional[str] = None):
        """
        4-qubit CARRY gate for quantum addition.
        Propagates the carry bit forward without destroying the input qubits,
        so the computation stays reversible.
        """
        qc = QuantumCircuit(4)
        qc.ccx(1, 2, 3)  # set carry_out if both a and b are 1
        qc.cx(1, 2)      # mix a into b so the next Toffoli sees (a XOR b), enabling carry detection
        qc.ccx(0, 2, 3)  # set carry_out if carry_in propagates through the mixed bit
        return qc.to_gate(label=name or 'carrier')

    @staticmethod
    def create_carrier_inv(name: Optional[str] = None):
        """
        Inverse of the CARRY gate. Used in the backward pass of the adder
        to uncompute carry ancillas and restore them to |0>.
        """
        qc = QuantumCircuit(4)
        qc.append(ModExpTools.create_carrier().inverse(), range(4))
        return qc.to_gate(label=name or 'carrier_inv')

    @staticmethod
    def create_adder(n: int):
        """
        n-bit quantum ripple-carry adder: |a, b> → |a, a + b>.
        Uses a forward carry-propagation pass followed by a backward
        pass to uncompute carries, keeping all ancillas clean.
        """
        a = QuantumRegister(n, 'a')
        b = QuantumRegister(n + 1, 'b')
        c = QuantumRegister(n, 'c')
        qc = QuantumCircuit(a, b, c)

        # forward pass: propagate carries from LSB (dx) to MSB (sx)
        for i in range(n - 1):
            qc.append(ModExpTools.create_carrier(), [c[i], a[i], b[i], c[i + 1]])
        
        # handle the final carry separately because it overflows into b[n]
        qc.append(ModExpTools.create_carrier(), [c[n - 1], a[n - 1], b[n - 1], b[n]])

        # compute the MSB of the sum before uncomputing the carry
        qc.cx(a[n - 1], b[n - 1])
        qc.append(ModExpTools.create_sum(), [c[n - 1], a[n - 1], b[n - 1]])

        # backward pass: compute remaining sums and restore carry ancillas to |0>
        for i in reversed(range(n - 1)):
            qc.append(ModExpTools.create_carrier_inv(), [c[i], a[i], b[i], c[i + 1]])
            qc.append(ModExpTools.create_sum(), [c[i], a[i], b[i]])

        return qc.to_gate(label='adder')

    @staticmethod
    def create_adder_inv(n: int):
        """
        Inverse of the quantum adder. Used to uncompute additions
        inside modular arithmetic blocks.
        """
        qc = QuantumCircuit(3 * n + 1)
        qc.append(ModExpTools.create_adder(n).inverse(), range(3 * n + 1))
        return qc.to_gate(label='adder_inv')

    # MODULAR ARITHMETIC ################

    @staticmethod
    def create_adder_mod(n: int, factorN: int):
        """
        Quantum modular adder: |a, b> → |a, a + b mod N>.
        Subtracts N after addition to wrap around, then uses a temporary
        flag to conditionally add N back if the result went negative.
        """
        a = QuantumRegister(n, 'a')
        b = QuantumRegister(n + 1, 'b')
        c = QuantumRegister(n, 'c')
        bN = QuantumRegister(n, 'factorN')
        t = QuantumRegister(1, 't')
        qc = QuantumCircuit(a, b, c, bN, t)

        # step 1: add a to b to get the raw (unreduced) sum
        qc.append(ModExpTools.create_adder(n), list(a) + list(b) + list(c))
        
        # swap a and factorN so we can reuse the adder to subtract N
        for i in range(n):
            qc.swap(a[i], bN[i])
        
        # subtract N from the sum; if the result is negative, b[n] will be 0
        qc.append(ModExpTools.create_adder_inv(n), list(a) + list(b) + list(c))

        # detect underflow: b[n]=0 means the subtraction went negative, so we need to add N back
        qc.x(b[n])         # invert so that the underflow condition drives t high
        qc.cx(b[n], t[0])  # record underflow in t so we can condition the correction on it
        qc.x(b[n])         # restore overflow bit to avoid corrupting subsequent gates

        # conditionally reconstruct N in register a so the next adder can add it back
        tempN = factorN
        i = 0
        while tempN != 0:
            if tempN % 2 != 0:
                qc.cx(t[0], a[i])  # flip the i-th bit of a only if underflow occurred
            i += 1
            tempN //= 2

        # add (the conditionally restored) N back to b to fix the underflow
        qc.append(ModExpTools.create_adder(n), list(a) + list(b) + list(c))

        # uncompute the conditional flips in a to restore it to its original value
        tempN = factorN
        i = 0
        while tempN != 0:
            if tempN % 2 != 0:
                qc.cx(t[0], a[i])
            i += 1
            tempN //= 2

        # restore the original a <-> factorN swap so all registers are back to their roles
        for i in range(n):
            qc.swap(a[i], bN[i])

        # uncompute the intermediate addition to clean up the carry ancillas
        qc.append(ModExpTools.create_adder_inv(n), list(a) + list(b) + list(c))
        # reset t by checking b[n] again, since the final result should have no overflow
        qc.cx(b[n], t[0]) 
        # re-add a to b to restore the correct modular result in b
        qc.append(ModExpTools.create_adder(n), list(a) + list(b) + list(c))

        return qc.to_gate(label='adder_mod')

    @staticmethod
    def create_ctrl_mult_mod(n: int, factorN: int, m: int):
        """
        Controlled modular multiplication: |x>|z> → |x>|x*z mod N>.
        Uses repeated controlled additions of (m*2^j mod N) to implement
        multiplication by m via the binary decomposition of z.
        """
        x = QuantumRegister(1, 'x')
        z = QuantumRegister(n, 'z')
        a = QuantumRegister(n, 'a')
        b = QuantumRegister(n + 1, 'b')
        c = QuantumRegister(n, 'c')
        bN = QuantumRegister(n, 'bN')
        t = QuantumRegister(1, 't')
        qc = QuantumCircuit(x, z, a, b, c, bN, t)

        next_mod = m  
        
        for j in range(n):
            # encode (m*2^j mod N) into a, gated on both x and z[j], to implement
            # the j-th term of the binary expansion of m*z
            temp_mod = next_mod
            i = 0
            while temp_mod != 0:
                if temp_mod % 2 != 0:
                    qc.ccx(x[0], z[j], a[i])  # only load a[i] when both control bits are 1
                i += 1
                temp_mod //= 2

            # accumulate a into b mod N to build up the partial product
            qc.append(
                ModExpTools.create_adder_mod(n, factorN),
                list(a) + list(b) + list(c) + list(bN) + list(t)
            )

            # uncompute a to return it to |0>, so it is clean for the next iteration
            temp_mod = next_mod
            i = 0
            while temp_mod != 0:
                if temp_mod % 2 != 0:
                    qc.ccx(x[0], z[j], a[i])
                i += 1
                temp_mod //= 2

            # double m mod N to move to the next power-of-two contribution
            next_mod = (next_mod * 2) % factorN

        # when x=0 the multiplication is skipped, so copy z to b to preserve the identity |z>
        qc.x(x[0])  # invert x so the following gates activate on the x=0 case
        for j in range(n):
            qc.ccx(x[0], z[j], b[j])  # copy z[j] into b[j] only when x was originally 0
        qc.x(x[0])  # restore x to its original value

        return qc.to_gate(label='ctrl_mult_mod')

    @staticmethod
    def create_ctrl_mult_mod_inv(n: int, factorN: int, m: int):
        """
        Inverse of controlled modular multiplication.
        Needed to uncompute the b register after a controlled SWAP,
        so that ancillas are clean for the next exponentiation step.
        """
        qc = QuantumCircuit(5 * n + 3)
        qc.append(
            ModExpTools.create_ctrl_mult_mod(n, factorN, m).inverse(),
            range(5 * n + 3)
        )
        return qc.to_gate(label='ctrl_mult_mod_inv')


    @staticmethod
    def create_mod_exp(n: int, factorN: int, y: int, nx: int):
        """
        Quantum modular exponentiation: computes y^x mod N coherently.

        Each bit x[i] of the exponent controls a multiplication by y^(2^i) mod N,
        implementing the binary method for exponentiation. A controlled SWAP moves
        the result into z after each step, and the inverse multiplication uncomputes b.

        Parameters
        ----------
        n : int
            number of bits for the result and modulus
        factorN : int
            modulus N (the number to factor in Shor's algorithm)
        y : int
            base of exponentiation (must be coprime with N)
        nx : int
            number of bits in the exponent register x

        Gate
        ----
        operates on (nx + 5n + 3) qubits:
            x  : nx qubits  — exponent register, supplied by QPE in superposition
            z  : n  qubits  — result register, initialized to |1>
            a  : n  qubits  — temporary workspace for encoding multiplier bits
            b  : n+1 qubits — accumulator for modular multiplication
            c  : n  qubits  — carry ancillas for the ripple-carry adder
            bN : n  qubits  — stores N so the modular adder can subtract it
            t  : 1  qubit   — temporary flag for underflow detection
        """
        x  = QuantumRegister(nx,     'x')
        z  = QuantumRegister(n,      'z')
        a  = QuantumRegister(n,      'a')
        b  = QuantumRegister(n + 1,  'b')
        c  = QuantumRegister(n,      'c')
        bN = QuantumRegister(n,      'bN')
        t  = QuantumRegister(1,      't')
        
        qc = QuantumCircuit(x, z, a, b, c, bN, t)
        
        m = y  # start with y^(2^0) = y; will be squared each iteration
        
        for i in range(nx):
            # multiply z by m mod N, controlled on x[i], to implement one bit of binary exponentiation
            qc.append(
                ModExpTools.create_ctrl_mult_mod(n, factorN, m),
                [x[i]] + z[:] + a[:] + b[:] + c[:] + bN[:] + list(t),
            )

            # move the product from b into z so z always holds the running result
            for j in range(n):
                qc.cswap(x[i], z[j], b[j])  # controlled swap: only active when x[i]=1

            # uncompute b using m^{-1} mod N so that b returns to |0>, keeping ancillas clean
            m_inv = int(mod_inverse(m, factorN)) 
            qc.append(
                ModExpTools.create_ctrl_mult_mod_inv(n, factorN, m_inv),
                [x[i]] + z[:] + a[:] + b[:] + c[:] + bN[:] + list(t),
            )

            # square m to get the multiplier for the next bit position: y^(2^(i+1))
            m = (m * m) % factorN

        return qc.to_gate(label='mod_exp')

    @staticmethod
    def myqft(n: int):
        """
        Quantum Fourier Transform on n qubits.
        The bit-reversal swap at the end corrects the reversed qubit ordering
        that arises naturally from the rotation pattern.
        """
        qc = QuantumCircuit(n)
        
        for i in reversed(range(n)):
            qc.h(i)  # put qubit i into the Fourier basis before applying phase corrections
            
            # apply controlled phase rotations from lower qubits to build up the full QFT phases
            for e, j in enumerate(reversed(range(i))):
                qc.cp(pi / 2 ** (e + 1), j, i)
        
        # reverse qubit order to match the standard QFT convention
        for i in range(n // 2):
            qc.swap(i, n - i - 1)
        
        return qc.to_gate(label=f'myqft{n}')



def binary_to_decimal(binary_str: str) -> int:
    """Convert binary string to decimal. See ModExpTools.binary_to_decimal."""
    return ModExpTools.binary_to_decimal(binary_str)


def create_sum(name=None):
    """Create SUM gate. See ModExpTools.create_sum."""
    return ModExpTools.create_sum(name)


def create_sum_inv(name=None):
    """Create inverse SUM gate. See ModExpTools.create_sum_inv."""
    return ModExpTools.create_sum_inv(name)


def create_carrier(name=None):
    """Create CARRY gate. See ModExpTools.create_carrier."""
    return ModExpTools.create_carrier(name)


def create_carrier_inv(name=None):
    """Create inverse CARRY gate. See ModExpTools.create_carrier_inv."""
    return ModExpTools.create_carrier_inv(name)


def create_adder(n: int):
    """Create quantum adder. See ModExpTools.create_adder."""
    return ModExpTools.create_adder(n)


def create_adder_inv(n: int):
    """Create inverse quantum adder. See ModExpTools.create_adder_inv."""
    return ModExpTools.create_adder_inv(n)


def create_adder_mod(n: int, factorN: int):
    """Create modular adder. See ModExpTools.create_adder_mod."""
    return ModExpTools.create_adder_mod(n, factorN)


def create_ctrl_mult_mod(n: int, factorN: int, m: int):
    """Create controlled modular multiplication. See ModExpTools.create_ctrl_mult_mod."""
    return ModExpTools.create_ctrl_mult_mod(n, factorN, m)


def create_ctrl_mult_mod_inv(n: int, factorN: int, m: int):
    """Create inverse controlled modular multiplication. See ModExpTools.create_ctrl_mult_mod_inv."""
    return ModExpTools.create_ctrl_mult_mod_inv(n, factorN, m)


def create_mod_exp(n: int, factorN: int, y: int, nx: int):
    """Create modular exponentiation circuit. See ModExpTools.create_mod_exp."""
    return ModExpTools.create_mod_exp(n, factorN, y, nx)


def myqft(n: int):
    """Create QFT circuit. See ModExpTools.myqft."""
    return ModExpTools.myqft(n)