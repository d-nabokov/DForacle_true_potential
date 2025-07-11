### ChatGPT conversion of some Kyber library functions from
### reference implementation

from dataclasses import dataclass
from typing import List

from src.kyber_encodings import (
    k,
)

#############################
# Kyber parameters & sizes ##
#############################

# Choose the parameter‑set you need by changing KYBER_K
#   * Kyber512  -> K = 2
#   * Kyber768  -> K = 3
#   * Kyber1024 -> K = 4
KYBER_K: int = k
KYBER_N: int = 256
KYBER_Q: int = 3329  # prime modulus

# Compression sizes (taken straight from the specification)
if KYBER_K in (2, 3):
    KYBER_POLYCOMPRESSEDBYTES = 128  # 4 bits / coefficient
    KYBER_POLYVECCOMPRESSEDBYTES = KYBER_K * 352
else:  # K == 4
    KYBER_POLYCOMPRESSEDBYTES = 160  # 5 bits / coefficient
    KYBER_POLYVECCOMPRESSEDBYTES = KYBER_K * 320

KYBER_INDCPA_BYTES = KYBER_POLYVECCOMPRESSEDBYTES + KYBER_POLYCOMPRESSEDBYTES

#############################
#       Data classes        #
#############################


@dataclass
class Poly:
    """A Kyber polynomial with N = 256 coefficients mod q."""

    coeffs: List[int]

    def __init__(self):
        # Initialise with zeros
        self.coeffs = [0] * KYBER_N


@dataclass
class PolyVec:
    """A Kyber vector of K polynomials (dimension depends on security level)."""

    vec: List[Poly]

    def __init__(self):
        self.vec = [Poly() for _ in range(KYBER_K)]


#######################################
#     Compression helper functions    #
#######################################


def poly_compress(a: Poly) -> bytearray:
    """Pure-Python port of `poly_compress` from the reference implementation.

    The output size depends on the chosen parameter set (128 or 160 bytes).
    """
    r = bytearray(KYBER_POLYCOMPRESSEDBYTES)
    if KYBER_POLYCOMPRESSEDBYTES == 128:  # 4‑bit packing (Kyber512,768)
        pos = 0
        for i in range(0, KYBER_N, 8):
            t = [0] * 8
            for j in range(8):
                u = a.coeffs[i + j]
                # map to positive standard reps
                u += (u >> 15) & KYBER_Q
                # the strange constants are an optimised way to compute
                #    floor((u * 16 + q/2)/q) mod 16
                d0 = (u << 4) + 1665  # multiply by 16, add rounding constant
                d0 *= 80635
                d0 >>= 28
                t[j] = d0 & 0xF
            # pack 8 4‑bit values into 4 bytes
            r[pos + 0] = t[0] | (t[1] << 4)
            r[pos + 1] = t[2] | (t[3] << 4)
            r[pos + 2] = t[4] | (t[5] << 4)
            r[pos + 3] = t[6] | (t[7] << 4)
            pos += 4
    elif KYBER_POLYCOMPRESSEDBYTES == 160:  # 5‑bit packing (Kyber1024)
        pos = 0
        for i in range(0, KYBER_N, 8):
            t = [0] * 8
            for j in range(8):
                u = a.coeffs[i + j]
                u += (u >> 15) & KYBER_Q
                d0 = (u << 5) + 1664
                d0 *= 40318
                d0 >>= 27
                t[j] = d0 & 0x1F
            # pack 8 5‑bit values into 5 bytes
            r[pos + 0] = (t[0] >> 0) | (t[1] << 5)
            r[pos + 1] = (t[1] >> 3) | (t[2] << 2) | (t[3] << 7)
            r[pos + 2] = (t[3] >> 1) | (t[4] << 4)
            r[pos + 3] = (t[4] >> 4) | (t[5] << 1) | (t[6] << 6)
            r[pos + 4] = (t[6] >> 2) | (t[7] << 3)
            pos += 5
    else:
        raise ValueError("KYBER_POLYCOMPRESSEDBYTES must be 128 or 160")
    return r


def polyvec_compress(a: PolyVec) -> bytearray:
    """Pure‑Python port of `polyvec_compress` (handles both 10‑ and 11‑bit packing)."""
    r = bytearray(KYBER_POLYVECCOMPRESSEDBYTES)
    if KYBER_POLYVECCOMPRESSEDBYTES == KYBER_K * 352:  # 11‑bit packing (Kyber512,768)
        pos = 0
        for i in range(KYBER_K):
            for j in range(0, KYBER_N, 8):
                t = [0] * 8
                for k in range(8):
                    u = a.vec[i].coeffs[j + k]
                    u += (u >> 15) & KYBER_Q
                    d0 = (u << 11) + 1664
                    d0 *= 645084
                    d0 >>= 31
                    t[k] = d0 & 0x7FF
                # pack 8×11‑bit into 11 bytes
                r[pos + 0] = (t[0] >> 0) & 0xFF
                r[pos + 1] = (t[0] >> 8) | (t[1] << 3)
                r[pos + 2] = (t[1] >> 5) | (t[2] << 6)
                r[pos + 3] = (t[2] >> 2) & 0xFF
                r[pos + 4] = (t[2] >> 10) | (t[3] << 1)
                r[pos + 5] = (t[3] >> 7) | (t[4] << 4)
                r[pos + 6] = (t[4] >> 4) | (t[5] << 7)
                r[pos + 7] = (t[5] >> 1) & 0xFF
                r[pos + 8] = (t[5] >> 9) | (t[6] << 2)
                r[pos + 9] = (t[6] >> 6) | (t[7] << 5)
                r[pos + 10] = (t[7] >> 3) & 0xFF
                pos += 11
    elif KYBER_POLYVECCOMPRESSEDBYTES == KYBER_K * 320:  # 10‑bit packing (Kyber1024)
        pos = 0
        for i in range(KYBER_K):
            for j in range(0, KYBER_N, 4):
                t = [0] * 4
                for k in range(4):
                    u = a.vec[i].coeffs[j + k]
                    u += (u >> 15) & KYBER_Q
                    d0 = (u << 10) + 1665
                    d0 *= 1290167
                    d0 >>= 32
                    t[k] = d0 & 0x3FF
                r[pos + 0] = (t[0] >> 0) & 0xFF
                r[pos + 1] = (t[0] >> 8) | (t[1] << 2)
                r[pos + 2] = (t[1] >> 6) | (t[2] << 4)
                r[pos + 3] = (t[2] >> 4) | (t[3] << 6)
                r[pos + 4] = (t[3] >> 2) & 0xFF
                pos += 5
    else:
        raise ValueError("KYBER_POLYVECCOMPRESSEDBYTES does not match parameter set")
    return r


def pack_ciphertext(b: PolyVec, v: Poly) -> bytearray:
    """Combine the compressed (u=vectors) and v (polynomial) part into one ciphertext buffer."""
    r = bytearray(KYBER_INDCPA_BYTES)
    r[:KYBER_POLYVECCOMPRESSEDBYTES] = polyvec_compress(b)
    r[KYBER_POLYVECCOMPRESSEDBYTES:] = poly_compress(v)
    return r
