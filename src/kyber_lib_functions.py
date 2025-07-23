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
if KYBER_K == 2:
    KYBER_POLYCOMPRESSEDBYTES = 128  # 4 bits / coefficient
    KYBER_POLYVECCOMPRESSEDBYTES = KYBER_K * 320
elif KYBER_K == 3:
    KYBER_POLYCOMPRESSEDBYTES = 128  # 4 bits / coefficient
    KYBER_POLYVECCOMPRESSEDBYTES = KYBER_K * 320
elif KYBER_K == 4:
    KYBER_POLYCOMPRESSEDBYTES = 160  # 5 bits / coefficient
    KYBER_POLYVECCOMPRESSEDBYTES = KYBER_K * 352
else:
    raise ValueError()

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


def csubq(a: int) -> int:
    """
    Constant-time conditional subtraction of KYBER_Q.
    Mirrors the 16-bit two’s-complement trick used in the C reference:
        a  = a - q
        a += (a >> 15) & q        # adds q back iff a was negative
    Python’s right-shift on a negative int is an arithmetic shift,
    so we get the same sign-propagation behaviour.
    """
    a -= KYBER_Q
    a += (a >> 15) & KYBER_Q
    return a


def poly_csubq(p: Poly) -> None:
    """
    Apply csubq() to every coefficient in-place.
    (Constant-time in C; in Python we keep the identical
    arithmetic but of course the interpreter adds overhead.)
    """
    for i in range(KYBER_N):
        p.coeffs[i] = csubq(p.coeffs[i])


def poly_compress(a: Poly) -> bytes:
    """
    Compress `a` into the packed-byte representation defined by
    the Kyber spec (section 5.1).  Returns a `bytes` object of
    length KYBER_POLYCOMPRESSEDBYTES (128 B for k ∈ {2,3}, 160 B for k = 4).

    The code follows the reference algorithm exactly, differing only
    in Python-friendly structure (bytearray instead of raw pointer math).
    """
    poly_csubq(a)  # canonicalise coefficients first
    r = bytearray(KYBER_POLYCOMPRESSEDBYTES)
    out = 0  # current write index in r

    if KYBER_POLYCOMPRESSEDBYTES == 128:  # 4 bits/coeff
        for i in range(KYBER_N // 8):
            t = [
                (((a.coeffs[8 * i + j] << 4) + KYBER_Q // 2) // KYBER_Q) & 0x0F
                for j in range(8)
            ]
            r[out + 0] = t[0] | (t[1] << 4)
            r[out + 1] = t[2] | (t[3] << 4)
            r[out + 2] = t[4] | (t[5] << 4)
            r[out + 3] = t[6] | (t[7] << 4)
            out += 4

    elif KYBER_POLYCOMPRESSEDBYTES == 160:  # 5 bits/coeff
        for i in range(KYBER_N // 8):
            t = [
                (((a.coeffs[8 * i + j] << 5) + KYBER_Q // 2) // KYBER_Q) & 0x1F
                for j in range(8)
            ]
            r[out + 0] = (t[0] >> 0) | (t[1] << 5)
            r[out + 1] = (t[1] >> 3) | (t[2] << 2) | (t[3] << 7)
            r[out + 2] = (t[3] >> 1) | (t[4] << 4)
            r[out + 3] = (t[4] >> 4) | (t[5] << 1) | (t[6] << 6)
            r[out + 4] = (t[6] >> 2) | (t[7] << 3)
            out += 5

    else:
        raise ValueError("KYBER_POLYCOMPRESSEDBYTES must be 128 or 160")

    return bytes(r)


def polyvec_csubq(v: PolyVec) -> None:
    """
    Canonical-reduce every coefficient of every polynomial in-place.
    """
    for i in range(KYBER_K):
        poly_csubq(v.vec[i])


def polyvec_compress(a: PolyVec) -> bytes:
    """
    Compress an entire vector of K polynomials into the packed-byte layout
    defined by the Kyber spec (§5.1).  The output length is
        • K × 352 B when k ∈ {2,3}   (11 B per 8 coefficients, 11-bit bins)
        • K × 320 B when k = 4        (5 B  per 4 coefficients, 10-bit bins)
    """
    polyvec_csubq(a)
    r = bytearray(KYBER_POLYVECCOMPRESSEDBYTES)
    out = 0  # current byte index

    if KYBER_POLYVECCOMPRESSEDBYTES == KYBER_K * 352:
        # ------------ 11-bit packing: 8 coeffs → 11 bytes ------------
        for i in range(KYBER_K):
            for j in range(KYBER_N // 8):
                t = [
                    (((a.vec[i].coeffs[8 * j + k] << 11) + KYBER_Q // 2) // KYBER_Q)
                    & 0x7FF  # 11-bit slice
                    for k in range(8)
                ]

                r[out + 0] = (t[0] >> 0) & 0xFF
                r[out + 1] = ((t[0] >> 8) | (t[1] << 3)) & 0xFF
                r[out + 2] = ((t[1] >> 5) | (t[2] << 6)) & 0xFF
                r[out + 3] = (t[2] >> 2) & 0xFF
                r[out + 4] = ((t[2] >> 10) | (t[3] << 1)) & 0xFF
                r[out + 5] = ((t[3] >> 7) | (t[4] << 4)) & 0xFF
                r[out + 6] = ((t[4] >> 4) | (t[5] << 7)) & 0xFF
                r[out + 7] = (t[5] >> 1) & 0xFF
                r[out + 8] = ((t[5] >> 9) | (t[6] << 2)) & 0xFF
                r[out + 9] = ((t[6] >> 6) | (t[7] << 5)) & 0xFF
                r[out + 10] = (t[7] >> 3) & 0xFF
                out += 11

    elif KYBER_POLYVECCOMPRESSEDBYTES == KYBER_K * 320:
        # ------------ 10-bit packing: 4 coeffs → 5 bytes -------------
        for i in range(KYBER_K):
            for j in range(KYBER_N // 4):
                t = [
                    (((a.vec[i].coeffs[4 * j + k] << 10) + KYBER_Q // 2) // KYBER_Q)
                    & 0x3FF  # 10-bit slice
                    for k in range(4)
                ]

                r[out + 0] = (t[0] >> 0) & 0xFF
                r[out + 1] = ((t[0] >> 8) | (t[1] << 2)) & 0xFF
                r[out + 2] = ((t[1] >> 6) | (t[2] << 4)) & 0xFF
                r[out + 3] = ((t[2] >> 4) | (t[3] << 6)) & 0xFF
                r[out + 4] = (t[3] >> 2) & 0xFF
                out += 5

    else:
        raise ValueError("KYBER_POLYVECCOMPRESSEDBYTES must be K×352 or K×320")

    return bytes(r)


def pack_ciphertext(b: PolyVec, v: Poly) -> bytearray:
    """Combine the compressed (u=vectors) and v (polynomial) part into one ciphertext buffer."""
    r = bytearray(KYBER_INDCPA_BYTES)
    r[:KYBER_POLYVECCOMPRESSEDBYTES] = polyvec_compress(b)
    r[KYBER_POLYVECCOMPRESSEDBYTES:] = poly_compress(v)
    return r
