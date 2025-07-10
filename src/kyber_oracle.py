import socket
import struct

from src.kyber_encodings import (
    k,
)

if k == 3:
    CRYPTO_CIPHERTEXTBYTES = 1088
elif k == 3:
    CRYPTO_CIPHERTEXTBYTES = 768
else:
    raise ValueError()


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

        # ---- initial 16-byte header -----------------------------------
        self.rand_mask = struct.unpack("<Q", recv_exact(self._sock, 8))[0]
        self.target_addr = struct.unpack("<Q", recv_exact(self._sock, 8))[0]

    # ---------------- public API --------------------------------------
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


def build_full_rotate_ciphertext():
    pass


def build_arbitrary_combination_ciphertext():
    pass
