#!/usr/bin/env python3
"""tq_ref.py - numpy-only reader and decoder for the Gyro trellis tensor formats
(TQ2_T, TQK4..TQK7, TQP5..TQP7), the Gyro block-Hadamard rotation, and a minimal GGUF reader.

Copyright 2026 AgentionAI (Laurent Zuijdwijk). Licensed under the Apache License, Version 2.0 (see LICENSE and NOTICE).
Tensor format: trellis-coded quantization with Hadamard rotation, in the family of
QTIP (Tseng et al., 2024). Format, encoder and kernels: AgentionAI.

The decode is bit-exact with the C reference (src/tq_ref.c); see SPEC.md.

Usage:
    python tq_ref.py list   model.gguf
    python tq_ref.py decode model.gguf <tensor name> out.npy [--unrotate]
"""
import base64
import struct

import numpy as np

QK = 128            # weights per block
STEPS = 32          # trellis steps per block (TQ2_T, TQK; TQP has 64)
STATE_MASK = 0x7FFF # 15 state bits reach the codebook

# type id -> (name, K new bits per step)
TYPES = {
    144: ("TQ2_T", 8),
    145: ("TQK6", 6),
    146: ("TQK7", 7),
    147: ("TQP5", 5),
    148: ("TQP6", 6),
    149: ("TQP7", 7),
    150: ("TQK5", 5),
    151: ("TQK4", 4),
}
PAIR_TYPES = {147, 148, 149}  # TQP: 64 steps of 2 weights
TYPE_IDS = {name: tid for tid, (name, _) in TYPES.items()}


def steps(type_id):
    return 64 if type_id in PAIR_TYPES else 32


def block_bytes(type_id):
    return 2 + steps(type_id) * TYPES[type_id][1] // 8


# The normative 2048-point codebook: 4096 little-endian binary16 values,
# point i = (CODEBOOK[2i], CODEBOOK[2i+1]).
_CODEBOOK_B64 = """
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"""
CODEBOOK_F16 = np.frombuffer(base64.b64decode("".join(_CODEBOOK_B64.split())), dtype="<u2").copy()
CODEBOOK = CODEBOOK_F16.view("<f2").astype(np.float32).reshape(2048, 2)


def codebook_index(s, p):
    """Codebook index (0..2047) of pair p of state(s) s."""
    x = ((np.asarray(s, dtype=np.uint64) & STATE_MASK) * 0x9E3779B1) & 0xFFFFFFFF
    return (x >> 10) & 2047 if p else x >> 21


def states(qs, k, nsteps=STEPS):
    """16-bit states, shape (nblocks, nsteps), of qs bytes shaped (nblocks, nsteps*k/8)."""
    qs = np.asarray(qs, dtype=np.uint8)
    if k == 8:  # TQ2_T: state of step t = qs[t-1] << 8 | qs[t], circular
        return (np.roll(qs, 1, axis=1).astype(np.uint32) << 8) | qs
    nbits = nsteps * k
    bits = np.unpackbits(qs, axis=1, bitorder="little").astype(np.uint32)  # stream bit i
    off = (nsteps - 1 - np.arange(nsteps))[:, None] * k                     # window starts
    idx = (off + np.arange(16)[None, :]) % nbits                            # (nsteps, 16)
    return (bits[:, idx] << np.arange(16, dtype=np.uint32)).sum(axis=2, dtype=np.uint32)


def dequantize(type_id, data, n=None):
    """Decode raw block bytes of a TQ type to float32 (length n, default: all blocks)."""
    if isinstance(type_id, str):
        type_id = TYPE_IDS[type_id]
    k = TYPES[type_id][1]
    bs = block_bytes(type_id)
    raw = np.frombuffer(data, dtype=np.uint8)
    nb = raw.size // bs if n is None else n // QK
    raw = raw[: nb * bs].reshape(nb, bs)
    d = raw[:, :2].copy().view("<f2").astype(np.float32)        # (nb, 1)
    if type_id in PAIR_TYPES:                                   # weight 2t+j: first pair of s_t
        s = states(raw[:, 2:], k, 64)                           # (nb, 64)
        vals = CODEBOOK[codebook_index(s, 0)].reshape(nb, QK)
        return (d * vals).astype(np.float32).reshape(-1)
    s = states(raw[:, 2:], k)                                   # (nb, 32)
    c0 = CODEBOOK[codebook_index(s, 0)]                         # (nb, 32, 2)
    c1 = CODEBOOK[codebook_index(s, 1)]
    vals = np.concatenate([c0, c1], axis=2).reshape(nb, QK)     # weight 4t+j
    return (d * vals).astype(np.float32).reshape(-1)


# ------------------------------------------------------ block-Hadamard rotation

def _fwht(v, block):
    """Normalised Walsh-Hadamard transform H_b of each block-wide slice of the last axis
    (same butterfly order and float32 arithmetic as src/tq_ref.c)."""
    v = np.array(v, dtype=np.float32)
    n = v.shape[-1]
    if block < 1 or block & (block - 1) or n % block:
        raise ValueError("block must be a power of two dividing the last dimension")
    x = v.reshape(-1, n // block, block)
    length = 1
    while length < block:
        y = x.reshape(x.shape[0], x.shape[1], block // (2 * length), 2, length)
        a, b = y[:, :, :, 0, :], y[:, :, :, 1, :]
        x = np.stack([a + b, a - b], axis=3).reshape(x.shape)
        length *= 2
    scale = np.float32(1.0) / np.sqrt(np.float32(block))
    return (x * scale).reshape(v.shape)


def hadamard_rotate(v, block, signs=None):
    """T v = H_b (s * v) along the last axis: how rotated weight rows are stored, and what a
    runtime applies to the input of a matmul with a rotated weight."""
    v = np.asarray(v, dtype=np.float32)
    if signs is not None:
        v = v * np.asarray(signs, dtype=np.float32)
    return _fwht(v, block)


def hadamard_unrotate(v, block, signs=None):
    """T^-1 v = s * (H_b v) along the last axis (H_b is symmetric and its own inverse)."""
    out = _fwht(v, block)
    if signs is not None:
        out = out * np.asarray(signs, dtype=np.float32)
    return out


class HadamardInfo:
    """The prism.hadamard.* metadata of a GGUF (SPEC.md section 8)."""

    def __init__(self, meta):
        self.version = int(meta["prism.hadamard.version"])
        if self.version not in (1, 2):
            raise ValueError("unsupported prism.hadamard.version %d" % self.version)
        self.block_size = int(meta["prism.hadamard.block_size"])
        if self.block_size < 1 or self.block_size & (self.block_size - 1):
            raise ValueError("prism.hadamard.block_size must be a power of two")
        if meta.get("prism.hadamard.transform") != "normalized-sylvester-walsh-hadamard":
            raise ValueError("unsupported prism.hadamard.transform")
        if meta.get("prism.hadamard.axis") != "input-last-dimension":
            raise ValueError("unsupported prism.hadamard.axis")
        self.sign_mode = meta.get("prism.hadamard.sign_mode")
        if self.sign_mode not in ("identity", "explicit"):
            raise ValueError("unsupported prism.hadamard.sign_mode")
        self.signs = {}
        if self.sign_mode == "explicit":
            widths = [int(w) for w in meta.get("prism.hadamard.sign_widths", [])]
            values = np.asarray(meta.get("prism.hadamard.sign_values", []), dtype=np.int32)
            if not widths or sum(widths) != values.size or len(set(widths)) != len(widths):
                raise ValueError("bad prism.hadamard.sign_widths / sign_values")
            off = 0
            for w in widths:
                if w % self.block_size:
                    raise ValueError("sign width %d is not a multiple of the block size" % w)
                vec = values[off:off + w]
                if not np.all(np.abs(vec) == 1):
                    raise ValueError("prism.hadamard sign values must be +1/-1")
                self.signs[w] = vec.astype(np.int8)
                off += w
        self.weight_names = set(meta.get("prism.hadamard.weight_names", []))
        self.inverse_weight_names = set(meta.get("prism.hadamard.inverse_weight_names", []))
        self.tied_output = bool(meta.get("prism.hadamard.tied_output", False))
        self.gdn_v_grouped = bool(meta.get("prism.hadamard.gdn_v_grouped", False))

    def signs_for(self, width):
        if self.sign_mode == "identity":
            return None
        if width not in self.signs:
            raise ValueError("prism.hadamard has no sign vector for width %d" % width)
        return self.signs[width]

    def is_rotated(self, name):
        return name in self.weight_names or name in self.inverse_weight_names


# ----------------------------------------------------------------- GGUF reader

_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
# ggml types this reader can return besides the TQ family: id -> (numpy dtype, bytes per element)
_PLAIN = {0: (np.float32, 4), 1: (np.float16, 2), 30: (None, 2)}  # 30 = BF16


class GGUF:
    """Minimal GGUF v2/v3 reader: metadata, tensor table, and tensor decode."""

    def __init__(self, path):
        self.path = path
        self._mm = np.memmap(path, np.uint8, "r")
        self._pos = 0
        if self._take(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        if self._u(4) < 2:
            raise ValueError("GGUF v1 is not supported")
        n_tensors, n_kv = self._u(8), self._u(8)
        self.metadata, self.metadata_types = {}, {}
        for _ in range(n_kv):
            key = self._str()
            vt = self._u(4)
            if vt == 9:
                et, cnt = self._u(4), self._u(8)
                self.metadata[key] = [self._value(et) for _ in range(cnt)]
                self.metadata_types[key] = (9, et)
            else:
                self.metadata[key] = self._value(vt)
                self.metadata_types[key] = (vt, None)
        self.tensors = {}
        for _ in range(n_tensors):
            name = self._str()
            ne = [self._u(8) for _ in range(self._u(4))]
            ttype, offset = self._u(4), self._u(8)
            self.tensors[name] = (ne, ttype, offset)
        align = int(self.metadata.get("general.alignment", 32))
        self.data_offset = (self._pos + align - 1) // align * align
        self.hadamard = HadamardInfo(self.metadata) if "prism.hadamard.version" in self.metadata else None

    # -- low-level parsing
    def _take(self, n):
        if self._pos + n > self._mm.size:
            raise ValueError("truncated GGUF file")
        out = self._mm[self._pos:self._pos + n].tobytes()
        self._pos += n
        return out

    def _u(self, n):
        return int.from_bytes(self._take(n), "little")

    def _str(self):
        return self._take(self._u(8)).decode("utf-8", errors="replace")

    def _value(self, vt):
        if vt == 8:
            return self._str()
        if vt == 9:
            et, cnt = self._u(4), self._u(8)
            return [self._value(et) for _ in range(cnt)]
        fmt = _SCALAR[vt]
        return struct.unpack(fmt, self._take(struct.calcsize(fmt)))[0]

    # -- tensors
    def tensor_type_name(self, name):
        t = self.tensors[name][1]
        return TYPES[t][0] if t in TYPES else {0: "F32", 1: "F16", 30: "BF16"}.get(t, "type%d" % t)

    def read(self, name, unrotate=False):
        """Decode a tensor to float32 with numpy shape (ne[n-1], ..., ne[0]).
        unrotate=True undoes the prism.hadamard rotation of a listed tensor, giving the
        weights in the model's original basis (a runtime instead rotates the matmul input).
        A gdn_v_grouped ssm_out tensor additionally has a feature permutation (SPEC.md 8.4)
        that this reader does not undo."""
        ne, t, off = self.tensors[name]
        n = int(np.prod(ne, dtype=np.int64))
        start = self.data_offset + off
        if t in TYPES:
            nbytes = n // QK * block_bytes(t)
            out = dequantize(t, self._mm[start:start + nbytes].tobytes(), n)
        elif t in _PLAIN:
            dt, size = _PLAIN[t]
            raw = np.array(self._mm[start:start + n * size])
            if t == 30:
                out = (raw.view("<u2").astype(np.uint32) << 16).view(np.float32)
            else:
                out = raw.view(dt).astype(np.float32)
        else:
            raise NotImplementedError("tensor type %d is not handled by this reader" % t)
        out = out.reshape(list(reversed(ne)))
        if unrotate and self.hadamard is not None and self.hadamard.is_rotated(name):
            h = self.hadamard
            if h.gdn_v_grouped and ".ssm_out." in name:
                raise NotImplementedError("unrotating a gdn_v_grouped ssm_out tensor is not implemented")
            out = hadamard_unrotate(out, h.block_size, h.signs_for(ne[0]))
        return out
