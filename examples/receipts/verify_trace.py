#!/usr/bin/env python3
"""Verify the activation trace of a llama-receipts receipt without running the model.

Usage:
    python3 examples/receipts/verify_trace.py receipt.json --model model.gguf [--trace receipt.trace.json] [--report out.json]

Checks, cheapest first:
1. the leaves hash to the root the receipt commits to; leaf and graph counts match;
2. the openings are exactly the Fiat-Shamir sample the root and receipt imply;
3. the graph count fits the token sequence, every graph's token and position inputs hash
   to what the claimed prompt and response say, and the output-row selection is as expected;
4. every data edge is consistent: an input's base hash equals the output hash of the leaf
   recorded as its producer, and the KV cache starts as zeros;
5. each graph that produces one logits row binds it to the receipt's per-token logits hash;
6. each opening sits in the tree, its bytes hash to the leaf, its weight inputs match this
   verifier's own per-tensor hashes of the GGUF, and re-executing the op in numpy reproduces
   the output within the op's tolerance.

Matmuls are judged against three references and take the closest: dequantized weight times
float32 activation (Metal decode kernels), both operands rounded to half (Metal prefill
kernels), and ggml's CPU path for Q4_K and Q6_K weights, where the activation is quantized to
Q8_K and the dot product is integer (reproduced exactly here).

Not covered: a cheat confined to one node is caught only if that node is sampled, the receipt's
signature (sign and check it with your own key tooling), and the verifier still needs the model
file for its weights.

Only numpy and the repo's gguf-py are required.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import struct
import sys
import time
import zlib
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(1, str(Path(__file__).resolve().parent.parent.parent / "gguf-py"))
from gguf import GGUFReader, quants  # noqa: E402

CHALLENGE_DOMAIN = b"llama-receipts/trace-challenge/v1/"
PRODUCER_INPUT = -1
PRODUCER_INITIAL = -2
BIG_MATMUL_ROWS = 16384  # weights with more output rows than this are checked on a row sample
ROWS_CHECKED = 2048
HARD_INPUTS = ("tokens", "positions", "out_ids")
SOFT_INPUTS = ("k_cells", "v_cells", "mask")
QK_K = 256


# ----------------------------------------------------------------------------------------------
# hashing, canonical JSON, Merkle tree
# ----------------------------------------------------------------------------------------------

def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def canonical_bytes(obj) -> bytes:
    """Same bytes nlohmann::json::dump() produces: sorted keys, no whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def _leaf_hash(data: bytes) -> bytes:
    return hashlib.sha256(b"\x00" + data).digest()


def _node_hash(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(b"\x01" + left + right).digest()


def merkle_root(leaves: list[bytes]) -> bytes:
    if not leaves:
        return hashlib.sha256(b"").digest()
    level = [_leaf_hash(x) for x in leaves]
    while len(level) > 1:
        if len(level) % 2:
            level = level + [level[-1]]
        level = [_node_hash(level[i], level[i + 1]) for i in range(0, len(level), 2)]
    return level[0]


def verify_path(leaf_data: bytes, path: list, root: bytes) -> bool:
    running = _leaf_hash(leaf_data)
    for side, sibling_hex in path:
        sibling = bytes.fromhex(sibling_hex)
        running = _node_hash(sibling, running) if side == "L" else _node_hash(running, sibling)
    return running == root


def derive_indices(root_hex: str, binding: bytes, n_leaves: int, k: int, seed: bytes = b"") -> list[int]:
    if n_leaves <= 0:
        return []
    k = min(k, n_leaves)
    out: list[int] = []
    seen: set[int] = set()
    counter = 0
    while len(out) < k:
        msg = CHALLENGE_DOMAIN + bytes.fromhex(root_hex) + binding + seed + counter.to_bytes(8, "big")
        idx = int.from_bytes(hashlib.sha256(msg).digest()[:8], "big") % n_leaves
        counter += 1
        if idx not in seen:
            seen.add(idx)
            out.append(idx)
    return sorted(out)


def decode_blob(s: str | None, enc: str) -> bytes | None:
    if s is None:
        return None
    raw = base64.b64decode(s)
    return zlib.decompress(raw) if enc == "zlib+base64" else raw


# ----------------------------------------------------------------------------------------------
# tensors as numpy views, reference ops
# ----------------------------------------------------------------------------------------------

NP_DTYPES = {"f32": np.float32, "f16": np.float16, "i32": np.int32, "i64": np.int64, "f64": np.float64,
             "bf16": np.uint16, "i8": np.int8, "i16": np.int16}
EXACT_OPS = frozenset({"CONT", "SET_ROWS", "DUP", "CPY"})


class OpError(Exception):
    pass


def tensor_from_bytes(blob: bytes, ttype: str, ne: list[int], nb: list[int], offset: int = 0) -> np.ndarray:
    """Read-only strided view: ggml ne=[n0..n3], nb in bytes -> numpy shape (n3, n2, n1, n0)."""
    if ttype not in NP_DTYPES:
        raise OpError(f"cannot view type {ttype} as an array")
    dt = np.dtype(NP_DTYPES[ttype])
    span_end = offset + sum((n - 1) * s for n, s in zip(ne, nb)) + dt.itemsize
    if span_end > len(blob):
        raise OpError(f"view needs {span_end} bytes but base has {len(blob)}")
    if offset % dt.itemsize or any(s % dt.itemsize for s in nb):
        raise OpError("view is not element aligned")
    n_items = (span_end - offset) // dt.itemsize
    typed = np.frombuffer(blob, dtype=np.uint8)[offset: offset + n_items * dt.itemsize].view(dt)
    return np.lib.stride_tricks.as_strided(typed, shape=tuple(reversed(ne)), strides=tuple(reversed(nb)), writeable=False)


def f32_params(params_hex: str, n: int, skip_bytes: int = 0) -> list[float]:
    raw = bytes.fromhex(params_hex).ljust(skip_bytes + 4 * n, b"\0")
    return list(struct.unpack("<" + "f" * n, raw[skip_bytes: skip_bytes + 4 * n]))


def i32_params(params_hex: str, n: int) -> list[int]:
    raw = bytes.fromhex(params_hex).ljust(4 * n, b"\0")
    return list(struct.unpack("<" + "i" * n, raw[: 4 * n]))


def to_f64(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint16:  # bf16 payload
        return (arr.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    return arr.astype(np.float64)


def normalized_error(out: np.ndarray, ref: np.ndarray, floor: float = 1e-6) -> float:
    out64, ref64 = to_f64(out), to_f64(ref)
    if out64.shape != ref64.shape:
        return math.inf
    if ref64.size == 0:
        return 0.0
    if not np.all(np.isfinite(out64)) or not np.all(np.isfinite(ref64)):
        return math.inf
    return float(np.max(np.abs(out64 - ref64))) / max(float(np.max(np.abs(ref64))), floor)


def repeat_to(b: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    reps = []
    for tgt, cur in zip(shape, b.shape):
        if tgt % cur:
            raise OpError(f"cannot repeat {b.shape} to {shape}")
        reps.append(tgt // cur)
    return np.tile(b, reps) if any(r != 1 for r in reps) else b


def silu(x):
    return x / (1.0 + np.exp(-x))


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


def op_mul_mat(a, b):
    """ggml_mul_mat: a (a3,a2,M,K), b (b3,b2,N,K) -> (b3,b2,N,M); a broadcasts over dims 2,3."""
    a3, a2, M, K = a.shape
    b3, b2, N, Kb = b.shape
    if K != Kb or b3 % a3 or b2 % a2:
        raise OpError(f"mul_mat shapes {a.shape} x {b.shape}")
    out = np.empty((b3, b2, N, M), dtype=np.float64)
    for i3 in range(b3):
        for i2 in range(b2):
            out[i3, i2] = b[i3, i2] @ a[i3 // (b3 // a3), i2 // (b2 // a2)].T
    return out


def op_get_rows(a, ids):
    a3, a2, R, C = a.shape
    ids2 = ids.reshape(ids.shape[-2], ids.shape[-1]) if ids.ndim >= 2 else ids.reshape(1, -1)
    out = np.empty((a3, a2, ids2.shape[-1], C), dtype=a.dtype)
    for i3 in range(a3):
        for i2 in range(a2):
            idx = ids2[i2 if ids2.shape[0] > 1 else 0]
            if np.any(idx < 0) or np.any(idx >= R):
                raise OpError("get_rows index out of range")
            out[i3, i2] = a[i3, i2, idx]
    return out


def op_rope(x, pos, params_hex):
    p = i32_params(params_hex, 5)
    freq_base, freq_scale, ext_factor, attn_factor, _bf, _bs = f32_params(params_hex, 6, skip_bytes=20)
    n_dims, mode = p[1], p[2]
    if ext_factor != 0.0:
        raise OpError("rope with YaRN ext_factor is not implemented")
    if mode & 8:
        raise OpError("multi-section rope is not implemented")
    _, n, H, D = x.shape
    pos = pos.reshape(-1).astype(np.float64)
    if pos.shape[0] != n:
        raise OpError(f"rope positions {pos.shape[0]} vs tokens {n}")
    half = n_dims // 2
    inv = freq_base ** (-np.arange(half, dtype=np.float64) * 2.0 / n_dims)
    theta = pos[:, None] * freq_scale * inv[None, :]
    cos = (np.cos(theta) * attn_factor)[:, None, :]
    sin = (np.sin(theta) * attn_factor)[:, None, :]
    out = x.copy()
    if mode & 2:  # NEOX
        x0, x1 = x[0, :, :, :half], x[0, :, :, half:n_dims]
        out[0, :, :, :half] = x0 * cos - x1 * sin
        out[0, :, :, half:n_dims] = x0 * sin + x1 * cos
    else:
        x0, x1 = x[0, :, :, 0:n_dims:2], x[0, :, :, 1:n_dims:2]
        out[0, :, :, 0:n_dims:2] = x0 * cos - x1 * sin
        out[0, :, :, 1:n_dims:2] = x0 * sin + x1 * cos
    return out


def op_soft_max(x, mask, params_hex):
    scale, max_bias = f32_params(params_hex, 2)
    if max_bias != 0.0:
        raise OpError("soft_max with ALiBi max_bias is not implemented")
    z = x * scale
    if mask is not None:
        z = z + repeat_to(mask[:, :, : x.shape[2], :], z.shape)
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


def op_glu(a, b, params_hex):
    glu_op, swapped = i32_params(params_hex, 2)
    if b is None:
        half = a.shape[-1] // 2
        a, b = (a[..., half:], a[..., :half]) if swapped else (a[..., :half], a[..., half:])
    elif swapped:
        a, b = b, a
    if glu_op == 0:
        return np.maximum(a, 0.0) * b
    if glu_op == 1:
        return gelu(a) * b
    if glu_op == 2:
        return silu(a) * b
    raise OpError(f"glu op {glu_op} not implemented")


def op_unary(a, name):
    fns = {"SILU": silu, "GELU": gelu, "RELU": lambda x: np.maximum(x, 0.0), "SIGMOID": lambda x: 1.0 / (1.0 + np.exp(-x)), "EXP": np.exp}
    if name not in fns:
        raise OpError(f"unary {name} not implemented")
    return fns[name](a)


def op_flash_attn_ext(q, k, v, mask, params_hex):
    """ggml_flash_attn_ext: q (1,H,n,D), k (1,Hk,n_kv,D), v (1,Hv,n_kv,Dv), mask (1,1,>=n,n_kv) -> (1,n,H,Dv)."""
    scale, max_bias, logit_softcap = f32_params(params_hex, 3)
    if max_bias != 0.0:
        raise OpError("flash_attn_ext with ALiBi max_bias is not implemented")
    _, H, n, D = q.shape
    _, Hk, n_kv, Dk = k.shape
    _, Hv, n_kv_v, Dv = v.shape
    if D != Dk or n_kv != n_kv_v or H % Hk or Hk != Hv:
        raise OpError(f"flash_attn_ext shapes q{q.shape} k{k.shape} v{v.shape}")
    out = np.empty((1, n, H, Dv), dtype=np.float64)
    for h in range(H):
        hk = h // (H // Hk)
        s = (q[0, h] @ k[0, hk].T) * scale
        if logit_softcap != 0.0:
            s = logit_softcap * np.tanh(s / logit_softcap)
        if mask is not None:
            s = s + mask[0, min(h, mask.shape[1] - 1), :n, :]
        s = s - np.max(s, axis=-1, keepdims=True)
        p = np.exp(s)
        p /= np.sum(p, axis=-1, keepdims=True)
        out[0, :, h, :] = p @ v[0, hk]
    return out


def op_set_rows(dst_before, src, idx):
    out = np.array(dst_before, copy=True)
    d3, d2, D1, D0 = out.shape
    s3, s2, R, C = src.shape
    if C != D0:
        raise OpError("set_rows row length mismatch")
    idx3 = idx.reshape(-1, idx.shape[-1]) if idx.ndim >= 2 else idx.reshape(1, -1)
    if idx3.shape[-1] != R:
        raise OpError("set_rows index count differs from source rows")
    for i3 in range(d3):
        for i2 in range(d2):
            rows = idx3[(i3 * d2 + i2) % idx3.shape[0]]
            if np.any(rows < 0) or np.any(rows >= D1):
                raise OpError("set_rows index out of range")
            out[i3, i2, rows, :] = src[i3 % s3, i2 % s2].astype(out.dtype)
    return out


# --- ggml's CPU quantized dot products, integer exact ---------------------------------------

def quantize_q8_k(x: np.ndarray):
    """quantize_row_q8_K_ref: (N, K) float32 -> d (N, nb), q (N, nb, 256) int8, bsums (N, nb, 16)."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    n, k = x.shape
    if k % QK_K:
        raise OpError(f"row length {k} is not a multiple of {QK_K}")
    nb = k // QK_K
    xb = x.reshape(n, nb, QK_K)
    idx = np.argmax(np.abs(xb), axis=-1)
    mx = np.take_along_axis(xb, idx[..., None], axis=-1)[..., 0]
    nz = mx != 0
    safe = np.where(nz, mx, np.float32(1.0))
    iscale = np.where(nz, np.float32(-127.0) / safe, np.float32(0.0)).astype(np.float32)
    q = np.minimum(np.rint(iscale[..., None] * xb), 127).astype(np.int8)
    d = np.where(nz, np.float32(1.0) / np.where(nz, iscale, np.float32(1.0)), np.float32(0.0)).astype(np.float32)
    return d, q, q.reshape(n, nb, 16, 16).astype(np.int32).sum(-1)


def _scale_min_k4(scales):
    s = scales.reshape(-1, 3, 4)
    d, m, m_d = s[:, 0], s[:, 1], s[:, 2]
    sc = np.concatenate([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], axis=-1)
    mn = np.concatenate([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], axis=-1)
    return sc, mn


def unpack_q4_k(blocks):
    r, nb = blocks.shape[0], blocks.shape[1] // 144
    b = np.ascontiguousarray(blocks).reshape(r * nb, 144)
    d = b[:, 0:2].copy().view(np.float16)[:, 0].astype(np.float32).reshape(r, nb)
    dmin = b[:, 2:4].copy().view(np.float16)[:, 0].astype(np.float32).reshape(r, nb)
    sc, mn = _scale_min_k4(b[:, 4:16])
    qs = b[:, 16:144].reshape(r * nb, 4, 32)
    q = np.stack([qs & 0x0F, qs >> 4], axis=2).reshape(r, nb, QK_K)
    return d, dmin, sc.reshape(r, nb, 8), mn.reshape(r, nb, 8), q


def unpack_q6_k(blocks):
    r, nb = blocks.shape[0], blocks.shape[1] // 210
    b = np.ascontiguousarray(blocks).reshape(r * nb, 210)
    ql = b[:, 0:128].reshape(-1, 2, 2, 32)
    qh = b[:, 128:192].reshape(-1, 2, 32)
    scales = b[:, 192:208].copy().view(np.int8).reshape(r, nb, 16)
    d = b[:, 208:210].copy().view(np.float16)[:, 0].astype(np.float32).reshape(r, nb)
    lo, hi = ql & 0x0F, ql >> 4
    q = np.empty((r * nb, 2, 4, 32), dtype=np.uint8)
    q[:, :, 0] = lo[:, :, 0] | (((qh >> 0) & 3) << 4)
    q[:, :, 1] = lo[:, :, 1] | (((qh >> 2) & 3) << 4)
    q[:, :, 2] = hi[:, :, 0] | (((qh >> 4) & 3) << 4)
    q[:, :, 3] = hi[:, :, 1] | (((qh >> 6) & 3) << 4)
    return d, scales, q.reshape(r, nb, QK_K).astype(np.int16) - 32


Q8K_SUPPORTED = ("Q4_K", "Q6_K")


def mul_mat_q8k(blocks: np.ndarray, qtype: str, acts: np.ndarray) -> np.ndarray:
    """ggml_vec_dot_{q4,q6}_K_q8_K over rows: weight rows `blocks` (M, bytes), acts (N, K) f32 -> (N, M) f64."""
    d_a, q8, bsums = quantize_q8_k(acts)
    n, nb = d_a.shape
    if qtype == "Q4_K":
        d_w, dmin, sc, mn, q4 = unpack_q4_k(blocks)
        m = d_w.shape[0]
        w_int = (sc.astype(np.int32)[..., None] * q4.reshape(m, nb, 8, 32).astype(np.int32)).reshape(m, nb, QK_K)
        mins16 = np.repeat(mn.astype(np.int32), 2, axis=-1)
        out = np.zeros((n, m), dtype=np.float64)
        for i in range(nb):
            s1 = q8[:, i].astype(np.float64) @ w_int[:, i].astype(np.float64).T
            s2 = bsums[:, i].astype(np.float64) @ mins16[:, i].astype(np.float64).T
            da = d_a[:, i, None].astype(np.float64)
            out += da * d_w[None, :, i].astype(np.float64) * s1 - da * dmin[None, :, i].astype(np.float64) * s2
        return out
    if qtype == "Q6_K":
        d_w, scales, q6 = unpack_q6_k(blocks)
        m = d_w.shape[0]
        w_int = (scales.astype(np.int32)[..., None] * q6.reshape(m, nb, 16, 16).astype(np.int32)).reshape(m, nb, QK_K)
        out = np.zeros((n, m), dtype=np.float64)
        for i in range(nb):
            s1 = q8[:, i].astype(np.float64) @ w_int[:, i].astype(np.float64).T
            out += d_a[:, i, None].astype(np.float64) * d_w[None, :, i].astype(np.float64) * s1
        return out
    raise OpError(f"no integer reference for {qtype}")


# --- tolerances and dispatch ---------------------------------------------------------------

# Calibrated Sep 11, 2026 on qwen2.5 1.5B Q4_K_M (Apple M5, Metal and CPU, several hundred openings each).
# Elementwise and normalization ops re-execute to about 1e-7; ROPE reached 1.8e-6 and SOFT_MAX 2.4e-5.
# MUL_MAT takes the best of three references: Metal decode 4e-7 (f32), CPU 8e-7 (Q8_K integer path),
# Metal prefill up to 2.9e-3 (half-precision simdgroup inputs the references cannot reproduce).
# FLASH_ATTN_EXT keeps its softmax and KV product in half precision: 1.9e-3 on Metal, 7.5e-3 on CPU.
TOLERANCES = {
    "ADD": 1e-5, "MUL": 1e-5, "SCALE": 1e-5, "GET_ROWS": 1e-5,
    "RMS_NORM": 1e-5, "NORM": 1e-5, "ROPE": 5e-5, "SOFT_MAX": 2e-4, "SWIGLU": 1e-5, "GLU": 1e-5,
    "SILU": 1e-5, "GELU": 1e-5,
    "MUL_MAT": 8e-3,
    "FLASH_ATTN_EXT": 2e-2,
}
DEFAULT_TOLERANCE = 1e-4


def bitwise_equal(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return bool(np.array_equal(np.ascontiguousarray(a).view(np.uint8), np.ascontiguousarray(b).view(np.uint8)))


def execute(leaf: dict, out: np.ndarray, srcs: list, rows=None, qweight=None) -> dict:
    """Recompute `leaf` from `srcs` and compare with the prover's `out`. Returns a report dict."""
    op = leaf["op"]
    params = leaf["params"]
    tol = TOLERANCES.get(op, DEFAULT_TOLERANCE)
    try:
        if op in EXACT_OPS:
            if op == "SET_ROWS":
                ref = op_set_rows(srcs[2], srcs[0], srcs[1])
            else:
                ref = np.ascontiguousarray(srcs[0]).astype(out.dtype)
                if ref.size == out.size:
                    ref = ref.reshape(out.shape)
            if ref.shape != out.shape:
                return {"op": op, "exact": True, "ok": False, "error": None, "detail": "shape mismatch"}
            same = bitwise_equal(ref.astype(out.dtype), out)
            return {"op": op, "exact": True, "ok": same, "error": 0.0 if same else normalized_error(out, ref),
                    "detail": "" if same else "exact op differs"}
        if op == "GET_ROWS":
            ref = op_get_rows(to_f64(srcs[0]), srcs[1].astype(np.int64))
        else:
            a = to_f64(srcs[0])
            if op == "ADD":
                ref = a + repeat_to(to_f64(srcs[1]), a.shape)
            elif op == "MUL":
                ref = a * repeat_to(to_f64(srcs[1]), a.shape)
            elif op == "SCALE":
                s, bias = f32_params(params, 2)
                ref = a * s + bias
            elif op == "RMS_NORM":
                (eps,) = f32_params(params, 1)
                ref = a / np.sqrt(np.mean(a * a, axis=-1, keepdims=True) + eps)
            elif op == "NORM":
                (eps,) = f32_params(params, 1)
                mu = np.mean(a, axis=-1, keepdims=True)
                ref = (a - mu) / np.sqrt(np.mean((a - mu) ** 2, axis=-1, keepdims=True) + eps)
            elif op == "MUL_MAT":
                b_raw = srcs[1]
                b = to_f64(b_raw)
                refs = {"f32": op_mul_mat(a, b)}
                if b_raw.dtype == np.float32:
                    refs["f16"] = op_mul_mat(a.astype(np.float16).astype(np.float64), b.astype(np.float16).astype(np.float64))
                    if qweight is not None and qweight[1] in Q8K_SUPPORTED:
                        b3, b2, n, k = b_raw.shape
                        refs["q8k"] = mul_mat_q8k(qweight[0], qweight[1], np.ascontiguousarray(b_raw).reshape(-1, k)).reshape(b3, b2, n, -1)
                cmp = out if rows is None else out[..., rows]
                errs = {name: normalized_error(cmp, ref) for name, ref in refs.items()}
                best = min(errs, key=errs.get)
                frac = 1.0 if rows is None else len(rows) / leaf["out"]["ne"][0]
                return {"op": op, "exact": False, "ok": errs[best] <= tol, "error": None if math.isinf(errs[best]) else errs[best],
                        "detail": f"matched {best}; " + ", ".join(f"{k}={v:.1e}" for k, v in errs.items()), "checked_fraction": frac}
            elif op == "ROPE":
                ref = op_rope(a, srcs[1], params)
            elif op == "SOFT_MAX":
                ref = op_soft_max(a, to_f64(srcs[1]) if len(srcs) > 1 and srcs[1] is not None else None, params)
            elif op == "FLASH_ATTN_EXT":
                mask = to_f64(srcs[3]) if len(srcs) > 3 and srcs[3] is not None else None
                ref = op_flash_attn_ext(a, to_f64(srcs[1]), to_f64(srcs[2]), mask, params)
            elif leaf["op_name"] == "GLU":
                ref = op_glu(a, to_f64(srcs[1]) if len(srcs) > 1 and srcs[1] is not None else None, params)
            elif leaf["op_name"] == "UNARY":
                ref = op_unary(a, op)
            else:
                return {"op": op, "exact": False, "ok": False, "error": None, "detail": "unsupported"}
        err = normalized_error(out, ref)
        return {"op": op, "exact": False, "ok": err <= tol, "error": None if math.isinf(err) else err, "detail": ""}
    except OpError as e:
        return {"op": op, "exact": False, "ok": False, "error": None, "detail": f"unsupported: {e}"}
    except (IndexError, ValueError, TypeError) as e:
        return {"op": op, "exact": False, "ok": False, "error": None, "detail": f"error: {e}"}


# ----------------------------------------------------------------------------------------------
# weights: per-tensor hashes and dequantized rows from the verifier's own GGUF
# ----------------------------------------------------------------------------------------------

FLOAT_TYPES = ("F32", "F16", "BF16", "F64")


class WeightStore:
    def __init__(self, model_path: str):
        self.path = model_path
        self.reader = GGUFReader(model_path)
        self.tensors = {t.name: t for t in self.reader.tensors}
        self._full: dict[str, np.ndarray] = {}
        self.hashes = self._load_hashes()

    def _load_hashes(self) -> dict[str, str]:
        st = os.stat(self.path)
        cache_dir = Path.home() / ".cache" / "llama-receipts"
        key = sha256(f"{os.path.realpath(self.path)}|{st.st_size}|{st.st_mtime_ns}".encode())[:24]
        cache = cache_dir / f"{key}.json"
        if cache.exists():
            return json.loads(cache.read_text())
        hashes = {}
        for name, t in self.tensors.items():
            hashes[name] = sha256(np.ascontiguousarray(t.data).view(np.uint8).ravel().tobytes())
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(hashes))
        return hashes

    def ne(self, name: str) -> list[int]:
        return [int(x) for x in self.tensors[name].shape]

    def type_name(self, name: str) -> str:
        return self.tensors[name].tensor_type.name

    def rows(self, name: str, rows=None) -> np.ndarray:
        t = self.tensors[name]
        ne0 = int(t.shape[0])
        if t.tensor_type.name in FLOAT_TYPES:
            arr = np.asarray(t.data)
            if t.tensor_type.name == "BF16" and arr.dtype != np.float32:
                arr = (arr.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
            full = arr.reshape(-1, ne0).astype(np.float32, copy=False)
            return full if rows is None else full[rows]
        blocks = np.asarray(t.data).reshape(-1, np.asarray(t.data).shape[-1])
        if rows is not None:
            return quants.dequantize(blocks[rows], t.tensor_type).reshape(-1, ne0).astype(np.float32)
        if name not in self._full:
            if len(self._full) >= 6:
                self._full.pop(next(iter(self._full)))
            self._full[name] = quants.dequantize(blocks, t.tensor_type).reshape(-1, ne0).astype(np.float32)
        return self._full[name]

    def blocks(self, name: str, rows=None):
        t = self.tensors[name]
        if t.tensor_type.name in FLOAT_TYPES:
            return None
        data = np.asarray(t.data).reshape(-1, np.asarray(t.data).shape[-1])
        return data if rows is None else data[rows]

    def array4d(self, name: str, rows=None) -> np.ndarray:
        ne = self.ne(name) + [1] * (4 - len(self.ne(name)))
        data = self.rows(name, rows)
        if rows is not None:
            return data.reshape(1, 1, len(rows), ne[0])
        return data.reshape(tuple(reversed(ne)))


# ----------------------------------------------------------------------------------------------
# expected graph inputs, regenerated from the receipt
# ----------------------------------------------------------------------------------------------

def graph_tokens(rec: dict, g: int):
    prompt, resp = rec["request"]["prompt_tokens"], rec["response"]["tokens"]
    if g == 0:
        return list(prompt)
    if 1 <= g <= len(resp):
        return [resp[g - 1]]
    return None


def graph_start(rec: dict, g: int) -> int:
    return 0 if g == 0 else len(rec["request"]["prompt_tokens"]) + (g - 1)


def expected_inputs(kind: str, rec: dict, g: int, leaf: dict, src: dict) -> list[bytes]:
    """All byte strings a given input may legitimately hash to."""
    toks = graph_tokens(rec, g)
    if toks is None:
        return []
    n, start = len(toks), graph_start(rec, g)
    if kind == "tokens":
        return [np.asarray(toks, dtype=np.int32).tobytes()]
    if kind == "positions":
        return [np.arange(start, start + n, dtype=np.int32).tobytes()]
    if kind == "out_ids":  # logits for every token, or only for the last one
        return [np.arange(n, dtype=np.int32).tobytes(), np.asarray([n - 1], dtype=np.int32).tobytes()]
    if kind == "k_cells":
        return [np.arange(start, start + n, dtype=np.int64).tobytes()]
    if kind == "v_cells":
        # with flash attention the V cache is stored like K (one row per cell); without it the cache
        # is transposed and element (token t, dim e) lands at e * kv_size + cell(t)
        dst = leaf["srcs"][2]["base"]
        kv_size, n_embd_v = dst["ne"][1], dst["ne"][0]
        e = np.arange(n_embd_v, dtype=np.int64)
        cells = np.arange(start, start + n, dtype=np.int64)
        return [cells.tobytes(), (cells[:, None] + e[None, :] * kv_size).reshape(-1).tobytes()]
    if kind == "mask":
        n_kv, rows = src["ne"][0], src["ne"][1]
        m = np.full((rows, n_kv), -np.inf, dtype=np.float32)
        for t in range(min(n, rows)):
            m[t, : start + t + 1] = 0.0
        if src["type"] == "f16":
            m = m.astype(np.float16)
        return [m.tobytes()]
    return []


def classify_input(leaf: dict, i: int) -> str | None:
    op = leaf["op"]
    if op == "GET_ROWS" and i == 1:
        return "tokens" if leaf["srcs"][0].get("kind") == "weight" else "out_ids"
    if op == "ROPE" and i == 1:
        return "positions"
    if op == "SET_ROWS" and i == 1:
        return "v_cells" if leaf["srcs"][2]["base"]["name"].startswith("cache_v") else "k_cells"
    if op == "SOFT_MAX" and i == 1:
        return "mask"
    if op == "FLASH_ATTN_EXT" and i == 3:
        return "mask"
    return None


# ----------------------------------------------------------------------------------------------
# the verifier
# ----------------------------------------------------------------------------------------------

def verify_trace(rec: dict, tdoc: dict, model_path: str, challenge_seed: bytes = b"") -> dict:
    t0 = time.time()
    checks: dict[str, dict] = {}
    hard_fail: list[str] = []

    def check(name, ok, detail="", hard=True, **extra):
        checks[name] = {"ok": bool(ok), "detail": detail, **extra}
        if hard and not ok:
            hard_fail.append(f"{name}: {detail}" if detail else name)

    tr = rec.get("trace")
    if not tr:
        return {"verdict": "reject", "reason": "receipt carries no trace commitment", "checks": checks}
    leaves = tdoc["leaves"]
    root = merkle_root([canonical_bytes(x) for x in leaves]).hex()
    n_graphs = (max((x["g"] for x in leaves), default=-1) + 1) if leaves else 0
    check("root", root == tr["root"] == tdoc["root"] and len(leaves) == tr["n_leaves"] and n_graphs == tr["n_graphs"],
          f"recomputed {root[:16]} vs committed {tr['root'][:16]}; {len(leaves)} leaves, {n_graphs} graphs")

    binding = bytes.fromhex(rec["commitments"]["tokens_sha256"]) + bytes.fromhex(rec["model"]["file_sha256"])
    ch = tdoc.get("challenge", {})
    k = int(ch.get("k", len(tdoc["openings"])))
    seed = bytes.fromhex(ch.get("seed", "") or "")
    expected_idx = derive_indices(tr["root"], binding, len(leaves), k, seed)
    got_idx = sorted(o["index"] for o in tdoc["openings"])
    if challenge_seed and seed != challenge_seed:
        check("challenge", False, "the recorded challenge seed is not the one this verifier issued")
    else:
        check("challenge", got_idx == expected_idx and len(got_idx) == k,
              f"{len(got_idx)} openings, expected {len(expected_idx)}" + ("" if got_idx == expected_idx else ", indices differ")
              + (", interactive seed" if seed else ", Fiat-Shamir"))

    resp = rec["response"]["tokens"]
    check("graph_count", len(resp) <= n_graphs <= len(resp) + 1, f"{n_graphs} graphs for {len(resp)} response tokens")

    zero_hashes: dict[int, str] = {}
    edge_bad = edge_total = unknown_inputs = 0
    inputs = {kind: Counter() for kind in HARD_INPUTS + SOFT_INPUTS}
    last_matmul_by_graph: dict[int, dict] = {}
    for idx, leaf in enumerate(leaves):
        if leaf["op"] == "MUL_MAT":
            last_matmul_by_graph[leaf["g"]] = leaf
        for i, s in enumerate(leaf["srcs"]):
            if s.get("kind") != "data":
                continue
            p = s["producer"]
            if p >= 0:
                edge_total += 1
                if p >= idx or leaves[p]["out"]["base"]["sha256"] != s["base"]["sha256"] or leaves[p]["out"]["base"]["nbytes"] != s["base"]["nbytes"]:
                    edge_bad += 1
            elif p == PRODUCER_INITIAL:
                edge_total += 1
                nb = s["base"]["nbytes"]
                zero_hashes.setdefault(nb, sha256(bytes(nb)))
                if zero_hashes[nb] != s["base"]["sha256"]:
                    edge_bad += 1
            else:
                kind = classify_input(leaf, i)
                if kind is None:
                    unknown_inputs += 1
                    continue
                candidates = expected_inputs(kind, rec, leaf["g"], leaf, s)
                if not candidates:
                    inputs[kind]["no_expectation"] += 1
                elif any(sha256(c) == s["base"]["sha256"] for c in candidates):
                    inputs[kind]["match"] += 1
                else:
                    inputs[kind]["mismatch"] += 1
    check("edges", edge_bad == 0, f"{edge_bad} of {edge_total} data edges inconsistent")
    for kind in HARD_INPUTS:
        c = inputs[kind]
        check(f"input_{kind}", c["mismatch"] == 0 and (c["match"] > 0 or kind == "out_ids"), f"{c['match']} match, {c['mismatch']} mismatch")
    for kind in SOFT_INPUTS:
        c = inputs[kind]
        check(f"input_{kind}", c["mismatch"] == 0, f"{c['match']} match, {c['mismatch']} mismatch", hard=False)
    checks["inputs_unclassified"] = {"ok": True, "detail": f"{unknown_inputs} inputs not classified"}

    per_token = rec["response"]["per_token"]
    bound = bad = 0
    for g in range(0, min(n_graphs, len(per_token))):
        leaf = last_matmul_by_graph.get(g)
        if leaf is None or leaf["out"]["ne"][1] != 1:
            if g > 0:
                bad += 1  # decode graphs always produce exactly one logits row
            continue
        bound += 1
        if leaf["out"]["base"]["sha256"] != per_token[g]["logits_sha256"]:
            bad += 1
    check("logits_binding", bad == 0, f"{bound} graphs bound to receipt logits hashes, {bad} failed")

    t_open = time.time()
    store = WeightStore(model_path)
    check("model_file", sha256_file(model_path) == rec["model"]["file_sha256"], "whole-file SHA-256")
    opening_reports = []
    ops_seen: Counter = Counter()
    worst: dict[str, float] = {}
    n_bad_open = 0
    for o in tdoc["openings"]:
        idx = o["index"]
        rep: dict = {"index": idx}
        if not 0 <= idx < len(leaves):
            rep["error"] = "index out of range"
            n_bad_open += 1
            opening_reports.append(rep)
            continue
        leaf = leaves[idx]
        enc = o.get("enc", "zlib+base64")
        rep.update({"g": leaf["g"], "name": leaf["name"], "op": leaf["op"]})
        ops_seen[leaf["op"]] += 1
        rep["path_ok"] = verify_path(canonical_bytes(leaf), o["path"], bytes.fromhex(tr["root"]))
        out_blob = decode_blob(o["out"], enc)
        src_blobs = [decode_blob(x, enc) for x in o["srcs"]]
        rep["out_hash_ok"] = sha256(out_blob) == leaf["out"]["base"]["sha256"] and len(out_blob) == leaf["out"]["base"]["nbytes"]
        arrays: list = []
        rows = None
        qweight = None
        weights_ok = srcs_ok = True
        problem = ""
        try:
            out_arr = tensor_from_bytes(out_blob, leaf["out"]["type"], leaf["out"]["ne"], leaf["out"]["nb"], leaf["out"]["offset"])
            for i, s in enumerate(leaf["srcs"]):
                if s["kind"] == "weight":
                    name = s["base"]["name"]
                    ne = s["base"]["ne"]
                    known = name in store.tensors
                    shape_ok = known and ne[: len(store.ne(name))] == store.ne(name) and all(x == 1 for x in ne[len(store.ne(name)):])
                    if not known or store.hashes[name] != s["sha256"] or store.type_name(name).lower() != s["base"]["type"].lower() or not shape_ok:
                        weights_ok = False
                        problem = f"weight {name} does not match this model"
                        arrays.append(None)
                        continue
                    if leaf["op"] == "MUL_MAT" and i == 0:
                        if ne[1] > BIG_MATMUL_ROWS:
                            rng = np.random.default_rng(int.from_bytes(hashlib.sha256(bytes.fromhex(tr["root"]) + idx.to_bytes(8, "big")).digest()[:8], "big"))
                            rows = np.sort(rng.choice(ne[1], size=min(ROWS_CHECKED, ne[1]), replace=False))
                        arrays.append(store.array4d(name, rows))
                        raw = store.blocks(name, rows)
                        if raw is not None:
                            qweight = (raw, store.type_name(name))
                    elif leaf["op"] == "GET_ROWS" and i == 0:
                        ids = np.frombuffer(src_blobs[1], dtype=np.int32 if leaf["srcs"][1]["type"] == "i32" else np.int64)
                        arrays.append(store.array4d(name, ids.astype(np.int64)))
                        rep["get_rows_remapped"] = True
                    else:
                        arrays.append(store.array4d(name))
                else:
                    blob = src_blobs[i]
                    if blob is None or sha256(blob) != s["base"]["sha256"] or len(blob) != s["base"]["nbytes"]:
                        srcs_ok = False
                        problem = f"input {i} bytes do not hash to the leaf"
                        arrays.append(None)
                        continue
                    if leaf["op"] == "GET_ROWS" and i == 1 and rep.get("get_rows_remapped"):
                        n_ids = int(np.prod(s["ne"]))
                        arrays.append(np.arange(n_ids, dtype=np.int32).reshape(tuple(reversed(s["ne"]))))
                    else:
                        arrays.append(tensor_from_bytes(blob, s["type"], s["ne"], s["nb"], s["offset"]))
        except OpError as e:
            srcs_ok = False
            problem = str(e)
        rep["weights_ok"], rep["srcs_ok"] = weights_ok, srcs_ok
        if rep["path_ok"] and rep["out_hash_ok"] and weights_ok and srcs_ok and all(a is not None for a in arrays):
            t1 = time.time()
            res = execute(leaf, out_arr, arrays, rows=rows, qweight=qweight)
            rep["reexec"] = res
            rep["seconds"] = round(time.time() - t1, 3)
            if res["error"] is not None:
                worst[leaf["op"]] = max(worst.get(leaf["op"], 0.0), res["error"])
            if not res["ok"]:
                n_bad_open += 1
        else:
            rep["error"] = problem or "opening failed integrity checks"
            n_bad_open += 1
        opening_reports.append(rep)
    check("openings", n_bad_open == 0, f"{len(tdoc['openings']) - n_bad_open} of {len(tdoc['openings'])} openings verified",
          ops=dict(ops_seen), worst_error=worst)

    verdict = "accept" if not hard_fail else "reject"
    return {
        "verdict": verdict,
        "reason": "trace verified" if verdict == "accept" else hard_fail[0],
        "checks": checks,
        "openings": opening_reports,
        "coverage": {"n_leaves": len(leaves), "n_graphs": n_graphs, "k": len(tdoc["openings"]), "ops": dict(ops_seen)},
        "seconds": {"total": round(time.time() - t0, 3), "openings": round(time.time() - t_open, 3)},
    }


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 24):
            h.update(block)
    return h.hexdigest()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Verify a llama-receipts activation trace without running the model.")
    p.add_argument("receipt")
    p.add_argument("--model", required=True, help="the GGUF the receipt claims")
    p.add_argument("--trace", help="sidecar path (default: <receipt without .json>.trace.json)")
    p.add_argument("--report", help="write the full report as JSON")
    p.add_argument("--challenge-seed", default="", help="hex seed this verifier issued; the sidecar must record the same one")
    a = p.parse_args(argv)
    rec = json.loads(Path(a.receipt).read_text())
    tpath = Path(a.trace) if a.trace else Path(a.receipt).with_name(Path(a.receipt).name.removesuffix(".json") + ".trace.json")
    if not tpath.exists():
        print(f"no trace sidecar at {tpath}")
        return 2
    tdoc = json.loads(tpath.read_text())
    res = verify_trace(rec, tdoc, a.model, challenge_seed=bytes.fromhex(a.challenge_seed))
    print(f"verdict: {res['verdict']} ({res['reason']})")
    for name, c in res.get("checks", {}).items():
        print(f"  {name:22s} {'ok ' if c['ok'] else 'BAD'} {c.get('detail', '')}")
    cov = res.get("coverage", {})
    worst = res["checks"].get("openings", {}).get("worst_error", {})
    print(f"  openings {cov.get('k')} of {cov.get('n_leaves')} leaves; ops {cov.get('ops')}; "
          f"worst error {{{', '.join(f'{k}: {v:.1e}' for k, v in worst.items())}}}; {res.get('seconds')}")
    if a.report:
        Path(a.report).write_text(json.dumps(res, indent=1))
    return 0 if res["verdict"] == "accept" else 1


if __name__ == "__main__":
    raise SystemExit(main())
