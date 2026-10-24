"""CUDA inference kernel tests; skipped unless nvcc and MSVC cl are on PATH."""

from __future__ import annotations

import shutil
import os

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA 不可用", allow_module_level=True)
if shutil.which("cl") is None and shutil.which("cl.exe") is None:
    pytest.skip("MSVC cl 不在 PATH（source vcvars64 或用 Developer Prompt 运行）", allow_module_level=True)

from tyloinfer.kernels.cuda.gated_delta_net import gated_delta_net as gdn_cuda  # noqa: E402
from tyloinfer.kernels.gated_delta_net import gated_delta_net as gdn_ref  # noqa: E402
from tyloinfer.kernels.cuda._ext import ext  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from tyloinfer.kernels.cuda.norm import rms_norm, l2_norm  # noqa: E402
from tyloinfer.kernels.cuda.acc import acc  # noqa: E402
from tyloinfer.kernels.cuda.rope import rope  # noqa: E402
from tyloinfer.kernels.cuda.attention import (  # noqa: E402
    attention,
    sliding_window_attention,
    sliding_window_attention_cached,
)
from tyloinfer.kernels.cuda.kv_cache import (  # noqa: E402
    KVCache,
    SlidingWindowKVCache,
    kv_cache_write,
    kv_cache_write_ring,
)
from tyloinfer.kernels.cuda.activation import gelu_mul, silu_mul  # noqa: E402
from tyloinfer.kernels.cuda.embedding import embedding  # noqa: E402
from tyloinfer.kernels.cuda.sampling import sample, sample_greedy  # noqa: E402
from tyloinfer.kernels.cuda.ssm_conv import ssm_conv_silu  # noqa: E402

DEV = "cuda"


def _dec_g(*shape):
    """Use genuine decay gating (g<0 and exp(g)<1) to avoid state explosion."""
    return -(torch.rand(*shape, device=DEV) * 3 + 0.5)


def _inputs(B, H, T, D, kda=False, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(B, H, T, D, device=DEV) * 0.1
    k = torch.randn(B, H, T, D, device=DEV) * 0.1
    v = torch.randn(B, H, T, D, device=DEV) * 0.1
    g = _dec_g(B, H, T, D) if kda else _dec_g(B, H, T)
    beta = torch.sigmoid(torch.randn(B, H, T, device=DEV)) * 0.5
    return q, k, v, g, beta


def test_gdn_cuda_scalar_gate_matches_ref():
    q, k, v, g, beta = _inputs(2, 4, 32, 32)
    yc, sc = gdn_cuda(q, k, v, g, beta)
    yr, sr = gdn_ref(q, k, v, g, beta)
    assert ((yc - yr).norm() / yr.norm()).item() < 1e-5
    assert (sc - sr).abs().max().item() < 1e-5


def test_gdn_cuda_kda_matches_ref():
    q, k, v, g, beta = _inputs(1, 2, 16, 32, kda=True)
    yc, _ = gdn_cuda(q, k, v, g, beta)
    yr, _ = gdn_ref(q, k, v, g, beta)
    assert ((yc - yr).norm() / yr.norm()).item() < 1e-5


def test_gdn_cuda_decode_column_path_matches_ref():
    q, k, v, g, beta = _inputs(1, 8, 1, 64, seed=33)
    state = torch.randn(1, 8, 64, 64, device=DEV) * 0.01
    old = os.environ.get("TYLOINFER_GDN_COLUMN")
    os.environ["TYLOINFER_GDN_COLUMN"] = "1"
    try:
        yc, sc = gdn_cuda(q, k, v, g, beta, state=state)
        yr, sr = gdn_ref(q, k, v, g, beta, state=state)
        assert ((yc - yr).norm() / yr.norm()).item() < 1e-5
        assert (sc - sr).abs().max().item() < 1e-5
    finally:
        if old is None:
            os.environ.pop("TYLOINFER_GDN_COLUMN", None)
        else:
            os.environ["TYLOINFER_GDN_COLUMN"] = old


def test_gdn_cuda_d128_shared_mem():
    q, k, v, g, beta = _inputs(1, 2, 16, 128)
    yc, sc = gdn_cuda(q, k, v, g, beta)
    yr, sr = gdn_ref(q, k, v, g, beta)
    assert ((yc - yr).norm() / yr.norm()).item() < 1e-5
    assert torch.isfinite(yc).all()


def test_gdn_cuda_state_carryover():
    q, k, v, g, beta = _inputs(1, 2, 12, 32)
    _, s_full = gdn_cuda(q, k, v, g, beta)
    _, s1 = gdn_cuda(q[:, :, :6], k[:, :, :6], v[:, :, :6], g[:, :, :6], beta[:, :, :6])
    _, s2 = gdn_cuda(q[:, :, 6:], k[:, :, 6:], v[:, :, 6:], g[:, :, 6:], beta[:, :, 6:], state=s1)
    assert (s2 - s_full).abs().max().item() < 1e-4


# ---------------------------------------------------------------------------
# Standard-operator CUDA kernels
# ---------------------------------------------------------------------------
def test_rms_norm_cuda():
    torch.manual_seed(0)
    x = torch.randn(8, 128, device=DEV)
    w = torch.randn(128, device=DEV)
    eps = 1e-6
    var = x.pow(2).mean(-1, keepdim=True)
    y_ref = (x * torch.rsqrt(var + eps)) * w
    torch.testing.assert_close(rms_norm(x, w, eps), y_ref, atol=1e-5, rtol=1e-5)


def test_l2_norm_cuda():
    torch.manual_seed(1)
    x = torch.randn(4, 64, device=DEV)
    torch.testing.assert_close(l2_norm(x), F.normalize(x, dim=-1, eps=1e-5), atol=1e-5, rtol=1e-5)


def test_acc_cuda():
    torch.manual_seed(2)
    a = torch.randn(7, 32, device=DEV)
    b = torch.randn(7, 32, device=DEV)
    torch.testing.assert_close(acc(a, b), a + b, atol=1e-6, rtol=1e-6)


def test_acc_cuda_f16_preserves_dtype():
    torch.manual_seed(202)
    a = torch.randn(7, 32, device=DEV, dtype=torch.float16)
    b = torch.randn(7, 32, device=DEV, dtype=torch.float16)
    y = acc(a, b)
    assert y.dtype == torch.float16
    torch.testing.assert_close(y, a + b, atol=0, rtol=0)


@pytest.mark.parametrize("rows", [1, 16])
def test_gemma4_fused_pre_norms_match_materialized_path(rows):
    torch.manual_seed(203 + rows)
    width = 2816
    eps = 1e-6
    residual = torch.randn(rows, width, device=DEV, dtype=torch.float16)
    attn = torch.randn_like(residual)
    weights = [
        torch.randn(width, device=DEV, dtype=torch.float32)
        for _ in range(4)
    ]

    attn_post = ext().rms_norm_f16_cuda(attn, weights[0], eps, 0.0)
    residual_ref = ext().acc_cuda(residual, attn_post)
    dense_ref = ext().rms_norm_f16_cuda(residual_ref, weights[1], eps, 0.0)
    router_ref = ext().rms_norm_offset_cuda(
        residual_ref.float().contiguous(), weights[2], eps, 0.0
    )
    moe_ref = ext().rms_norm_f16_cuda(residual_ref, weights[3], eps, 0.0)

    actual = ext().gemma4_attn_residual_pre_norms_f16_cuda(
        residual, attn, *weights, eps
    )
    for got, expected in zip(actual, (residual_ref, dense_ref, router_ref, moe_ref)):
        torch.testing.assert_close(got, expected, atol=0, rtol=0)


@pytest.mark.parametrize("rows", [1, 16])
def test_gemma4_fused_ffn_merge_matches_materialized_path(rows):
    torch.manual_seed(223 + rows)
    width = 2816
    eps = 1e-6
    dense = torch.randn(rows, width, device=DEV, dtype=torch.float16)
    moe = torch.randn_like(dense)
    residual = torch.randn_like(dense)
    weights = [
        torch.randn(width, device=DEV, dtype=torch.float32)
        for _ in range(3)
    ]
    layer_scale = torch.tensor([0.9375], device=DEV, dtype=torch.float16)

    dense_post = ext().rms_norm_f16_cuda(dense, weights[0], eps, 0.0)
    moe_post = ext().rms_norm_f16_cuda(moe, weights[1], eps, 0.0)
    combined = (dense_post + moe_post).contiguous()
    post = ext().rms_norm_f16_cuda(combined, weights[2], eps, 0.0)
    expected = (ext().acc_cuda(residual, post) * layer_scale).contiguous()

    actual = ext().gemma4_ffn_merge_f16_cuda(
        dense, moe, residual, *weights, layer_scale, eps
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_silu_mul_cuda_f32():
    torch.manual_seed(21)
    gate = torch.randn(5, 37, device=DEV, dtype=torch.float32)
    up = torch.randn(5, 37, device=DEV, dtype=torch.float32)
    torch.testing.assert_close(silu_mul(gate, up), F.silu(gate) * up, atol=1e-6, rtol=1e-6)


def test_silu_mul_cuda_f16():
    torch.manual_seed(22)
    gate = torch.randn(6, 41, device=DEV, dtype=torch.float16)
    up = torch.randn(6, 41, device=DEV, dtype=torch.float16)
    ref = (F.silu(gate.float()) * up.float()).to(torch.float16)
    torch.testing.assert_close(silu_mul(gate, up), ref, atol=1e-3, rtol=1e-3)


def _ssm_conv_ref(conv_input, weight, n_tokens):
    if weight.dim() == 3:
        K = weight.shape[-1]
        w = weight[:, 0, :]
    elif weight.shape[0] == conv_input.shape[-1]:
        K = weight.shape[1]
        w = weight
    else:
        K = weight.shape[0]
        w = weight.t()
    windows = conv_input[:, : n_tokens + K - 1].unfold(1, K, 1)
    return F.silu((windows * w.view(1, 1, w.size(0), K)).sum(dim=-1))


def test_ssm_conv_silu_cuda_c1k_matches_torch():
    torch.manual_seed(29)
    B, T, C, K = 2, 7, 65, 4
    conv_input = torch.randn(B, T + K - 1, C, device=DEV)
    weight = torch.randn(C, 1, K, device=DEV)
    y = ssm_conv_silu(conv_input, weight, T)
    torch.testing.assert_close(y, _ssm_conv_ref(conv_input, weight, T), atol=1e-5, rtol=1e-5)


def test_ssm_conv_silu_cuda_kc_matches_torch():
    torch.manual_seed(30)
    B, T, C, K = 1, 9, 33, 5
    conv_input = torch.randn(B, T + K - 1, C, device=DEV)
    weight = torch.randn(K, C, device=DEV)
    y = ssm_conv_silu(conv_input, weight, T)
    torch.testing.assert_close(y, _ssm_conv_ref(conv_input, weight, T), atol=1e-5, rtol=1e-5)


def test_ssm_conv_silu_cuda_ck_matches_torch():
    torch.manual_seed(32)
    B, T, C, K = 1, 9, 33, 5
    conv_input = torch.randn(B, T + K - 1, C, device=DEV)
    weight = torch.randn(C, K, device=DEV)
    y = ssm_conv_silu(conv_input, weight, T)
    torch.testing.assert_close(y, _ssm_conv_ref(conv_input, weight, T), atol=1e-5, rtol=1e-5)


def test_ssm_conv_silu_cuda_bias_matches_torch():
    torch.manual_seed(31)
    B, T, C, K = 2, 6, 17, 4
    conv_input = torch.randn(B, T + K - 1, C, device=DEV)
    weight = torch.randn(C, 1, K, device=DEV)
    bias = torch.randn(C, device=DEV)
    windows = conv_input[:, : T + K - 1].unfold(1, K, 1)
    ref = F.silu((windows * weight[:, 0, :].view(1, 1, C, K)).sum(dim=-1) + bias.view(1, 1, C))
    y = ssm_conv_silu(conv_input, weight, T, bias)
    torch.testing.assert_close(y, ref, atol=1e-5, rtol=1e-5)


def test_kv_cache_write_matches_assignment():
    torch.manual_seed(23)
    B, H, T, D, max_seq = 2, 3, 4, 8, 11
    k = torch.randn(B, H, T, D, device=DEV)
    v = torch.randn(B, H, T, D, device=DEV)
    kc = torch.zeros(B, H, max_seq, D, device=DEV)
    vc = torch.zeros(B, H, max_seq, D, device=DEV)
    pos = torch.tensor([1, 3, 7, 9], device=DEV, dtype=torch.int64)
    kv_cache_write(kc, vc, k, v, pos)
    k_ref = torch.zeros_like(kc)
    v_ref = torch.zeros_like(vc)
    k_ref[:, :, pos.cpu().tolist(), :] = k
    v_ref[:, :, pos.cpu().tolist(), :] = v
    torch.testing.assert_close(kc, k_ref, atol=0, rtol=0)
    torch.testing.assert_close(vc, v_ref, atol=0, rtol=0)


def test_kv_cache_write_batch_positions():
    torch.manual_seed(24)
    B, H, T, D, max_seq = 2, 2, 3, 8, 9
    k = torch.randn(B, H, T, D, device=DEV)
    v = torch.randn(B, H, T, D, device=DEV)
    kc = torch.zeros(B, H, max_seq, D, device=DEV)
    vc = torch.zeros(B, H, max_seq, D, device=DEV)
    pos = torch.tensor([[0, 2, 4], [1, 3, 5]], device=DEV, dtype=torch.int64)
    kv_cache_write(kc, vc, k, v, pos)
    k_ref = torch.zeros_like(kc)
    v_ref = torch.zeros_like(vc)
    for b in range(B):
        for t in range(T):
            p = int(pos[b, t].item())
            k_ref[b, :, p, :] = k[b, :, t, :]
            v_ref[b, :, p, :] = v[b, :, t, :]
    torch.testing.assert_close(kc, k_ref, atol=0, rtol=0)
    torch.testing.assert_close(vc, v_ref, atol=0, rtol=0)


def test_kv_cache_append_attention_matches_sdpa():
    torch.manual_seed(25)
    B, Hq, Hkv, D = 1, 4, 2, 16
    cache = KVCache(B, Hkv, 8, D, device=DEV, dtype=torch.float32)
    k0 = torch.randn(B, Hkv, 3, D, device=DEV)
    v0 = torch.randn(B, Hkv, 3, D, device=DEV)
    k1 = torch.randn(B, Hkv, 2, D, device=DEV)
    v1 = torch.randn(B, Hkv, 2, D, device=DEV)
    cache.append(k0, v0)
    kc, vc = cache.append(k1, v1)
    q = torch.randn(B, Hq, 2, D, device=DEV)
    y = attention(q, kc, vc, causal=True)
    rep = Hq // Hkv
    kr = torch.cat([k0, k1], dim=2).repeat_interleave(rep, dim=1)
    vr = torch.cat([v0, v1], dim=2).repeat_interleave(rep, dim=1)
    att = torch.full((2, 5), float("-inf"), device=DEV)
    offset = 3
    for tq in range(2):
        for s in range(5):
            if s <= tq + offset:
                att[tq, s] = 0.0
    y_ref = F.scaled_dot_product_attention(q, kr, vr, attn_mask=att.view(1, 1, 2, 5))
    torch.testing.assert_close(y, y_ref, atol=2e-4, rtol=2e-4)


def test_kv_cache_default_f16_and_attention_matches_sdpa():
    torch.manual_seed(251)
    B, Hq, Hkv, D = 1, 4, 2, 16
    cache = KVCache(B, Hkv, 8, D, device=DEV)
    assert cache.k.dtype == torch.float16
    k = torch.randn(B, Hkv, 5, D, device=DEV, dtype=torch.float16)
    v = torch.randn(B, Hkv, 5, D, device=DEV, dtype=torch.float16)
    kc, vc = cache.append(k, v)
    assert kc.dtype == torch.float16
    q = torch.randn(B, Hq, 2, D, device=DEV, dtype=torch.float16)
    y = attention(q, kc, vc, causal=True)
    assert y.dtype == torch.float16
    rep = Hq // Hkv
    kr = k.repeat_interleave(rep, dim=1)
    vr = v.repeat_interleave(rep, dim=1)
    att = torch.full((2, 5), float("-inf"), device=DEV, dtype=torch.float16)
    offset = 3
    for tq in range(2):
        for s in range(5):
            if s <= tq + offset:
                att[tq, s] = 0.0
    y_ref = F.scaled_dot_product_attention(q, kr, vr, attn_mask=att.view(1, 1, 2, 5))
    torch.testing.assert_close(y, y_ref, atol=1e-3, rtol=1e-3)


def test_kv_cache_grows_without_full_prealloc():
    B, Hkv, D = 1, 2, 8
    cache = KVCache(B, Hkv, 1024, D, device=DEV, initial_capacity=2)
    assert cache.capacity == 2
    k = torch.randn(B, Hkv, 5, D, device=DEV)
    v = torch.randn(B, Hkv, 5, D, device=DEV)
    kc, vc = cache.append(k, v)
    assert cache.capacity == 8
    assert tuple(kc.shape) == (B, Hkv, 5, D)
    torch.testing.assert_close(kc, k.to(torch.float16), atol=0, rtol=0)
    torch.testing.assert_close(vc, v.to(torch.float16), atol=0, rtol=0)


def test_embedding_lookup_cuda_f16():
    torch.manual_seed(26)
    weight = torch.randn(17, 13, device=DEV, dtype=torch.float16)
    ids = torch.tensor([[0, 3, 16], [8, 2, 5]], device=DEV, dtype=torch.int64)
    torch.testing.assert_close(embedding(weight, ids), weight[ids], atol=0, rtol=0)


def test_embedding_lookup_cuda_f32():
    torch.manual_seed(27)
    weight = torch.randn(19, 11, device=DEV, dtype=torch.float32)
    ids = torch.tensor([18, 1, 7, 0], device=DEV, dtype=torch.int64)
    torch.testing.assert_close(embedding(weight, ids), weight[ids], atol=0, rtol=0)


def test_sample_greedy_cuda_f32_f16_bf16():
    logits = torch.tensor(
        [[1.0, 5.0, 5.0, -1.0], [0.0, -2.0, 3.0, 1.0]],
        device=DEV,
        dtype=torch.float32,
    )
    ref = torch.tensor([1, 2], device=DEV, dtype=torch.int64)
    torch.testing.assert_close(sample_greedy(logits), ref, atol=0, rtol=0)
    torch.testing.assert_close(sample_greedy(logits.to(torch.float16)), ref, atol=0, rtol=0)
    torch.testing.assert_close(sample_greedy(logits.to(torch.bfloat16)), ref, atol=0, rtol=0)


def test_sample_softmax_cuda_matches_reference():
    logits = torch.tensor(
        [[0.1, 0.4, -0.2, 1.0], [2.0, -1.0, 0.0, 0.5]],
        device=DEV,
        dtype=torch.float32,
    )
    rnd = torch.tensor([0.10, 0.95], device=DEV, dtype=torch.float32)
    y = sample(logits, temperature=0.7, random=rnd)
    probs = torch.softmax(logits / 0.7, dim=-1)
    ref = torch.searchsorted(torch.cumsum(probs, dim=-1), rnd[:, None]).squeeze(1).to(torch.int64)
    torch.testing.assert_close(y, ref, atol=0, rtol=0)


def test_sample_top_k_top_p_cuda_matches_reference():
    logits = torch.tensor(
        [[0.1, 2.0, 1.2, -0.5, 0.8], [1.0, 0.9, 0.1, 2.5, -1.0]],
        device=DEV,
        dtype=torch.float32,
    )
    rnd = torch.tensor([0.40, 0.80], device=DEV, dtype=torch.float32)
    y = sample(logits, temperature=1.0, top_k=3, top_p=0.75, random=rnd)
    refs = []
    vals, idx = torch.topk(logits, k=3, dim=-1)
    probs = torch.softmax(vals, dim=-1)
    for b in range(logits.size(0)):
        cutoff = 0.75
        c = torch.cumsum(probs[b], dim=0)
        keep = int((c >= cutoff).nonzero()[0].item()) + 1
        kept = probs[b, :keep]
        kept = kept / kept.sum()
        j = int(torch.searchsorted(torch.cumsum(kept, dim=0), rnd[b]).item())
        refs.append(int(idx[b, j].item()))
    ref = torch.tensor(refs, device=DEV, dtype=torch.int64)
    torch.testing.assert_close(y, ref, atol=0, rtol=0)


def test_sample_top_k_cuda_tie_breaks_by_lowest_token_id():
    logits = torch.ones((1, 8), device=DEV, dtype=torch.float32)
    rnd = torch.tensor([0.9], device=DEV, dtype=torch.float32)
    y = sample(logits, temperature=1.0, top_k=3, top_p=1.0, random=rnd)
    torch.testing.assert_close(y, torch.tensor([2], device=DEV), atol=0, rtol=0)


def test_sample_penalties_cuda_counts_and_updates_in_place():
    counts = torch.zeros(5, device=DEV, dtype=torch.int32)
    tokens = torch.tensor([1, 1, 3], device=DEV, dtype=torch.int64)
    ext().sample_token_counts_add_cuda(counts, tokens)
    torch.testing.assert_close(
        counts,
        torch.tensor([0, 2, 0, 1, 0], device=DEV, dtype=torch.int32),
        atol=0,
        rtol=0,
    )

    logits = torch.tensor([[2.0, -2.0, 1.0, 3.0, 4.0]], device=DEV, dtype=torch.float32)
    out = ext().sample_apply_penalties_cuda(logits, counts, 0.5, 0.25, 2.0)
    expected = torch.tensor([[2.0, -5.0, 1.0, 0.75, 4.0]], device=DEV, dtype=torch.float32)
    assert out.data_ptr() == logits.data_ptr()
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


def _rope_ref(x, pos, base=1e6, rotary_dim=None, sections=None):
    T, D = x.shape[-2], x.shape[-1]
    RD = D if rotary_dim is None else int(rotary_dim)
    half = RD // 2
    out = x.clone()
    freqs = base ** (-torch.arange(0, RD, 2, device=x.device, dtype=torch.float32) / RD)
    if sections is None:
        pos_used = pos.float().view(1, T).expand(half, T)
    else:
        axes = torch.empty(half, device=x.device, dtype=torch.long)
        s0, s1, s2 = sections
        axes[:s0] = 0
        axes[s0 : s0 + s1] = 1
        axes[s0 + s1 : s0 + s1 + s2] = 2
        pos2 = pos.float()
        if pos2.dim() == 1:
            pos2 = pos2.view(1, T).expand(3, T)
        pos_used = pos2[axes]
    angles = pos_used.t() * freqs.view(1, half)
    shape = (1,) * (x.dim() - 2) + (T, half)
    cos, sin = angles.cos().view(shape), angles.sin().view(shape)
    x0, x1 = x[..., :half], x[..., half:RD]
    out[..., :half] = x0 * cos - x1 * sin
    out[..., half:RD] = x1 * cos + x0 * sin
    return out


def test_rope_cuda_pos0_identity():
    torch.manual_seed(3)
    x = torch.randn(2, 3, 4, 8, device=DEV)
    torch.testing.assert_close(rope(x, torch.zeros(4, device=DEV)), x, atol=1e-5, rtol=1e-5)


def test_rope_cuda_matches_ref():
    torch.manual_seed(4)
    x = torch.randn(2, 3, 6, 16, device=DEV)
    pos = torch.arange(6.0, device=DEV)
    torch.testing.assert_close(rope(x, pos), _rope_ref(x, pos), atol=1e-5, rtol=1e-5)


def test_rope_cuda_table_matches_ref():
    torch.manual_seed(41)
    x = torch.randn(2, 3, 6, 16, device=DEV)
    pos = torch.arange(6, device=DEV, dtype=torch.int64) + 5
    torch.testing.assert_close(rope(x, pos, table_len=64), _rope_ref(x, pos), atol=1e-5, rtol=1e-5)


def test_rope_cuda_partial_matches_ref():
    torch.manual_seed(31)
    x = torch.randn(2, 3, 4, 10, device=DEV)
    pos = torch.arange(4.0, device=DEV) + 2
    y = rope(x, pos, rotary_dim=6)
    y_ref = _rope_ref(x, pos, rotary_dim=6)
    torch.testing.assert_close(y, y_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(y[..., 6:], x[..., 6:], atol=0, rtol=0)


def test_rope_cuda_mrope_sections_matches_ref():
    torch.manual_seed(32)
    x = torch.randn(1, 2, 5, 12, device=DEV)
    pos = torch.stack(
        [
            torch.arange(5, device=DEV),
            torch.arange(5, device=DEV) + 10,
            torch.arange(5, device=DEV) + 20,
        ],
        dim=0,
    )
    sections = (1, 2, 1)
    y = rope(x, pos, rotary_dim=8, sections=sections)
    y_ref = _rope_ref(x, pos, rotary_dim=8, sections=sections)
    torch.testing.assert_close(y, y_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(y[..., 8:], x[..., 8:], atol=0, rtol=0)


def test_attention_cuda_causal():
    torch.manual_seed(5)
    B, H, T, D = 2, 4, 8, 16
    q = torch.randn(B, H, T, D, device=DEV)
    k = torch.randn(B, H, T, D, device=DEV)
    v = torch.randn(B, H, T, D, device=DEV)
    y = attention(q, k, v, causal=True)
    y_ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(y, y_ref, atol=2e-4, rtol=2e-4)


def test_attention_cuda_gqa():
    torch.manual_seed(6)
    B, Hq, Hk, T, D = 1, 8, 2, 6, 16
    q = torch.randn(B, Hq, T, D, device=DEV)
    k = torch.randn(B, Hk, T, D, device=DEV)
    v = torch.randn(B, Hk, T, D, device=DEV)
    rep = Hq // Hk
    kr = k.repeat_interleave(rep, dim=1)
    vr = v.repeat_interleave(rep, dim=1)
    y_ref = F.scaled_dot_product_attention(q, kr, vr, is_causal=False)
    torch.testing.assert_close(attention(q, k, v, causal=False), y_ref, atol=2e-4, rtol=2e-4)


def test_attention_cuda_decode_unequal():
    """Tq != Tk: validate the causal offset s <= tq + (Tk - T) against an explicit mask."""
    torch.manual_seed(7)
    B, Hq, Hk, Tq, Tk, D = 1, 4, 2, 2, 4, 16
    q = torch.randn(B, Hq, Tq, D, device=DEV)
    k = torch.randn(B, Hk, Tk, D, device=DEV)
    v = torch.randn(B, Hk, Tk, D, device=DEV)
    rep = Hq // Hk
    kr = k.repeat_interleave(rep, dim=1)
    vr = v.repeat_interleave(rep, dim=1)
    # explicit mask matching the kernel convention
    offset = Tk - Tq
    att = torch.full((Tq, Tk), float("-inf"), device=DEV)
    for tq in range(Tq):
        for s in range(Tk):
            if s <= tq + offset:
                att[tq, s] = 0.0
    y_ref = F.scaled_dot_product_attention(q, kr, vr, attn_mask=att.view(1, 1, Tq, Tk))
    torch.testing.assert_close(attention(q, k, v, causal=True), y_ref, atol=2e-4, rtol=2e-4)


def test_attention_cuda_splitk_decode_matches_sdpa():
    torch.manual_seed(71)
    B, Hq, Hk, Tq, Tk, D = 1, 8, 2, 1, 1024, 64
    q = torch.randn(B, Hq, Tq, D, device=DEV, dtype=torch.float16)
    k = torch.randn(B, Hk, Tk, D, device=DEV, dtype=torch.float16)
    v = torch.randn(B, Hk, Tk, D, device=DEV, dtype=torch.float16)
    rep = Hq // Hk
    kr = k.repeat_interleave(rep, dim=1)
    vr = v.repeat_interleave(rep, dim=1)
    y_ref = F.scaled_dot_product_attention(q, kr, vr, is_causal=False)
    y = attention(q, k, v, causal=True)
    torch.testing.assert_close(y, y_ref, atol=1e-3, rtol=1e-3)


def _swa_reference(q, k, v, window):
    rep = q.size(1) // k.size(1)
    kr = k.repeat_interleave(rep, dim=1)
    vr = v.repeat_interleave(rep, dim=1)
    tq, tk = q.size(2), k.size(2)
    offset = tk - tq
    mask = torch.full((tq, tk), float("-inf"), device=q.device, dtype=q.dtype)
    for row in range(tq):
        end = row + offset + 1
        start = max(0, end - window)
        mask[row, start:end] = 0
    return F.scaled_dot_product_attention(q, kr, vr, attn_mask=mask.view(1, 1, tq, tk))


def test_attention_swa_gqa_unequal_lengths_matches_sdpa():
    torch.manual_seed(72)
    q = torch.randn(1, 8, 7, 64, device=DEV, dtype=torch.float16)
    k = torch.randn(1, 2, 19, 64, device=DEV, dtype=torch.float16)
    v = torch.randn_like(k)
    actual = sliding_window_attention(q, k, v, window=5)
    expected = _swa_reference(q, k, v, 5)
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)


def test_attention_swa_circular_cache_wrap_matches_sdpa():
    torch.manual_seed(73)
    B, Hq, Hk, D, capacity, window = 1, 8, 2, 64, 8, 5
    k_all = torch.randn(B, Hk, 19, D, device=DEV, dtype=torch.float16)
    v_all = torch.randn_like(k_all)
    k_cache = torch.empty(B, Hk, capacity, D, device=DEV, dtype=torch.float16)
    v_cache = torch.empty_like(k_cache)
    for start, end in ((0, 8), (8, 16), (16, 19)):
        kv_cache_write_ring(k_cache, v_cache, k_all[:, :, start:end], v_all[:, :, start:end], start)
    q = torch.randn(B, Hq, 3, D, device=DEV, dtype=torch.float16)
    seq_len = torch.tensor([19], device=DEV, dtype=torch.int64)
    actual = sliding_window_attention_cached(q, k_cache, v_cache, seq_len, window)
    expected = _swa_reference(q, k_all, v_all, window)
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)


def test_attention_swa_circular_cache_splitk_matches_sdpa():
    torch.manual_seed(76)
    B, Hq, Hk, D, capacity, window, length = 1, 8, 2, 64, 768, 512, 900
    k_all = torch.randn(B, Hk, length, D, device=DEV, dtype=torch.float16)
    v_all = torch.randn_like(k_all)
    k_cache = torch.empty(B, Hk, capacity, D, device=DEV, dtype=torch.float16)
    v_cache = torch.empty_like(k_cache)
    kv_cache_write_ring(k_cache, v_cache, k_all, v_all, 0)
    q = torch.randn(B, Hq, 3, D, device=DEV, dtype=torch.float16)
    seq_len = torch.tensor([length], device=DEV, dtype=torch.int64)
    actual = sliding_window_attention_cached(q, k_cache, v_cache, seq_len, window)
    expected = _swa_reference(q, k_all, v_all, window)
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)


def test_sliding_window_cache_rejects_oversized_ubatch():
    cache = SlidingWindowKVCache(1, 2, 8, 64, ubatch_capacity=3)
    k = torch.randn(1, 2, 4, 64, device=DEV, dtype=torch.float16)
    with pytest.raises(ValueError, match="exceeding ubatch_capacity"):
        cache.append(k, k)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_gelu_mul_matches_tanh_approximation(dtype):
    torch.manual_seed(74)
    gate = torch.randn(33, 71, device=DEV, dtype=dtype)
    up = torch.randn_like(gate)
    actual = gelu_mul(gate, up)
    expected = F.gelu(gate, approximate="tanh") * up
    atol = rtol = 1e-3 if dtype == torch.float16 else 2e-6
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def test_gelu_mul_saturates_f16_overflow_without_nonfinite_values():
    gate = torch.tensor(
        [[200.0, 200.0, -200.0]], device=DEV, dtype=torch.float16
    )
    up = torch.tensor(
        [[400.0, -400.0, 400.0]], device=DEV, dtype=torch.float16
    )
    actual = gelu_mul(gate, up)
    expected = torch.tensor(
        [[65504.0, -65504.0, -0.0]], device=DEV, dtype=torch.float16
    )
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
