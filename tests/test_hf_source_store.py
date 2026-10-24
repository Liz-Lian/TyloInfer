from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from tyloinfer.formats.assets import MODEL_CONFIG_ASSET, MODEL_GRAPH_ASSET
from tyloinfer.formats.hf_source import HfSourceTensorStore
from tyloinfer.formats.io import BFloat16Array
from tyloinfer.formats.mx import MXFP4_DTYPE, MxTensor


def _write_config(root: Path, value: dict[str, object]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(value), encoding="utf-8")


def _write_raw_safetensors(
    path: Path,
    tensors: dict[str, tuple[str, tuple[int, ...], bytes]],
) -> None:
    header: dict[str, object] = {}
    payloads: list[bytes] = []
    offset = 0
    for name, (dtype, shape, payload) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + len(payload)],
        }
        payloads.append(payload)
        offset += len(payload)
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(payloads))


def test_hf_source_store_exposes_registered_architecture_canonically(
    tmp_path: Path,
) -> None:
    root = tmp_path / "qwen"
    _write_config(
        root,
        {"model_type": "qwen3_5", "num_hidden_layers": 1},
    )
    source = np.arange(8, dtype=np.float32).reshape(2, 4)
    raw = (source.view(np.uint32) >> 16).astype(np.uint16)
    _write_raw_safetensors(root / "model.safetensors", {
        "model.language_model.embed_tokens.weight": ("BF16", source.shape, raw.tobytes())})

    with HfSourceTensorStore(root) as store:
        assert MODEL_CONFIG_ASSET in store
        assert MODEL_GRAPH_ASSET in store
        assert "model.token_embedding.weight" in store
        value = store["model.token_embedding.weight"]

    assert isinstance(value, BFloat16Array)
    assert value.shape == (2, 4)
    recovered = (np.asarray(value, dtype=np.uint16).astype(np.uint32) << 16).view(
        np.float32
    )
    np.testing.assert_array_equal(recovered, source)


def test_hf_source_store_builds_generic_split_expert_views(tmp_path: Path) -> None:
    root = tmp_path / "glm"
    _write_config(
        root,
        {
            "model_type": "glm5_next",
            "text_config": {
                "model_type": "glm5_next",
                "num_hidden_layers": 1,
            },
        },
    )
    tensors = {
        f"model.language_model.layers.0.mlp.experts.{expert}.gate_proj.weight": (
            np.full((2, 4), float(expert + 1), dtype=np.float16)
        )
        for expert in range(2)
    }
    save_file(tensors, root / "model.safetensors")

    with HfSourceTensorStore(root) as store:
        bank = store["model.block.0.mlp.experts.gate.weight"]

    assert isinstance(bank, np.ndarray)
    assert bank.shape == (2, 2, 4)
    np.testing.assert_array_equal(bank[0], np.ones((2, 4), dtype=np.float16))
    np.testing.assert_array_equal(bank[1], np.full((2, 4), 2, dtype=np.float16))


def test_hf_source_store_reads_native_mx_in_generic_expert_view(
    tmp_path: Path,
) -> None:
    root = tmp_path / "glm-mx"
    _write_config(
        root,
        {
            "model_type": "glm5_next",
            "text_config": {
                "model_type": "glm5_next",
                "num_hidden_layers": 1,
            },
        },
    )
    tensors: dict[str, tuple[str, tuple[int, ...], bytes]] = {}
    for expert in range(2):
        prefix = f"model.language_model.layers.0.mlp.experts.{expert}.gate_proj"
        tensors[prefix + ".weight"] = (
            "I8",
            (2, 16),
            bytes([0x21 + expert] * 32),
        )
        tensors[prefix + ".weight_scale"] = (
            "F8_E8M0",
            (2, 1),
            bytes([127 + expert] * 2),
        )
    _write_raw_safetensors(root / "model.safetensors", tensors)

    with HfSourceTensorStore(root) as store:
        bank = store["model.block.0.mlp.experts.gate.weight"]

    assert isinstance(bank, np.ndarray)
    assert bank.shape == (2, 2, 32)
    for expert in range(2):
        weight = MxTensor(MXFP4_DTYPE, (2, 32), np.full((2, 16), 0x21 + expert, np.uint8),
            np.full((2, 1), 127 + expert, np.uint8))
        from tyloinfer.formats.hf_source import _dequantize_mx
        np.testing.assert_array_equal(bank[expert], _dequantize_mx(weight))
