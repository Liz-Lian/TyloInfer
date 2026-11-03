from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from tyloinfer.cli import main
from tyloinfer.formats.gguf_residual import RowResidual
from tyloinfer.formats.gyro import FORMATS, decode_row, row_bytes
from tyloinfer.formats.weight_rows import GgufWeightRows


def _string(value):
    raw = value.encode()
    return struct.pack("<Q", len(raw)) + raw


def write_gguf(path, name, shape, kind, packed):
    metadata = _string("general.architecture") + struct.pack("<I", 8) + _string("qwen3_5")
    header = b"GGUF" + struct.pack("<IQQ", 3, 1, 1) + metadata
    header += (
        _string(name)
        + struct.pack("<I", len(shape))
        + struct.pack("<" + "Q" * len(shape), *reversed(shape))
        + struct.pack("<IQ", kind, 0)
    )
    path.write_bytes(header + bytes((-len(header)) % 32) + packed)


@pytest.mark.parametrize("name", FORMATS)
def test_gguf_trellis_source_rows(tmp_path, name):
    width, count = 256, 3
    size = row_bytes(name, width)
    raw = bytearray((i * 17 + 5) % 256 for i in range(count * size))
    block = size // 2
    for offset in range(0, len(raw), block):
        raw[offset : offset + 2] = np.float16(0.125).tobytes()
    path = tmp_path / "base.gguf"
    write_gguf(path, "blk.0.ffn_gate.weight", (count, width), FORMATS[name][0], bytes(raw))
    source = GgufWeightRows(path, "blk.0.ffn_gate.weight")
    assert source.shape == (count, width)
    np.testing.assert_array_equal(
        source.read_rows(1, 2)[0], decode_row(bytes(raw[size : 2 * size]), name, width)
    )


def test_cli_array_residual(tmp_path, capsys):
    reference = np.random.default_rng(41).normal(size=(3, 131)).astype(np.float32)
    base = np.rint(reference * 2) / 2
    np.save(tmp_path / "base.npy", base)
    np.save(tmp_path / "reference.npy", reference)
    (tmp_path / "rows.json").write_text(json.dumps(["TQK4", None, "TQP7"]))
    output = tmp_path / "weights.tqres"
    assert (
        main(
            [
                "fit-residual",
                "--base",
                str(tmp_path / "base.npy"),
                "--reference",
                str(tmp_path / "reference.npy"),
                "--row-formats",
                str(tmp_path / "rows.json"),
                "--output",
                str(output),
                "--workers",
                "2",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["corrected_rows"] == 2
    residual = RowResidual.load(output)
    assert residual.formats == ("TQK4", None, "TQP7")
    np.testing.assert_array_equal(np.load(tmp_path / "base.npy"), base)


def test_cli_quantized_source_and_manifest(tmp_path, capsys):
    import gguf
    from gguf.constants import GGMLQuantizationType
    from gguf.quants import quantize

    target = np.random.default_rng(51).normal(size=(3, 256)).astype(np.float32)
    base = tmp_path / "base.gguf"
    writer = gguf.GGUFWriter(base, "qwen3_5")
    writer.add_tensor(
        "blk.0.ffn_gate.weight",
        quantize(target, GGMLQuantizationType.Q8_0),
        raw_dtype=GGMLQuantizationType.Q8_0,
    )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    original = base.read_bytes()
    np.save(tmp_path / "target.npy", target)
    manifest, output = tmp_path / "model.row-residual.json", tmp_path / "gate.tqres"
    assert (
        main(
            [
                "fit-residual",
                "--base",
                str(base),
                "--reference",
                str(tmp_path / "target.npy"),
                "--tensor",
                "blk.0.ffn_gate.weight",
                "--output",
                str(output),
                "--manifest",
                str(manifest),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["rows"] == 3
    document = json.loads(manifest.read_text())
    assert document["base"] == "base.gguf"
    assert document["tensors"] == {"blk.0.ffn_gate.weight": "gate.tqres"}
    assert base.read_bytes() == original


def test_cli_refuses_source_overwrite(tmp_path):
    path = tmp_path / "source.npy"
    np.save(path, np.ones((1, 128), dtype=np.float32))
    original = path.read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        main(["fit-residual", "--base", str(path), "--reference", str(path), "--output", str(path)])
    assert path.read_bytes() == original
