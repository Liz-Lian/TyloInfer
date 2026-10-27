"""TQ model containers and native floating-point tensor storage."""

from __future__ import annotations

import json
import math
import mmap
import os
import struct
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np

from tyloinfer.formats.assets import ASSET_DTYPE, canonical_asset_record_name
from tyloinfer.formats.companions import is_engram_shard, is_engram_table
from tyloinfer.formats.compat import LEGACY_DTYPES, canonical_dtype
from tyloinfer.formats.header import TYLOINFER_MAGIC, FileHeader, TENSOR_DTYPES
from tyloinfer.formats.mx import MX_DTYPES, MxTensor, pack_mx, unpack_mx
from tyloinfer.formats.tq import TQTensor, TQExpertTensor

MfqTensor: TypeAlias = TQTensor | TQExpertTensor | MxTensor | np.ndarray | bytes


class BFloat16Array(np.ndarray):
    """NumPy-compatible view of raw little-endian BF16 storage.

    NumPy does not provide a native BF16 dtype.  Keeping the payload as a
    tagged ``uint16`` ndarray preserves the exact on-disk bits while allowing
    the Torch and MLX runtimes to reinterpret them without a float32 staging
    copy.
    """


class Float8E4M3Array(np.ndarray):
    """Tagged raw E4M3FN bytes retained without widening to F16/BF16."""


def is_bfloat16_array(value: object) -> bool:
    return isinstance(value, BFloat16Array)


def is_float8_e4m3_array(value: object) -> bool:
    return isinstance(value, Float8E4M3Array)


def bfloat16_to_float32(value: np.ndarray) -> np.ndarray:
    """Decode a tagged/raw BF16 array into numerically exact float32 values."""

    bits = np.asarray(value, dtype="<u2")
    return (bits.astype(np.uint32) << 16).view(np.float32)


@dataclass(frozen=True)
class MMapTensorRecord:
    name: str
    dtype: str
    offset: int
    nbytes: int
    source_index: int = 0
    stored_dtype: str | None = None


def _u32(x: int) -> bytes:
    return struct.pack("<I", int(x))


def _read_u32(buf: bytes, off: int) -> tuple[int, int]:
    return struct.unpack_from("<I", buf, off)[0], off + 4


def _read_str(buf: bytes, off: int) -> tuple[str, int]:
    n, off = _read_u32(buf, off)
    s = buf[off : off + n].decode("utf-8")
    return s, off + n


# ---------------------------------------------------------------------------
# Bitstream helpers
# ---------------------------------------------------------------------------
def _uint_dtype(maximum: int) -> np.dtype:
    return np.dtype('<u1' if maximum <= 255 else '<u2' if maximum <= 65535 else '<u4' if maximum <= 4294967295 else '<u8')


def pack_bits(values: np.ndarray, bits: int) -> bytes:
    """Pack unsigned integer values into a little-endian bitstream."""

    arr = np.ascontiguousarray(values).reshape(-1)
    if bits == 8:
        return arr.astype(np.uint8, copy=False).tobytes()
    if bits == 4:
        u = arr.astype(np.uint8, copy=False)
        if u.size % 2:
            u = np.concatenate([u, np.zeros(1, dtype=np.uint8)])
        packed = u[0::2] | (u[1::2] << 4)
        return np.ascontiguousarray(packed, dtype=np.uint8).tobytes()
    if 0 < bits < 8:
        count = arr.size
        u = arr.astype(np.uint8, copy=False) & ((1 << bits) - 1)
        if u.size % 8:
            u = np.concatenate(
                [u, np.zeros((-u.size) % 8, dtype=np.uint8)]
            )
        blocks = u.reshape(-1, 8)
        packed = np.zeros((blocks.shape[0], bits), dtype=np.uint8)
        for value_index in range(8):
            bit_position = value_index * bits
            byte_index, shift = divmod(bit_position, 8)
            packed[:, byte_index] |= blocks[:, value_index] << shift
            if shift + bits > 8:
                packed[:, byte_index + 1] |= blocks[:, value_index] >> (8 - shift)
        nbytes = (count * bits + 7) // 8
        return packed.reshape(-1)[:nbytes].tobytes()
    maxval = (1 << bits) - 1
    udt = _uint_dtype(maxval)
    return arr.astype(udt, copy=False).tobytes()


def unpack_bits(blob: bytes, off: int, count: int, bits: int) -> tuple[np.ndarray, int]:
    """Unpack ``count`` unsigned integer values from a little-endian bitstream."""

    if bits == 8:
        end = off + count
        return np.frombuffer(blob, dtype=np.uint8, count=count, offset=off).copy(), end
    if bits == 4:
        nbytes = (count + 1) // 2
        packed = np.frombuffer(blob, dtype=np.uint8, count=nbytes, offset=off)
        out = np.empty(nbytes * 2, dtype=np.uint8)
        out[0::2] = packed & 0x0F
        out[1::2] = packed >> 4
        return out[:count].copy(), off + nbytes
    if bits < 8:
        nbytes = (count * bits + 7) // 8
        packed = np.frombuffer(blob, dtype=np.uint8, count=nbytes, offset=off)
        bitstream = np.unpackbits(packed, bitorder="little")[: count * bits].reshape(count, bits)
        shifts = (1 << np.arange(bits, dtype=np.uint16))
        out = (bitstream.astype(np.uint16) * shifts).sum(axis=1).astype(_uint_dtype((1 << bits) - 1))
        return out, off + nbytes
    maxval = (1 << bits) - 1
    udt = _uint_dtype(maxval)
    end = off + count * udt.itemsize
    return np.frombuffer(blob, dtype=udt, count=count, offset=off).copy(), end


_DENSE_DTYPES = {
    "BF16": np.dtype("<u2"),
    "F8_E4M3": np.dtype(np.uint8),
    "F16": np.dtype(np.float16),
    "F32": np.dtype(np.float32),
    "I32": np.dtype(np.int32),
    "I64": np.dtype(np.int64),
}
_DENSE_NAMES = {
    value: name
    for name, value in _DENSE_DTYPES.items()
    if name not in {"BF16", "F8_E4M3"}
}


def pack_dense(tensor: np.ndarray) -> tuple[str, bytes]:
    """Pack a small dense tensor, used for norm weights and metadata-like arrays."""

    bfloat16 = is_bfloat16_array(tensor)
    float8_e4m3 = is_float8_e4m3_array(tensor)
    arr = np.ascontiguousarray(tensor)
    dtype = (
        "BF16"
        if bfloat16
        else "F8_E4M3"
        if float8_e4m3
        else _DENSE_NAMES.get(arr.dtype)
    )
    if dtype is None:
        raise ValueError(f"unsupported dense dtype: {arr.dtype}")
    parts = [struct.pack("<I", arr.ndim)]
    parts.append(struct.pack(f"<{arr.ndim}q", *arr.shape))
    parts.append(arr.tobytes())
    return dtype, b"".join(parts)


def unpack_dense(dtype: str, blob: bytes) -> np.ndarray:
    if dtype not in _DENSE_DTYPES:
        raise ValueError(f"unknown dense dtype: {dtype}")
    off = 0
    ndim = struct.unpack_from("<I", blob, off)[0]
    off += 4
    shape = struct.unpack_from(f"<{ndim}q", blob, off)
    off += 8 * ndim
    arr = np.frombuffer(blob, dtype=_DENSE_DTYPES[dtype], offset=off).copy()
    result = arr.reshape(shape)
    if dtype == "BF16":
        return result.view(BFloat16Array)
    if dtype == "F8_E4M3":
        return result.view(Float8E4M3Array)
    return result


def _unpack_tensor(dtype: str, blob: bytes | memoryview) -> MfqTensor:
    if dtype == ASSET_DTYPE:
        return bytes(blob)
    if dtype == "TQ":
        return (TQExpertTensor if bytes(blob[:8]) == b'RTQEXP01' else TQTensor).from_bytes(blob)
    if dtype in MX_DTYPES:
        return unpack_mx(dtype, blob)
    if canonical_dtype(dtype) in LEGACY_DTYPES:
        raise ValueError(f"TyloInfer does not support tensor format {dtype!r} in the Python tensor loader; use the native compatibility runtime")
    return unpack_dense(dtype, blob)


def unpack_tensor_payload(dtype: str, blob: bytes | memoryview) -> MfqTensor:
    """Decode one self-contained tensor payload using its public MFQ dtype."""

    return _unpack_tensor(dtype, blob)


def pack_tensor_payload(tensor: MfqTensor) -> tuple[str, bytes]:
    """Encode one tensor into its public MFQ dtype and self-contained payload."""

    return _pack_tensor(tensor)


def _pack_tensor(tensor: MfqTensor, *, allow_moe: bool = True) -> tuple[str, bytes]:
    """Return one tensor's public dtype label and native payload."""

    if isinstance(tensor, bytes):
        return ASSET_DTYPE, tensor
    if isinstance(tensor, (TQTensor, TQExpertTensor)):
        return "TQ", tensor.to_bytes()
    if isinstance(tensor, MxTensor):
        return tensor.dtype, pack_mx(tensor)
    if isinstance(tensor, np.ndarray):
        return pack_dense(tensor)
    raise TypeError(f"unsupported tensor type: {type(tensor)!r}")


# ---------------------------------------------------------------------------
# File-level save / load
# ---------------------------------------------------------------------------
def save(path: str | Path, header: FileHeader, tensors: dict[str, MfqTensor]) -> None:
    """Write an ``.mfq`` file."""

    packed: dict[str, tuple[str, bytes]] = {}
    for name, t in tensors.items():
        try:
            packed[name] = _pack_tensor(t)
        except TypeError as exc:
            raise TypeError(f"unsupported tensor type for {name!r}: {type(t)!r}") from exc
    version = int(header.version)
    extra = dict(header.extra)
    if extra and version < 2:
        version = 2
    with open(path, "wb") as f:
        f.write(TYLOINFER_MAGIC)
        f.write(_u32(version))
        arch_b = header.model_arch.encode("utf-8")
        f.write(_u32(len(arch_b)))
        f.write(arch_b)
        if version >= 2:
            f.write(_u32(len(extra)))
            for k, v in extra.items():
                kb = str(k).encode("utf-8")
                vb = json.dumps(v).encode("utf-8")
                f.write(_u32(len(kb)))
                f.write(kb)
                f.write(_u32(len(vb)))
                f.write(vb)
        f.write(_u32(len(tensors)))
        for name in tensors:
            name_b = name.encode("utf-8")
            dtype, blob = packed[name]
            dtype_b = dtype.encode("utf-8")
            f.write(_u32(len(name_b)))
            f.write(name_b)
            f.write(_u32(len(dtype_b)))
            f.write(dtype_b)
            f.write(struct.pack("<Q", len(blob)))
        for name in tensors:
            f.write(packed[name][1])


def load(path: str | Path) -> tuple[FileHeader, dict[str, MfqTensor]]:
    """Read a TQ model container and return its header and tensors."""

    with open_mmap(path) as store:
        return store.header, {name: store[name] for name in store}


class MMapTensorStore(Mapping[str, MfqTensor]):
    """Lazy mmap-backed MFQ tensor store.

    The file header and tensor table are parsed eagerly. Tensor blobs are decoded
    only when ``__getitem__`` is called. By default decoded tensors are not cached,
    so model construction can load one tensor, move it to GPU, and let CPU memory
    be reclaimed before the next tensor.
    """

    def __init__(
        self,
        path: str | Path,
        header: FileHeader,
        records: dict[str, MMapTensorRecord],
        file_obj,
        mm: mmap.mmap,
        *,
        cache: bool = False,
        paths: list[Path] | None = None,
        file_objs: list[object] | None = None,
        mmaps: list[mmap.mmap] | None = None,
    ) -> None:
        self.path = Path(path)
        self.header = header
        self.records = records
        self.paths = list(paths or [self.path])
        self._files = list(file_objs or [file_obj])
        self._mmaps = list(mmaps or [mm])
        # Kept for single-file callers; shard-aware code uses the helpers below.
        self._file = self._files[0]
        self._mmap = self._mmaps[0]
        self._cache_enabled = cache
        self._cache: dict[str, MfqTensor] = {}

    def mmap_for(self, record: MMapTensorRecord | str) -> mmap.mmap:
        rec = self.records[record] if isinstance(record, str) else record
        return self._mmaps[rec.source_index]

    def file_for(self, record: MMapTensorRecord | str):
        rec = self.records[record] if isinstance(record, str) else record
        return self._files[rec.source_index]

    def blob_view(self, record: MMapTensorRecord | str) -> memoryview:
        rec = self.records[record] if isinstance(record, str) else record
        return memoryview(self.mmap_for(rec))[rec.offset : rec.offset + rec.nbytes]

    def evict_blob(self, record: MMapTensorRecord | str) -> None:
        """Release file-backed pages after a tensor has been copied elsewhere."""

        advice = getattr(mmap, "MADV_DONTNEED", None)
        rec = self.records[record] if isinstance(record, str) else record
        mapping = self.mmap_for(rec)
        if advice is None or not hasattr(mapping, "madvise"):
            return
        try:
            page = int(getattr(mmap, "PAGESIZE", os.sysconf("SC_PAGE_SIZE")))
            start = int(rec.offset) - int(rec.offset) % page
            end = min(
                len(mapping),
                ((int(rec.offset) + int(rec.nbytes) + page - 1) // page) * page,
            )
            mapping.madvise(advice, start, end - start)
        except (OSError, TypeError, ValueError):
            # Page residency is an optimization; the tensor copy remains valid.
            pass

    def __getitem__(self, name: str) -> MfqTensor:
        if self._cache_enabled and name in self._cache:
            return self._cache[name]
        rec = self.records[name]
        blob = self.blob_view(rec)
        try:
            tensor = _unpack_tensor(rec.dtype, blob)
        finally:
            blob.release()
        self.evict_blob(rec)
        if self._cache_enabled:
            self._cache[name] = tensor
        return tensor

    def __iter__(self) -> Iterator[str]:
        return iter(self.records)

    def __len__(self) -> int:
        return len(self.records)

    def read_blob(self, name: str) -> bytes:
        """Copy one tensor's original packed payload without decoding it."""

        rec = self.records[name]
        payload = bytes(
            self.mmap_for(rec)[rec.offset : rec.offset + rec.nbytes]
        )
        self.evict_blob(rec)
        return payload

    def embedding_reader(self, name: str) -> MMapEmbeddingReader:
        """Open a row-selective reader without copying the tensor payload."""

        return MMapEmbeddingReader(self, name)

    def close(self) -> None:
        self._cache.clear()
        for mm in self._mmaps:
            mm.close()
        for file_obj in self._files:
            file_obj.close()

    def __enter__(self) -> MMapTensorStore:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class MMapEmbeddingReader:
    """Read selected rows from a native floating-point mmap embedding.

    The reader retains only tensor geometry and byte offsets.  Calls to
    :meth:`read_rows` fault and copy the requested rows, leaving the complete
    embedding payload file-backed and reclaimable by the operating system.
    This is the MFQ equivalent of oMLX's disk-backed PLE embedding path, but it
    is deliberately format-generic rather than tied to a model architecture.
    """

    _DENSE_ROW_DTYPES = {
        "BF16": np.dtype("<u2"),
        "F8_E4M3": np.dtype("u1"),
        "F16": np.dtype("<f2"),
        "F32": np.dtype("<f4"),
    }

    def __init__(self, store: object, name: str) -> None:
        records = getattr(store, "records", None)
        mmap_for = getattr(store, "mmap_for", None)
        if not isinstance(records, Mapping) or not callable(mmap_for):
            raise TypeError("row-selective embeddings require an mmap-backed tensor store")
        if name not in records:
            raise KeyError(f"tensor {name!r} is not present in the MFQ model")

        record = records[name]
        self.name = name
        self.dtype = str(record.dtype)
        self._stored_dtype = str(record.stored_dtype or record.dtype)
        self._mapping = mmap_for(record)
        self._blob_start = int(record.offset)
        self._blob_end = self._blob_start + int(record.nbytes)
        self.mapped_nbytes = int(record.nbytes)
        self.last_rows_read = 0
        self.total_rows_read = 0
        self.last_logical_bytes = 0
        self.total_logical_bytes = 0

        if self.dtype in self._DENSE_ROW_DTYPES:
            self._init_dense()
        else:
            raise TypeError(
                f"mmap row lookup does not support MFQ dtype {self.dtype!r}"
            )
        self._advise_random()

    def _require(self, offset: int, nbytes: int, what: str) -> None:
        if offset < self._blob_start or nbytes < 0 or offset + nbytes > self._blob_end:
            raise ValueError(f"truncated {what} for mmap embedding {self.name!r}")

    def _init_dense(self) -> None:
        offset = self._blob_start
        self._require(offset, 4, "dense shape header")
        ndim = int(struct.unpack_from("<I", self._mapping, offset)[0])
        offset += 4
        shape_nbytes = ndim * 8
        self._require(offset, shape_nbytes, "dense shape")
        shape = tuple(
            int(value)
            for value in struct.unpack_from(f"<{ndim}q", self._mapping, offset)
        )
        offset += shape_nbytes
        if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
            raise ValueError(
                f"mmap embedding {self.name!r} must be a non-empty matrix, got {shape}"
            )
        item_size = self._DENSE_ROW_DTYPES[self.dtype].itemsize
        payload_nbytes = math.prod(shape) * item_size
        self._require(offset, payload_nbytes, "dense payload")
        if offset + payload_nbytes != self._blob_end:
            raise ValueError(f"invalid trailing bytes in dense tensor {self.name!r}")
        self.shape = shape
        self.out, self.neuron_len = shape
        self._dense_offset = offset
        self._dense_row_nbytes = self.neuron_len * item_size
        self._storage_kind = "dense"


    def _advise_random(self) -> None:
        advice = getattr(mmap, "MADV_RANDOM", None)
        if advice is None or not hasattr(self._mapping, "madvise"):
            return
        try:
            page = int(getattr(mmap, "PAGESIZE", os.sysconf("SC_PAGE_SIZE")))
            start = self._blob_start - self._blob_start % page
            end = min(
                len(self._mapping),
                ((self._blob_end + page - 1) // page) * page,
            )
            self._mapping.madvise(advice, start, end - start)
        except (OSError, TypeError, ValueError):
            # Range advice is an optimization. Older Darwin/Python pairs can
            # reject it even though the mapping itself is perfectly usable.
            pass

    @staticmethod
    def _normalized_rows(rows: np.ndarray, out: int) -> tuple[np.ndarray, np.ndarray]:
        requested = np.asarray(rows)
        if not np.issubdtype(requested.dtype, np.integer):
            raise TypeError("embedding row indices must be integers")
        flat = np.ascontiguousarray(requested, dtype=np.int64).reshape(-1)
        if np.any(flat < 0) or np.any(flat >= out):
            raise IndexError("embedding row index is outside the stored matrix")
        if flat.size == 0:
            return flat, flat
        unique, inverse = np.unique(flat, return_inverse=True)
        return unique, inverse


    def _read_dense(self, unique: np.ndarray) -> np.ndarray:
        storage_dtype = self._DENSE_ROW_DTYPES[self.dtype]
        table = np.ndarray(
            self.shape,
            dtype=storage_dtype,
            buffer=self._mapping,
            offset=self._dense_offset,
        )
        copied = np.array(table[unique], copy=True)
        if self.dtype == "BF16":
            return (copied.astype(np.uint32) << np.uint32(16)).view(np.float32)
        if self.dtype == "F8_E4M3":
            raw = copied.astype(np.uint8, copy=False)
            sign = np.where(raw & 0x80, -1.0, 1.0).astype(np.float32)
            exponent = ((raw >> 3) & 0x0F).astype(np.int16)
            mantissa = (raw & 0x07).astype(np.float32)
            subnormal = np.ldexp(mantissa, -9)
            normal = np.ldexp(1.0 + mantissa / 8.0, exponent - 7)
            decoded = sign * np.where(exponent == 0, subnormal, normal)
            decoded[(raw & 0x7F) == 0x7F] = np.nan
            return decoded
        return copied


    def read_rows(self, rows: np.ndarray) -> np.ndarray:
        """Return requested rows while touching only their mmap-backed pages."""

        unique, inverse = self._normalized_rows(rows, self.out)
        if unique.size == 0:
            return np.empty((0, self.neuron_len), dtype=np.float32)
        selected = self._read_dense(unique)
        logical_bytes = int(unique.size) * self._dense_row_nbytes
        self.last_rows_read = int(unique.size)
        self.total_rows_read += self.last_rows_read
        self.last_logical_bytes = logical_bytes
        self.total_logical_bytes += logical_bytes
        return np.ascontiguousarray(selected[inverse])


def _open_single_mmap(
    path: str | Path,
    *,
    source_index: int = 0,
) -> tuple[FileHeader, dict[str, MMapTensorRecord], object, mmap.mmap]:
    p = Path(path)
    f = p.open("rb")
    mm = None
    try:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        if mm[:4] != TYLOINFER_MAGIC:
            raise ValueError(f"非 MFQ 文件（魔数不匹配）: {mm[:4]!r}")
        off = 4
        version, off = _read_u32(mm, off)
        arch, off = _read_str(mm, off)
        extra: dict[str, str] = {}
        if version >= 2:
            n_extra, off = _read_u32(mm, off)
            for _ in range(n_extra):
                k, off = _read_str(mm, off)
                v_json, off = _read_str(mm, off)
                extra[k] = json.loads(v_json)
        manifest = extra.get("runtime_assets")
        if isinstance(manifest, dict) and isinstance(manifest.get("assets"), dict):
            for asset in manifest["assets"].values():
                if isinstance(asset, dict) and isinstance(asset.get("record"), str):
                    asset["record"] = canonical_asset_record_name(asset["record"])
        num_tensors, off = _read_u32(mm, off)

        raw_records: list[tuple[str, str, int]] = []
        for _ in range(num_tensors):
            name, off = _read_str(mm, off)
            name = canonical_asset_record_name(name)
            dtype, off = _read_str(mm, off)
            blob_nbytes = struct.unpack_from("<Q", mm, off)[0]
            off += 8
            raw_records.append((name, dtype, blob_nbytes))

        records: dict[str, MMapTensorRecord] = {}
        blob_off = off
        for name, stored_dtype, blob_nbytes in raw_records:
            if name in records:
                raise ValueError(f"duplicate MFQ tensor record: {name}: {p}")
            if canonical_dtype(stored_dtype) not in TENSOR_DTYPES | LEGACY_DTYPES:
                raise ValueError(f'TyloInfer does not support tensor format {stored_dtype!r}: {name}')
            records[name] = MMapTensorRecord(
                name=name,
                dtype=canonical_dtype(stored_dtype),
                offset=blob_off,
                nbytes=blob_nbytes,
                source_index=source_index,
                stored_dtype=stored_dtype,
            )
            blob_off += blob_nbytes
        if blob_off != mm.size():
            raise ValueError(f"MFQ 文件长度不匹配: records end={blob_off}, file size={mm.size()}")
        header = FileHeader(version=version, model_arch=arch, num_tensors=num_tensors, extra=extra)
        return header, records, f, mm
    except Exception:
        if mm is not None:
            mm.close()
        f.close()
        raise


def open_mmap(path: str | Path, *, cache: bool = False) -> MMapTensorStore:
    """Open a single-file or sharded MFQ as one lazy tensor mapping."""

    from tyloinfer.formats.assets import is_asset_record
    from tyloinfer.formats.shards import (
        parse_shard_path,
        shard_paths_from_any,
        split_values,
    )

    requested = Path(path)
    first_header, first_records, first_file, first_mmap = _open_single_mmap(requested)
    opened_files = [first_file]
    opened_mmaps = [first_mmap]
    try:
        split_no, split_count, expected_tensors, expected_records = split_values(
            first_header.extra
        )
        parsed = parse_shard_path(requested)
        if split_count == 1:
            if parsed is not None and parsed[2] != 1:
                raise ValueError(
                    f"MFQ filename claims multiple shards but metadata does not: {requested}"
                )
            return MMapTensorStore(
                requested,
                first_header,
                first_records,
                first_file,
                first_mmap,
                cache=cache,
            )
        if parsed is None and (split_no != 0 or requested.suffix.lower() not in {".tq", ".mfq"}):
            raise ValueError(
                f"sharded MFQ path lacks -00001-of-00000 suffix: {requested}"
            )
        if parsed is not None and parsed[1] - 1 != split_no:
            raise ValueError(
                f"MFQ shard index mismatch in filename/metadata: "
                f"{parsed[1] - 1} != {split_no}: {requested}"
            )
        if split_no:
            first_records = {
                name: MMapTensorRecord(
                    name=record.name,
                    dtype=record.dtype,
                    offset=record.offset,
                    nbytes=record.nbytes,
                    source_index=split_no,
                    stored_dtype=record.stored_dtype,
                )
                for name, record in first_records.items()
            }
        paths = shard_paths_from_any(requested, split_count)
        missing = [value for value in paths if not value.is_file()]
        if missing:
            raise FileNotFoundError(f"missing MFQ shard: {missing[0]}")

        headers: list[FileHeader | None] = [None] * split_count
        shard_records: list[dict[str, MMapTensorRecord] | None] = [None] * split_count
        files: list[object | None] = [None] * split_count
        mmaps: list[mmap.mmap | None] = [None] * split_count
        headers[split_no] = first_header
        shard_records[split_no] = first_records
        files[split_no] = first_file
        mmaps[split_no] = first_mmap

        for index, shard_path in enumerate(paths):
            if index == split_no:
                continue
            header, records, file_obj, mm = _open_single_mmap(
                shard_path, source_index=index
            )
            opened_files.append(file_obj)
            opened_mmaps.append(mm)
            headers[index] = header
            shard_records[index] = records
            files[index] = file_obj
            mmaps[index] = mm

        combined: dict[str, MMapTensorRecord] = {}
        total_tensors = 0
        reference_version = first_header.version
        reference_arch = first_header.model_arch
        for index in range(split_count):
            header = headers[index]
            records = shard_records[index]
            assert header is not None and records is not None
            no, count, tensors, record_count = split_values(header.extra)
            if no != index or count != split_count:
                raise ValueError(
                    f"MFQ shard metadata mismatch: expected {index}/{split_count}, "
                    f"got {no}/{count}: {paths[index]}"
                )
            if header.version != reference_version or header.model_arch != reference_arch:
                raise ValueError(f"MFQ shard architecture/version mismatch: {paths[index]}")
            if is_engram_shard(paths[index].as_posix()) and (
                not records or any(not is_engram_table(name) for name in records)
            ):
                raise ValueError(f"shared Engram shard contains non-table records: {paths[index]}")
            if expected_tensors is None:
                expected_tensors = tensors
            elif tensors is not None and tensors != expected_tensors:
                raise ValueError(f"MFQ shard tensor count mismatch: {paths[index]}")
            if expected_records is None:
                expected_records = record_count
            elif record_count is not None and record_count != expected_records:
                raise ValueError(f"MFQ shard record count mismatch: {paths[index]}")
            for name, record in records.items():
                if name in combined:
                    raise ValueError(f"duplicate MFQ tensor across shards: {name}")
                combined[name] = record
                total_tensors += not is_asset_record(name)

        if expected_records is not None and len(combined) != expected_records:
            raise ValueError(
                f"MFQ shard record total mismatch: {len(combined)} != {expected_records}"
            )
        if expected_tensors is not None and total_tensors != expected_tensors:
            raise ValueError(
                f"MFQ shard tensor total mismatch: {total_tensors} != {expected_tensors}"
            )
        primary = headers[0]
        assert primary is not None
        primary.num_tensors = len(combined)
        return MMapTensorStore(
            paths[0],
            primary,
            combined,
            files[0],
            mmaps[0],
            cache=cache,
            paths=paths,
            file_objs=[value for value in files if value is not None],
            mmaps=[value for value in mmaps if value is not None],
        )
    except Exception:
        for mm in opened_mmaps:
            if not mm.closed:
                mm.close()
        for file_obj in opened_files:
            if not file_obj.closed:
                file_obj.close()
        raise


def load_mmap(path: str | Path, *, cache: bool = False) -> tuple[FileHeader, MMapTensorStore]:
    """Return ``(header, lazy mmap tensor store)`` without reading all blobs."""

    store = open_mmap(path, cache=cache)
    return store.header, store
