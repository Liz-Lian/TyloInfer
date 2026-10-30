from __future__ import annotations

import re
import struct
from bisect import bisect_right
from contextlib import contextmanager
from pathlib import Path

_SHARD = re.compile(r"^(?P<name>.+)-(?P<index>\d{5})-of-(?P<count>\d{5})\.gguf$", re.IGNORECASE)
_RAW_PART = re.compile(r"^(?P<name>.+\.gguf)\.part-(?P<index>[0-9]{1,5})-of-(?P<count>[0-9]{1,5})$")
_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def is_gguf_path(path: str | Path) -> bool:
    path = Path(path)
    return path.suffix.casefold() == '.gguf' or _RAW_PART.fullmatch(path.name) is not None


def gguf_part_paths(path: str | Path, *, require_complete: bool = True) -> tuple[Path, ...]:
    path = Path(path)
    match = _RAW_PART.fullmatch(path.name)
    if match is None:
        return (path,)
    index, count = int(match['index']), int(match['count'])
    if not 1 <= index <= count:
        raise ValueError('invalid GGUF part number')
    width = len(match['index'])
    if width > 1 and not match['index'].startswith('0'):
        padded_first = path.with_name(f"{match['name']}.part-{1:0{width}d}-of-{match['count']}")
        plain_first = path.with_name(f"{match['name']}.part-1-of-{match['count']}")
        if not padded_first.is_file() and plain_first.is_file():
            width = 1
    paths = tuple(path.with_name(f"{match['name']}.part-{number:0{width}d}-of-{match['count']}")
        for number in range(1, count + 1))
    if require_complete:
        for part in paths:
            if not part.is_file():
                raise FileNotFoundError(f'missing GGUF part: {part.name}')
    return paths


class _RawPartsReader:
    def __init__(self, paths: tuple[Path, ...]):
        self.paths, self.starts, self.size = paths, [], 0
        for path in paths:
            self.starts.append(self.size)
            size = path.stat().st_size
            if not size:
                raise ValueError('empty GGUF part')
            self.size += size
        self.position, self.part, self.stream = 0, -1, None

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        if whence not in (0, 1, 2):
            raise ValueError('invalid GGUF seek mode')
        position = offset + (0 if whence == 0 else self.position if whence == 1 else self.size)
        if position < 0:
            raise ValueError('invalid GGUF seek offset')
        self.position = position
        return position

    def read(self, count):
        if count < 0:
            raise ValueError('GGUF metadata reads must be bounded')
        result = bytearray()
        while count and self.position < self.size:
            part = bisect_right(self.starts, self.position) - 1
            if part != self.part:
                self.close()
                self.stream, self.part = self.paths[part].open('rb'), part
            self.stream.seek(self.position - self.starts[part])
            end = self.starts[part + 1] if part + 1 < len(self.paths) else self.size
            size = min(count, end - self.position)
            value = self.stream.read(size)
            if len(value) != size:
                raise ValueError('truncated GGUF part')
            result.extend(value)
            self.position += size
            count -= size
        return bytes(result)

    def close(self):
        if self.stream is not None:
            self.stream.close()
        self.stream, self.part = None, -1


@contextmanager
def _metadata_stream(path: Path):
    paths = gguf_part_paths(path)
    if len(paths) == 1:
        with paths[0].open('rb') as stream:
            yield stream, paths[0].stat().st_size
    else:
        stream = _RawPartsReader(paths)
        try:
            yield stream, stream.size
        finally:
            stream.close()


def gguf_vocabulary_decoder(path: Path):
    from tokenizers import Tokenizer, decoders, models

    metadata = {}
    with _metadata_stream(path) as (stream, size):
        reader = _MetadataReader(stream, size)
        for _ in range(reader.fields):
            name, kind = reader.string(keep=True), reader.integer('<I')
            keep = name in {b'tokenizer.ggml.model', b'tokenizer.ggml.tokens'}
            value = reader.value(kind, keep=keep)
            if keep:
                if name in metadata:
                    raise ValueError('duplicate GGUF tokenizer metadata')
                metadata[name] = value
    if metadata.get(b'tokenizer.ggml.model') != b'gpt2':
        raise ValueError('This GGUF tokenizer does not support cache text decoding')
    raw = metadata.get(b'tokenizer.ggml.tokens')
    if not isinstance(raw, list) or not raw:
        raise ValueError('GGUF tokenizer vocabulary is missing')
    pieces = [piece.decode('utf-8') for piece in raw]
    vocabulary = {piece: index for index, piece in enumerate(pieces)}
    if len(vocabulary) != len(pieces):
        raise ValueError('GGUF tokenizer contains duplicate vocabulary entries')
    tokenizer = Tokenizer(models.BPE(vocabulary, []))
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


def parse_gguf_shard_path(path: str | Path) -> tuple[Path, int, int] | None:
    path = Path(path)
    raw = _RAW_PART.fullmatch(path.name)
    match = raw or _SHARD.fullmatch(path.name)
    if match is None:
        return None
    index, count = int(match['index']), int(match['count'])
    if not 1 <= index <= count:
        raise ValueError('invalid GGUF shard number')
    return path.with_name(match['name'] if raw else match['name'] + '.gguf'), index, count


def matching_gguf_shard_paths(base: str | Path) -> list[Path]:
    base = Path(base)
    if _RAW_PART.fullmatch(base.name):
        return [path for path in gguf_part_paths(base, require_complete=False) if path.is_file()]
    parsed = parse_gguf_shard_path(base)
    source = base
    if parsed is not None:
        base = parsed[0]
    result = []
    raw_result = []
    for path in base.parent.iterdir():
        if not path.is_file():
            continue
        try:
            parsed = parse_gguf_shard_path(path)
        except ValueError:
            continue
        if parsed is not None and parsed[0] == base:
            (raw_result if _RAW_PART.fullmatch(path.name) else result).append(path)
    return sorted(result if result or _SHARD.fullmatch(source.name) else raw_result)


class _MetadataReader:
    def __init__(self, stream, size: int):
        self.stream, self.size = stream, size
        if self.read(4) != b'GGUF' or self.integer('<I') != 3:
            raise ValueError('unsupported GGUF header')
        self.tensors, self.fields = self.integer('<Q'), self.integer('<Q')
        if self.fields > 1 << 20:
            raise ValueError('too many GGUF metadata fields')

    def read(self, count: int) -> bytes:
        value = self.stream.read(count)
        if len(value) != count:
            raise ValueError('truncated GGUF metadata')
        return value

    def integer(self, fmt: str) -> int:
        return struct.unpack(fmt, self.read(struct.calcsize(fmt)))[0]

    def string(self, *, keep: bool = False) -> bytes:
        count = self.integer('<Q')
        if count > self.size - self.stream.tell() or (keep and count > 1 << 20):
            raise ValueError('invalid GGUF string extent')
        if keep:
            return self.read(count)
        self.stream.seek(count, 1)
        return b''

    def value(self, kind: int, *, keep: bool = False):
        if kind == 8:
            return self.string(keep=keep)
        if kind == 9:
            subtype, count = self.integer('<I'), self.integer('<Q')
            if subtype == 8:
                if count > (self.size - self.stream.tell()) // 8 or (keep and count > 1 << 22):
                    raise ValueError('invalid GGUF string array extent')
                result = [] if keep else None
                for _ in range(count):
                    value = self.string(keep=keep)
                    if keep:
                        result.append(value)
                return result
            width = _SIZES.get(subtype)
            if width is None or count > (self.size - self.stream.tell()) // width:
                raise ValueError('invalid GGUF array extent')
            self.stream.seek(count * width, 1)
            return None
        width = _SIZES.get(kind)
        if width is None or width > self.size - self.stream.tell():
            raise ValueError('invalid GGUF metadata type')
        self.stream.seek(width, 1)
        return None


def gguf_is_table_component(path: str | Path) -> bool:
    path = Path(path)
    with _metadata_stream(path) as (stream, size):
        reader = _MetadataReader(stream, size)
        if reader.tensors > 1 << 20:
            raise ValueError('too many GGUF tensors')
        if reader.tensors == 0:
            return True
        for _ in range(reader.fields):
            reader.string()
            reader.value(reader.integer('<I'))
        for _ in range(reader.tensors):
            name = reader.string(keep=True)
            if name != b'per_layer_token_embd.weight' and not re.fullmatch(rb'blk\.\d+\.engram_embd\.weight', name):
                return False
            rank = reader.integer('<I')
            if not 1 <= rank <= 4:
                raise ValueError('invalid GGUF tensor rank')
            reader.read(rank * 8 + 12)
        return True
