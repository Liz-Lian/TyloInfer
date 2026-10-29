from __future__ import annotations

import re
import struct
from pathlib import Path

_SHARD = re.compile(r"^(?P<name>.+)-(?P<index>\d{5})-of-(?P<count>\d{5})\.gguf$", re.IGNORECASE)
_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def gguf_vocabulary_decoder(path: Path):
    from tokenizers import Tokenizer, decoders, models

    metadata = {}
    with path.open('rb') as stream:
        reader = _MetadataReader(stream, path.stat().st_size)
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
    match = _SHARD.fullmatch(path.name)
    if match is None:
        return None
    index, count = int(match['index']), int(match['count'])
    if not 1 <= index <= count:
        raise ValueError('invalid GGUF shard number')
    return path.with_name(match['name'] + '.gguf'), index, count


def matching_gguf_shard_paths(base: str | Path) -> list[Path]:
    base = Path(base)
    result = []
    for path in base.parent.iterdir():
        if not path.is_file():
            continue
        try:
            parsed = parse_gguf_shard_path(path)
        except ValueError:
            continue
        if parsed is not None and parsed[0] == base:
            result.append(path)
    return sorted(result)


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
    with path.open('rb') as stream:
        reader = _MetadataReader(stream, path.stat().st_size)
        if reader.tensors > 1:
            return False
        if reader.tensors == 0:
            return True
        for _ in range(reader.fields):
            reader.string()
            reader.value(reader.integer('<I'))
        return reader.string(keep=True) in {b'per_layer_token_embd.weight'}
