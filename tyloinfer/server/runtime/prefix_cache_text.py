from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=4)
def _tokenizer(path: str, modified: int | tuple[tuple[int, int], ...], size: int):
    from tokenizers import Tokenizer
    from tyloinfer.formats.assets import HF_TOKENIZER_JSON_ASSET
    from tyloinfer.formats.io import open_mmap

    source = Path(path)
    try:
        if source.is_dir():
            return Tokenizer.from_file(str(source / 'tokenizer.json'))
        from tyloinfer.formats.gguf import gguf_vocabulary_decoder, is_gguf_path
        if is_gguf_path(source):
            return gguf_vocabulary_decoder(source)
        with open_mmap(source) as store:
            return Tokenizer.from_str(store.read_blob(HF_TOKENIZER_JSON_ASSET).decode('utf-8'))
    except Exception as error:
        raise ValueError('cache tokenizer is unavailable') from error


def decode_prefix_cache_tokens(path: Path, tokens: list[int]) -> str:
    source = path / 'tokenizer.json' if path.is_dir() else path
    status = source.stat()
    from tyloinfer.formats.gguf import gguf_part_paths, is_gguf_path
    parts = gguf_part_paths(path) if is_gguf_path(path) else (source,)
    signature = tuple((info.st_mtime_ns, info.st_size) for info in (part.stat() for part in parts))
    return _tokenizer(str(path), signature if len(parts) > 1 else status.st_mtime_ns,
        sum(item[1] for item in signature)).decode(tokens, skip_special_tokens=False)
