"""Toy tokenizer for native worker integration checks."""
import json
import pytest

def _worker_tokenizer(tmp_path, *, cacheable_tokens=False):
    from tyloinfer.server.runtime.hf_tokenizer import ensure_hf_tokenizer_gguf
    from tests.test_hf_native_models import _hf_fixture
    tokenizer_source = tmp_path / "tokenizer-source"
    _hf_fixture(tokenizer_source)
    config = json.loads((tokenizer_source / "config.json").read_text())
    config["text_config"]["vocab_size"] = 32
    (tokenizer_source / "config.json").write_text(json.dumps(config))
    vocabulary = json.loads((tokenizer_source / "tokenizer.json").read_text())
    vocabulary["model"]["vocab"].update({f"token{index}": index for index in range(5, 32)})
    if cacheable_tokens:
        vocabulary["added_tokens"].extend({"id": index, "content": f"token{index}", "special": False,
            "single_word": False, "lstrip": False, "rstrip": False, "normalized": False}
            for index in range(5, 32))
    (tokenizer_source / "tokenizer.json").write_text(json.dumps(vocabulary))
    return ensure_hf_tokenizer_gguf(tokenizer_source, tmp_path / "tokenizer-cache")
