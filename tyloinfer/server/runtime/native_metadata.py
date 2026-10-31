from __future__ import annotations

import json
import subprocess
from pathlib import Path

from tyloinfer.server.runtime.native import native_gguf_companion_arguments, native_runtime_environment


class NativeModelInspector:
    def __init__(self, executable: str | Path, backend: str) -> None:
        self.executable = Path(executable)
        self.backend = backend

    def __call__(self, model: Path, companions: list[str] | None = None) -> dict:
        if self.backend not in {'metal', 'cuda'}:
            raise ValueError('This model format is not available on the selected backend')
        command = [str(self.executable), '--model', str(model), '--inspect-model-json']
        command.extend(native_gguf_companion_arguments(model, companions or [], self.backend))
        result = subprocess.run(command,
            env=native_runtime_environment(self.executable, self.backend), capture_output=True, timeout=30, check=False)
        if result.returncode:
            message = result.stderr.decode('utf-8', errors='replace').strip().splitlines()
            raise ValueError(message[-1] if message else 'Native model metadata inspection failed')
        if len(result.stdout) > 16 << 20:
            raise ValueError('Model metadata exceeds the inspection limit')
        metadata = json.loads(result.stdout)
        if not isinstance(metadata, dict) or not isinstance(metadata.get('config'), dict) or not isinstance(metadata.get('tensors'), dict):
            raise ValueError('Native model inspector returned invalid metadata')
        return metadata
