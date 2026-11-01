from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def stage(destination: Path) -> None:
    source = Path(__file__).resolve().parents[1] / 'agent-runtime'
    node = shutil.which('node')
    npm = shutil.which('npm')
    if not node or not npm:
        raise SystemExit('Node.js 24.12+ and npm are required to package DeepSeek Harness.')
    version = subprocess.check_output([node, '-p', 'process.versions.node'], text=True).strip()
    if tuple(int(part) for part in version.split('.')[:2]) < (24, 12):
        raise SystemExit('DeepSeek Harness requires Node.js 24.12 or later.')
    destination.mkdir(parents=True, exist_ok=True)
    for name in ('package.json', 'package-lock.json'):
        shutil.copy2(source / name, destination / name)
    npm_cli = Path(npm).parent / 'node_modules/npm/bin/npm-cli.js'
    installer = [node, str(npm_cli)] if npm_cli.is_file() else [npm]
    subprocess.run([*installer, 'ci', '--omit=dev', '--no-audit', '--no-fund'], cwd=destination, check=True)
    executable = destination / Path(node).name
    shutil.copy2(node, executable)
    executable.chmod(0o755)
    if sys.platform == 'darwin':
        identity = os.environ.get('TYLOINFER_RELEASE_SIGNING_IDENTITY') or os.environ.get('APPLE_SIGNING_IDENTITY') or '-'
        magic = {bytes.fromhex(value) for value in ('feedface', 'feedfacf', 'cefaedfe', 'cffaedfe', 'cafebabe', 'bebafeca')}
        for path in destination.rglob('*'):
            if not path.is_file() or path.is_symlink():
                continue
            with path.open('rb') as stream:
                native = stream.read(4) in magic
            if native:
                arguments = ['codesign', '--force', '--sign', identity]
                if path == executable:
                    arguments += ['--options', 'runtime', '--entitlements', str(Path(__file__).with_name('agent-node-entitlements.plist'))]
                subprocess.run([*arguments, str(path)], check=True)
    cli = destination / 'node_modules/@deepseek-ai/dsh/lib/bin.js'
    subprocess.run([str(executable.resolve()), str(cli.resolve()), '--version'], check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('destination', type=Path)
    stage(parser.parse_args().destination)
