"""Package the Windows release as a standalone executable or ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import zipfile


def payload(project: Path, executable: Path) -> dict[str, Path]:
    """Validate and enumerate the native release resources before writing output."""
    tauri = project / "TyloInferStudio/src-tauri"
    bundle = json.loads((tauri / "tauri.release-windows.conf.json").read_text())["bundle"]
    files: dict[str, Path] = {}
    names: set[str] = set()
    if bundle.get("externalBin"):
        raise ValueError("Native releases must not include legacy externalBin sidecars")

    def add(source: Path, destination: str) -> None:
        path = PurePosixPath(destination.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts or ":" in str(path):
            raise ValueError(f"Unsafe archive path: {destination}")
        name = str(path)
        if name.casefold() in names:
            raise ValueError(f"Duplicate archive path: {name}")
        if not source.is_file():
            raise FileNotFoundError(source)
        if any(part.casefold() in {"tyloinfer-cli", "agent-runtime", "_internal"} for part in path.parts) or (
            path.name.casefold().startswith(("python", "libpython", "mfq-runtime", "mfq-diagnostics"))
            or path.suffix.casefold() in {".py", ".pyc", ".pyd"}
        ):
            raise ValueError(f"Forbidden legacy product runtime: {name}")
        files[name] = source
        names.add(name.casefold())

    add(executable, "tyloinfer.exe")
    for source, destination in bundle["resources"].items():
        root = (tauri / source).resolve()
        if not root.is_relative_to(project.resolve()):
            raise ValueError(f"Resource source escapes project: {source}")
        if root.is_dir():
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    if not path.resolve().is_relative_to(root):
                        raise ValueError(f"Linked resource escapes its source: {path}")
                    add(path, str(PurePosixPath(destination) / path.relative_to(root).as_posix()))
        else:
            add(root, destination)
    for required in ("tyloinfer-server/tyloinfer-server.exe", "web-ui/index.html"):
        if required not in files:
            raise FileNotFoundError(f"Missing portable resource: {required}")
    return files


def build_archive(project: Path, executable: Path, output: Path) -> Path:
    """Write a validated directory ZIP atomically and produce its SHA-256 checksum."""
    files = payload(project, executable)
    config = json.loads((project / "TyloInferStudio/src-tauri/tauri.conf.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    archive = output / f"TyloInfer_{config['version']}_x64-portable.zip"
    partial = archive.with_suffix(".zip.partial")
    total = sum(path.stat().st_size for path in files.values())
    completed = 0
    next_progress = 256 * 1024 * 1024
    print(f"Packing {len(files)} files ({total / 1024**3:.2f} GiB)", flush=True)
    try:
        with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as package:
            for name, path in sorted(files.items()):
                package.write(path, "TyloInfer/" + name)
                completed += path.stat().st_size
                if completed >= next_progress:
                    print(f"Packed {completed / 1024**3:.2f} / {total / 1024**3:.2f} GiB", flush=True)
                    next_progress = completed + 256 * 1024 * 1024
        partial.replace(archive)
    finally:
        partial.unlink(missing_ok=True)
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    archive.with_suffix(".zip.sha256").write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    print(f"Built {archive} ({archive.stat().st_size / 1024**2:.1f} MiB)", flush=True)
    return archive


def nsis_string(value: str) -> str:
    """Escape literal payload paths for the generated NSIS script."""
    return value.replace("$", "$$").replace('"', '$\\"').replace("\r", "$\\r").replace("\n", "$\\n")


def build_executable(project: Path, executable: Path, output: Path, makensis: Path) -> Path:
    """Build the cached standalone launcher and atomically publish its executable and ZIP."""
    files = payload(project, executable)
    version = json.loads((project / "TyloInferStudio/src-tauri/tauri.conf.json").read_text())["version"]
    digest = hashlib.sha256()
    instructions = []
    checks = []
    directory = None
    for name, path in sorted(files.items()):
        digest.update(name.encode() + b"\0")
        with path.open("rb") as stream:
            digest.update(hashlib.file_digest(stream, "sha256").digest())
        target = PurePosixPath(name)
        cache_name = name.replace("/", "\\")
        checks.append(f'  IfFileExists "$CachePath\\{nsis_string(cache_name)}" 0 extract')
        parent = str(target.parent).replace("/", "\\")
        if parent != directory:
            instructions.append(f'  SetOutPath "$Stage\\{nsis_string(parent)}"')
            directory = parent
        instructions += [f'  File "/oname={nsis_string(target.name)}" "{nsis_string(str(path.resolve()))}"',
                         '  IfErrors failed']
    output.mkdir(parents=True, exist_ok=True)
    destination = output / f"TyloInfer_{version}_x64.exe"
    partial = destination.with_suffix(".exe.partial")
    script = project / "packaging/build/standalone/launcher.nsi"
    script.parent.mkdir(parents=True, exist_ok=True)
    template = (project / "packaging/windows_standalone.nsi").read_text(encoding="utf-8")
    for key, value in {
        "OUTPUT": nsis_string(str(partial.resolve())), "VERSION": version,
        "ICON": nsis_string(str(project / "TyloInferStudio/src-tauri/icons/icon.ico")),
        "PAYLOAD_ID": f"{version}-{digest.hexdigest()[:16]}",
        "PAYLOAD_FILES": "\n".join(instructions),
        "PAYLOAD_CHECKS": "\n".join(checks),
    }.items():
        template = template.replace(f"@{key}@", value)
    script.write_text(template, encoding="utf-8")
    print(f"Packing {len(files)} files into a standalone executable", flush=True)
    try:
        subprocess.run([str(makensis), "-INPUTCHARSET", "UTF8", "-V2", str(script)], check=True)
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)
    with destination.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    destination.with_suffix(".exe.sha256").write_text(f"{checksum}  {destination.name}\n", encoding="utf-8")
    print(f"Built {destination} ({destination.stat().st_size / 1024**2:.1f} MiB)", flush=True)
    archive = output / f"TyloInfer_{version}_x64-portable.zip"
    zip_partial = archive.with_suffix(".zip.partial")
    try:
        with zipfile.ZipFile(zip_partial, "w", compression=zipfile.ZIP_STORED) as package:
            package.write(destination, "TyloInfer.exe")
        zip_partial.replace(archive)
    finally:
        zip_partial.unlink(missing_ok=True)
    with archive.open("rb") as stream:
        zip_checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    archive.with_suffix(".zip.sha256").write_text(f"{zip_checksum}  {archive.name}\n", encoding="utf-8")
    print(f"Built {archive} (one executable)", flush=True)
    return destination


def main() -> None:
    """Validate resources or build the selected portable format using build-only Python."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--studio-executable", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--makensis", type=Path, help="Build a single-file app using this NSIS compiler")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    if args.validate_only:
        files = payload(project, args.studio_executable)
        print(f"PASS: {len(files)} native release resources")
    elif not args.output_directory:
        parser.error("--output-directory is required unless --validate-only is set")
    elif args.makensis:
        build_executable(project, args.studio_executable, args.output_directory, args.makensis)
    else:
        build_archive(project, args.studio_executable, args.output_directory)


if __name__ == "__main__":
    main()
