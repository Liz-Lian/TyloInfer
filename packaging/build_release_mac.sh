#!/usr/bin/env bash
# Build the self-contained Apple Silicon TyloInfer DMG.

# Report a release failure without terminating callers that source this script.
fail() {
  printf 'error: %s\n' "$*" >&2
  return 1
}

# Add one private build prefix to native flags and the artifact scanner.
mfq_add_release_path() {
  local mfq_source="$1" mfq_destination="$2" mfq_spelling mfq_flag
  [[ -n "${mfq_source}" ]] || return 0
  [[ "${mfq_source}" == /* ]] || mfq_source="${tyloinfer_dir}/${mfq_source}"
  local mfq_resolved
  mfq_resolved="$("${mfq_venv_dir}/bin/python" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "${mfq_source}")"
  [[ "${mfq_resolved}" != / ]] || { printf 'error: refusing to remap the filesystem root\n' >&2; return 1; }
  for mfq_spelling in "${mfq_source%/}" "${mfq_resolved}"; do
    mfq_private_prefix_args+=(--private-prefix "${mfq_spelling}")
    CARGO_ENCODED_RUSTFLAGS+="${mfq_rust_separator}--remap-path-prefix=${mfq_spelling}=${mfq_destination}"
    mfq_flag="$("${mfq_venv_dir}/bin/python" -c 'import shlex, sys; print(shlex.quote(sys.argv[1]))' "-ffile-prefix-map=${mfq_spelling}=${mfq_destination}")"
    mfq_release_native_flags+=" ${mfq_flag}"
    [[ "${mfq_spelling}" != "${mfq_resolved}" ]] || break
  done
}

# Configure relocatable compiler flags while preserving caller flag boundaries.
mfq_configure_release_paths() {
  mfq_rust_separator=$'\x1f'
  if [[ -z "${CARGO_ENCODED_RUSTFLAGS+x}" ]]; then
    CARGO_ENCODED_RUSTFLAGS="$("${mfq_venv_dir}/bin/python" -c 'import os; print("\x1f".join(os.environ.get("RUSTFLAGS", "").split()), end="")')"
  fi
  mfq_private_prefix_args=()
  mfq_release_native_flags=""
  mfq_add_release_path "${HOME}" /mfq-build/home
  mfq_add_release_path "$("${mfq_venv_dir}/bin/python" -c 'import sys; print(sys.base_prefix)')" /mfq-build/python
  mfq_add_release_path "${mfq_venv_dir}" /mfq-build/venv
  mfq_add_release_path "${CARGO_HOME:-${HOME}/.cargo}" /mfq-build/cargo
  mfq_add_release_path "${RUSTUP_HOME:-${HOME}/.rustup}" /mfq-build/rustup
  mfq_add_release_path "${CARGO_TARGET_DIR:-${tyloinfer_dir}/src-tauri/target}" /mfq-build/target
  if [[ -n "${mfq_native_build_dir:-}" ]]; then
    mfq_add_release_path "${mfq_native_build_dir}" /mfq-build/native
  fi
  mfq_add_release_path "${mfq_project_dir}" /mfq-src
  CARGO_ENCODED_RUSTFLAGS="${CARGO_ENCODED_RUSTFLAGS#"${mfq_rust_separator}"}"
  OBJCXXFLAGS="${OBJCXXFLAGS:-${CXXFLAGS:-}}${mfq_release_native_flags}"
  CFLAGS="${CFLAGS:-}${mfq_release_native_flags}"
  CXXFLAGS="${CXXFLAGS:-}${mfq_release_native_flags}"
  export CARGO_ENCODED_RUSTFLAGS CFLAGS CXXFLAGS OBJCXXFLAGS
  export CC_SHELL_ESCAPED_FLAGS=1
}

# Write the PyInstaller hook into this build's generated hooks directory.
mfq_write_sysconfig_hook() {
  local mfq_hooks="$1"
  mkdir -p "${mfq_hooks}"
  cat > "${mfq_hooks}/hook-sysconfig.py" <<'TYLOINFER_SYSCONFIG_HOOK'
"""Remove build-interpreter paths from frozen Python configuration metadata."""

import sys
import sysconfig
from types import CodeType


def replace_build_prefix(value, prefix):
    """Rewrite build prefixes in nested frozen configuration constants."""
    if isinstance(value, CodeType):
        return value.replace(co_consts=tuple(replace_build_prefix(item, prefix) for item in value.co_consts))
    if isinstance(value, str):
        return value.replace(prefix, "/mfq-build/python")
    if isinstance(value, tuple):
        return tuple(replace_build_prefix(item, prefix) for item in value)
    if isinstance(value, frozenset):
        return frozenset(replace_build_prefix(item, prefix) for item in value)
    return value


def hook(hook_api):
    """Sanitize the sysconfig module before PyInstaller freezes it."""
    name = sysconfig._get_sysconfigdata_name()
    for node in hook_api.module_graph.import_hook(name, hook_api.module):
        if isinstance(node.code, CodeType):
            node.code = replace_build_prefix(node.code, sys.base_prefix)
TYLOINFER_SYSCONFIG_HOOK
}

# Scan release files and embedded Python archives for private build prefixes.
mfq_check_release_paths() {
  "${mfq_venv_dir}/bin/python" - "$@" <<'TYLOINFER_PATH_CHECKER'
# TYLOINFER_PATH_CHECKER_BEGIN
"""Reject local build paths in release files and frozen Python code."""

from __future__ import annotations

import argparse
import marshal
from pathlib import Path
from types import CodeType
import zipfile


class PathChecker:
    """Inspect release files and embedded Python metadata for private build paths."""
    def __init__(self, prefixes: list[str]):
        """Prepare normalized path spellings and counters for one release scan."""
        self.needles: set[bytes] = set()
        self.ascii_needles: set[bytes] = set()
        self.unicode_prefixes: set[str] = set()
        for prefix in prefixes:
            normalized = prefix.rstrip("/\\").replace("\\", "/")
            for spelling in (normalized, normalized.replace("/", "\\")):
                for escaped in (spelling, spelling.replace("\\", "\\\\")):
                    for encoding in ("utf-8", "utf-16le"):
                        self.needles.add(escaped.encode(encoding))
                        if escaped.isascii():
                            self.ascii_needles.add(escaped.lower().encode(encoding))
                    if not escaped.isascii():
                        self.unicode_prefixes.add(escaped.casefold())
        self.overlap = max(
            [len(needle) for needle in self.needles]
            + [4 * len(prefix) for prefix in self.unicode_prefixes], default=1,
        ) - 1
        self.failures: set[str] = set()
        self.files = 0
        self.python_modules = 0

    def check_bytes(self, data: bytes, label: str) -> None:
        """Record the artifact when its bytes contain a private build prefix."""
        if any(needle in data for needle in self.needles):
            self.failures.add(label)
            return
        if self.ascii_needles:
            lowered = data.lower()
            if any(needle in lowered for needle in self.ascii_needles):
                self.failures.add(label)
                return
        if self.unicode_prefixes:
            for encoding, offset in (("utf-8", 0), ("utf-16le", 0), ("utf-16le", 1)):
                text = data[offset:].decode(encoding, errors="replace").casefold()
                if any(prefix in text for prefix in self.unicode_prefixes):
                    self.failures.add(label)
                    return

    def check_code(self, code: CodeType, label: str) -> None:
        """Inspect marshalled code including nested filenames and constants."""
        # Marshalled data includes nested filenames, constant tuples, and shared strings.
        self.check_bytes(marshal.dumps(code), label)

    def check_file(self, path: Path) -> None:
        """Scan a file and inspect compressed Python metadata in ZIP archives."""
        self.files += 1
        label = str(path)
        previous = b""
        with path.open("rb") as stream:
            while block := stream.read(4 * 1024 * 1024):
                data = previous + block
                self.check_bytes(data, label)
                previous = data[-self.overlap:] if self.overlap else b""
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                for name in archive.namelist():
                    data = archive.read(name)
                    member = f"{label}!{name}"
                    self.check_bytes(data, member)
                    if name.endswith(".pyc"):
                        code = marshal.loads(data[16:])
                        if isinstance(code, CodeType):
                            self.check_code(code, member)

    def check_python_cli(self, path: Path) -> None:
        """Inspect every code object embedded in a frozen PyInstaller executable."""
        from PyInstaller.archive.readers import CArchiveReader

        archive = CArchiveReader(str(path))
        for name, entry in archive.toc.items():
            if entry[-1] == "z":
                pyz = archive.open_embedded_archive(name)
                for module in pyz.toc:
                    code = pyz.extract(module)
                    self.python_modules += 1
                    if isinstance(code, CodeType):
                        self.check_code(code, f"{path.name}:{module}")
                    elif isinstance(code, bytes):
                        self.check_bytes(code, f"{path.name}:{module}")
            elif entry[-1] in ("s", "m", "M"):
                data = archive.extract(name)
                code = marshal.loads(data)
                if isinstance(code, CodeType):
                    self.check_code(code, f"{path.name}:{name}")


def main() -> int:
    """Validate supplied release artifacts and return a nonzero status on leakage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--private-prefix", action="append", required=True)
    parser.add_argument("--python-cli", type=Path)
    args = parser.parse_args()
    checker = PathChecker(args.private_prefix)
    for artifact in args.artifacts:
        if not artifact.exists():
            parser.error(f"missing release artifact: {artifact}")
        files = artifact.rglob("*") if artifact.is_dir() else (artifact,)
        for path in files:
            if path.is_file():
                checker.check_file(path)
    if args.python_cli:
        checker.check_python_cli(args.python_cli)
    if checker.failures:
        for label in sorted(checker.failures):
            print(f"Private build path in {label}")
        print(f"FAIL: {len(checker.failures)} affected files or Python modules; checked {checker.files} files and {checker.python_modules} frozen modules")
        return 1
    print(f"PASS: {checker.files} files and {checker.python_modules} frozen Python modules contain no private build prefixes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# TYLOINFER_PATH_CHECKER_END
TYLOINFER_PATH_CHECKER
}

# Reject directory moves outside the packaging workspace or against its root.
mfq_assert_managed_path() {
  "${mfq_venv_dir}/bin/python" - "${mfq_script_dir}" "$1" <<'TYLOINFER_MANAGED_PATH'
from pathlib import Path
import sys

root, target = (Path(value).resolve() for value in sys.argv[1:])
if target == root or root not in target.parents:
    raise SystemExit(f"release path is outside packaging: {target}")
TYLOINFER_MANAGED_PATH
}

# Publish staged resource trees, retaining each previous tree for rollback.
mfq_publish_resources() {
  local mfq_name mfq_target mfq_backup mfq_source
  for mfq_name in sidecars Frameworks Resources; do
    mfq_source="${mfq_stage_root}/${mfq_name}"
    mfq_target="${mfq_script_dir}/${mfq_name}"
    mfq_backup="${mfq_target}.previous-${mfq_build_id}"
    mfq_assert_managed_path "${mfq_source}" || return 1
    mfq_assert_managed_path "${mfq_target}" || return 1
    mfq_assert_managed_path "${mfq_backup}" || return 1
    [[ -d "${mfq_source}" ]] || { fail "staged release resource is missing: ${mfq_source}"; return 1; }
    [[ ! -e "${mfq_backup}" && ! -L "${mfq_backup}" ]] || { fail "release resource backup already exists: ${mfq_backup}"; return 1; }
    if [[ -e "${mfq_target}" || -L "${mfq_target}" ]]; then
      mv "${mfq_target}" "${mfq_backup}" || return 1
    else
      mfq_backup=""
    fi
    mfq_published_targets+=("${mfq_target}")
    mfq_published_backups+=("${mfq_backup}")
    mv "${mfq_source}" "${mfq_target}" || return 1
  done
  mfq_sidecar_dir="${mfq_script_dir}/sidecars"
  mfq_framework_dir="${mfq_script_dir}/Frameworks"
  mfq_resource_dir="${mfq_script_dir}/Resources"
}

# Restore previous resources after a failed build, retaining failed new trees.
mfq_rollback_resources() {
  local mfq_index mfq_target mfq_backup mfq_failed
  for ((mfq_index=${#mfq_published_targets[@]}-1; mfq_index>=0; mfq_index--)); do
    mfq_target="${mfq_published_targets[$mfq_index]}"
    mfq_backup="${mfq_published_backups[$mfq_index]}"
    mfq_failed="${mfq_target}.failed-${mfq_build_id}"
    if ! mfq_assert_managed_path "${mfq_target}" || ! mfq_assert_managed_path "${mfq_failed}"; then
      printf 'error: release rollback path validation failed\n' >&2
      return 1
    fi
    if [[ -e "${mfq_target}" || -L "${mfq_target}" ]]; then
      if [[ -e "${mfq_failed}" || -L "${mfq_failed}" ]] || ! mv "${mfq_target}" "${mfq_failed}"; then
        printf 'error: cannot preserve failed release resource: %s\n' "${mfq_target}" >&2
        return 1
      fi
    fi
    if [[ -n "${mfq_backup}" ]]; then
      mfq_assert_managed_path "${mfq_backup}" || return 1
      mv "${mfq_backup}" "${mfq_target}" || return 1
    fi
  done
}

# Hold an atomic checkout lock while shared native caches and release resources change.
mfq_acquire_release_lock() {
  local mfq_requested_lock="$1"
  if ! mkdir "${mfq_requested_lock}" 2>/dev/null; then
    fail "another macOS release may be running; if none is active, remove the empty stale lock with: rmdir \"${mfq_requested_lock}\""
    return 1
  fi
  mfq_release_lock_dir="${mfq_requested_lock}"
  mfq_release_lock_held=true
}

# Unmount validation media, restore failed resources, and release the owned checkout lock.
mfq_cleanup_release() {
  local mfq_status="$1"
  set +e
  if [[ "${mfq_mounted}" == true ]]; then
    hdiutil detach "${mfq_mount_dir}" >/dev/null
  fi
  if [[ -n "${mfq_mount_dir}" ]]; then
    rmdir "${mfq_mount_dir}" 2>/dev/null || true
  fi
  if [[ "${mfq_release_succeeded}" != true ]]; then
    mfq_rollback_resources || printf 'error: restore the preserved release resource backups manually\n' >&2
    if [[ -n "${mfq_build_root}" ]]; then
      printf 'release build retained at %s\n' "${mfq_build_root}" >&2
    fi
  fi
  if [[ "${mfq_release_lock_held:-false}" == true ]]; then
    if rmdir "${mfq_release_lock_dir}"; then
      mfq_release_lock_held=false
    else
      printf 'error: could not release checkout lock: %s\n' "${mfq_release_lock_dir}" >&2
      [[ "${mfq_status}" != 0 ]] || mfq_status=1
    fi
  fi
  return "${mfq_status}"
}

# Publish only a fully verified DMG and retain any previous artifact.
mfq_publish_dmg() {
  local mfq_candidate="$1" mfq_destination mfq_partial mfq_backup
  mfq_destination="${mfq_output_dir}/$(basename "${mfq_candidate}")"
  mfq_partial="${mfq_destination}.partial-${mfq_build_id}"
  mfq_backup="${mfq_destination}.previous-${mfq_build_id}"
  [[ ! -e "${mfq_partial}" && ! -L "${mfq_partial}" ]] || { fail "release artifact staging path already exists: ${mfq_partial}"; return 1; }
  [[ ! -e "${mfq_backup}" && ! -L "${mfq_backup}" ]] || { fail "release artifact backup already exists: ${mfq_backup}"; return 1; }
  install -m 644 "${mfq_candidate}" "${mfq_partial}" || return 1
  if [[ -e "${mfq_destination}" || -L "${mfq_destination}" ]]; then
    [[ -f "${mfq_destination}" && ! -L "${mfq_destination}" ]] || { fail "release artifact target must be a regular file: ${mfq_destination}"; return 1; }
    mv "${mfq_destination}" "${mfq_backup}" || return 1
  fi
  if ! mv "${mfq_partial}" "${mfq_destination}"; then
    if [[ -f "${mfq_backup}" ]]; then
      mv "${mfq_backup}" "${mfq_destination}" || true
    fi
    fail "could not publish the verified DMG"
    return 1
  fi
}

# Build, validate, and publish the native Metal runtime and desktop DMG.
mfq_main() {
  set -euo pipefail
  mfq_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  mfq_project_dir="$(cd -- "${mfq_script_dir}/.." && pwd)"
  mfq_python_version="${TYLOINFER_RELEASE_PYTHON:-3.12}"
  tyloinfer_dir="${mfq_project_dir}/TyloInferStudio"
  mfq_tauri_target="aarch64-apple-darwin"
  mfq_output_dir="${TYLOINFER_RELEASE_OUTPUT_DIR:-${mfq_script_dir}/dist}"
  mfq_cli_name="tyloinfer-cli"
  mfq_runtime_name="mfq-decode-metal-aarch64-apple-darwin"
  mfq_macos_deployment_target="${TYLOINFER_RELEASE_MACOS_DEPLOYMENT_TARGET:-26.2}"
  [[ "$(uname -s)" == "Darwin" ]] || fail "this release target requires macOS"
  [[ "$(uname -m)" == "arm64" ]] || fail "this release target requires native Apple Silicon"
  for mfq_command in uv cmake codesign hdiutil install_name_tool lipo npm cargo; do
    command -v "${mfq_command}" >/dev/null 2>&1 || fail "${mfq_command} is required"
  done
  xcrun --find metal >/dev/null 2>&1 || fail "Xcode Metal Toolchain is required to build release shaders"
  xcrun --find metallib >/dev/null 2>&1 || fail "Xcode Metal Toolchain is required to build release shaders"

  mkdir -p "${mfq_script_dir}/build"
  mfq_published_targets=()
  mfq_published_backups=()
  mfq_mounted=false
  mfq_mount_dir=""
  mfq_release_succeeded=false
  mfq_release_lock_held=false
  mfq_build_root=""
  trap 'mfq_cleanup_release "$?"' EXIT
  mfq_acquire_release_lock "${mfq_script_dir}/build/.mac-release.lock"
  mfq_build_root="$(mktemp -d "${mfq_script_dir}/build/mac.XXXXXXXX")"
  mfq_build_id="$(basename "${mfq_build_root}")"
  mfq_native_build_dir="${TYLOINFER_RELEASE_NATIVE_BUILD_DIR:-${mfq_script_dir}/build/runtime}"
  [[ "${mfq_native_build_dir}" == /* ]] || mfq_native_build_dir="${mfq_project_dir}/${mfq_native_build_dir}"
  mfq_pyinstaller_build_dir="${mfq_build_root}/pyinstaller"
  mfq_spec_dir="${mfq_build_root}/spec"
  mfq_hook_dir="${mfq_build_root}/hooks"
  mfq_stage_root="${mfq_build_root}/stage"
  mfq_sidecar_dir="${mfq_stage_root}/sidecars"
  mfq_framework_dir="${mfq_stage_root}/Frameworks"
  mfq_resource_dir="${mfq_stage_root}/Resources"
  mfq_venv_dir="${TYLOINFER_RELEASE_VENV:-${mfq_build_root}/venv}"
  [[ "${mfq_venv_dir}" == /* ]] || mfq_venv_dir="${mfq_project_dir}/${mfq_venv_dir}"
  [[ "${mfq_output_dir}" == /* ]] || mfq_output_dir="${mfq_script_dir}/${mfq_output_dir}"
  mkdir -p \
    "${mfq_native_build_dir}" \
    "${mfq_pyinstaller_build_dir}" \
    "${mfq_spec_dir}" \
    "${mfq_sidecar_dir}" \
    "${mfq_framework_dir}" \
    "${mfq_resource_dir}" \
    "${mfq_output_dir}"

  cd "${mfq_project_dir}"
  if [[ "${TYLOINFER_RELEASE_REUSE_VENV:-0}" == "1" ]]; then
    [[ -x "${mfq_venv_dir}/bin/python" ]] || fail "reused release environment has no Python"
    [[ -x "${mfq_venv_dir}/bin/pyinstaller" ]] || fail "reused release environment has no PyInstaller"
  else
    [[ ! -e "${mfq_venv_dir}" ]] || fail "release environment already exists; choose a fresh TYLOINFER_RELEASE_VENV or enable TYLOINFER_RELEASE_REUSE_VENV=1"
    uv venv --python "${mfq_python_version}" "${mfq_venv_dir}"
  fi
  source "${mfq_venv_dir}/bin/activate"
  mfq_cargo_target_dir="${CARGO_TARGET_DIR:-${tyloinfer_dir}/src-tauri/target}"
  [[ "${mfq_cargo_target_dir}" == /* ]] || mfq_cargo_target_dir="${tyloinfer_dir}/${mfq_cargo_target_dir}"
  export CARGO_TARGET_DIR="${mfq_cargo_target_dir}"
  mfq_configure_release_paths
  if [[ "${TYLOINFER_RELEASE_REUSE_VENV:-0}" != "1" ]]; then
    uv sync \
      --locked \
      --active \
      --no-default-groups \
      --no-editable \
      --group release \
      --extra metal
  fi

  mfq_write_sysconfig_hook "${mfq_hook_dir}"
  mfq_dmg_dir="${mfq_cargo_target_dir}/${mfq_tauri_target}/release/bundle/dmg"
  mfq_build_marker="${mfq_build_root}/build-started"
  touch "${mfq_build_marker}"

  command -v ninja >/dev/null 2>&1 || fail "ninja is required"
  mfq_mlx_root="$("${mfq_venv_dir}/bin/python" -c 'import mlx; print(next(iter(mlx.__path__)))')"
  for mfq_mlx_file in \
    "${mfq_mlx_root}/lib/libmlx.dylib" \
    "${mfq_mlx_root}/lib/libjaccl.dylib" \
    "${mfq_mlx_root}/lib/mlx.metallib"; do
    [[ -f "${mfq_mlx_file}" ]] || fail "missing MLX runtime file: ${mfq_mlx_file}"
  done

  cmake -S "${mfq_project_dir}/cpp_runtime" -B "${mfq_native_build_dir}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="${CFLAGS}" \
    -DCMAKE_CXX_FLAGS="${CXXFLAGS}" \
    -DCMAKE_OBJCXX_FLAGS="${OBJCXXFLAGS}" \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_OSX_DEPLOYMENT_TARGET="${mfq_macos_deployment_target}" \
    -DTYLOINFER_BUILD_RUNTIME_COMMUNICATION=ON \
    -DTYLOINFER_ENABLE_MFQ_COMPAT="${TYLOINFER_ENABLE_MFQ_COMPAT:-OFF}" \
    -DTYLOINFER_BUILD_METAL_RUNTIME=ON \
    -DTYLOINFER_BUILD_PRECOMPILED_METAL=ON \
    -DTYLOINFER_MLX_ROOT="${mfq_mlx_root}" \
    -DTYLOINFER_MLX_METALLIB_DEFAULT:STRING=
  cmake --build "${mfq_native_build_dir}" \
    --target mfq-decode-metal \
    --parallel "${TYLOINFER_RELEASE_JOBS:-$(sysctl -n hw.ncpu)}"

  mfq_runtime_source="${mfq_native_build_dir}/metal/mfq-decode-metal"
  mfq_video_source="${mfq_native_build_dir}/metal/libmfq_avfoundation_video.dylib"
  [[ -x "${mfq_runtime_source}" ]] || fail "native build did not create mfq-decode-metal"
  [[ -f "${mfq_video_source}" ]] || fail "native build did not create the AVFoundation video library"

  install -m 755 "${mfq_runtime_source}" "${mfq_sidecar_dir}/${mfq_runtime_name}"
  install -m 644 "${mfq_mlx_root}/lib/libmlx.dylib" "${mfq_framework_dir}/libmlx.dylib"
  install -m 644 "${mfq_mlx_root}/lib/libjaccl.dylib" "${mfq_framework_dir}/libjaccl.dylib"
  install -m 755 "${mfq_video_source}" "${mfq_framework_dir}/libmfq_avfoundation_video.dylib"
  install -m 644 "${mfq_mlx_root}/lib/mlx.metallib" "${mfq_resource_dir}/mlx.metallib"
  [[ -s "${mfq_native_build_dir}/metal/lib/mfq-metal/manifest.json" ]] \
    || fail "native build did not create the precompiled MFQ Metal libraries"
  cp -R "${mfq_native_build_dir}/metal/lib/mfq-metal" "${mfq_resource_dir}/mfq-metal"

  mfq_license_dir="${mfq_resource_dir}/licenses"
  mkdir -p "${mfq_license_dir}"
  install -m 644 "${mfq_project_dir}/LICENSE" \
    "${mfq_license_dir}/MFQ-Apache-2.0.txt"
  install -m 644 "${mfq_project_dir}/NOTICE" \
    "${mfq_license_dir}/NOTICE.txt"
  install -m 644 "${mfq_project_dir}"/LICENSES/*.txt "${mfq_license_dir}/"

  install_name_tool -add_rpath "@executable_path/../Frameworks" \
    "${mfq_sidecar_dir}/${mfq_runtime_name}"

  mfq_signing_identity="${TYLOINFER_RELEASE_SIGNING_IDENTITY:-${APPLE_SIGNING_IDENTITY:--}}"
  for mfq_signed_file in \
    "${mfq_framework_dir}/libjaccl.dylib" \
    "${mfq_framework_dir}/libmlx.dylib" \
    "${mfq_framework_dir}/libmfq_avfoundation_video.dylib" \
    "${mfq_sidecar_dir}/${mfq_runtime_name}"; do
    codesign --force --sign "${mfq_signing_identity}" "${mfq_signed_file}"
    codesign --verify --strict --verbose=2 "${mfq_signed_file}"
  done

  TYLOINFER_MLX_METALLIB="${mfq_resource_dir}/mlx.metallib" \
  DYLD_LIBRARY_PATH="${mfq_framework_dir}" \
    "${mfq_sidecar_dir}/${mfq_runtime_name}" --self-test-metal

  mfq_pyinstaller_args=(
    --noconfirm
    --onedir
    --name "${mfq_cli_name}"
    --target-architecture arm64
    --paths "${mfq_project_dir}"
    --additional-hooks-dir "${mfq_hook_dir}"
    --collect-data tyloinfer
    --collect-all av
    --collect-all fastapi
    --collect-all httpx
    --collect-all socksio
    --collect-all PIL
    --collect-all pydantic
    --collect-all pypdf
    --collect-all mlx
    --collect-all tokenizers
    --collect-all gguf
    --collect-all huggingface_hub
    --collect-all modelscope_hub
    --collect-all uvicorn
    --collect-all websockets
    --copy-metadata requests
    --copy-metadata huggingface-hub
    --copy-metadata modelscope-hub
    --exclude-module torch
    --exclude-module transformers
    --exclude-module safetensors
    --exclude-module scipy
    --exclude-module tiktoken
    --exclude-module tyloinfer.runtime
    --exclude-module minicpmo_utils
    --exclude-module stepaudio2_minicpmo
    --exclude-module torchaudio
    --exclude-module onnxruntime
    --exclude-module s3tokenizer
    --exclude-module hyperpyyaml
    --exclude-module librosa
    --distpath "${mfq_resource_dir}"
    --workpath "${mfq_pyinstaller_build_dir}"
    --specpath "${mfq_spec_dir}"
  )
  if [[ "${mfq_signing_identity}" != "-" ]]; then
    mfq_pyinstaller_args+=(--codesign-identity "${mfq_signing_identity}")
  fi
  mfq_pyinstaller_args+=("${mfq_project_dir}/tyloinfer/cli.py")
  "${mfq_venv_dir}/bin/pyinstaller" "${mfq_pyinstaller_args[@]}"

  mfq_cli_path="${mfq_resource_dir}/${mfq_cli_name}/${mfq_cli_name}"
  [[ -x "${mfq_cli_path}" ]] || fail "PyInstaller did not create the unified mfq CLI"
  for mfq_arm64_file in \
    "${mfq_cli_path}" \
    "${mfq_sidecar_dir}/${mfq_runtime_name}"; do
    lipo -archs "${mfq_arm64_file}" | tr ' ' '\n' | grep -qx arm64 \
      || fail "${mfq_arm64_file} is not arm64"
  done
  codesign --verify --strict --verbose=2 "${mfq_cli_path}"

  "${mfq_cli_path}" --version
  "${mfq_cli_path}" serve --help >/dev/null

  # Publish validated resources to Tauri's fixed paths and retain previous resources as backups.
  mfq_publish_resources
  cd "${tyloinfer_dir}"
  npm ci
  npm run check
  if [[ "${mfq_signing_identity}" != "-" ]]; then
    CI=true \
    APPLE_SIGNING_IDENTITY="${APPLE_SIGNING_IDENTITY:-${mfq_signing_identity}}" \
      npm run tauri -- build \
        --config src-tauri/tauri.release-macos.conf.json \
        --config '{"bundle":{"macOS":{"hardenedRuntime":true}}}' \
        --target "${mfq_tauri_target}"
  else
    CI=true \
    APPLE_SIGNING_IDENTITY="${APPLE_SIGNING_IDENTITY:-${mfq_signing_identity}}" \
      npm run tauri -- build \
        --config src-tauri/tauri.release-macos.conf.json \
        --target "${mfq_tauri_target}"
  fi

  [[ -d "${mfq_dmg_dir}" ]] || fail "Tauri did not create a DMG directory"
  mfq_dmg_path="$(find "${mfq_dmg_dir}" -maxdepth 1 -type f -name '*.dmg' -newer "${mfq_build_marker}" -print -quit)"
  [[ -n "${mfq_dmg_path}" ]] || fail "Tauri did not create a DMG for this build"
  mkdir -p "${mfq_build_root}/verified"
  mfq_release_dmg="${mfq_build_root}/verified/$(basename "${mfq_dmg_path}")"
  install -m 644 "${mfq_dmg_path}" "${mfq_release_dmg}"
  hdiutil verify "${mfq_release_dmg}"
  mfq_mount_dir="$(mktemp -d "${TMPDIR:-/tmp}/mfq-release-verify.XXXXXX")"
  hdiutil attach -readonly -nobrowse -mountpoint "${mfq_mount_dir}" "${mfq_release_dmg}" >/dev/null
  mfq_mounted=true
  mfq_packaged_app="${mfq_mount_dir}/TyloInfer.app"
  codesign --verify --deep --strict --verbose=2 "${mfq_packaged_app}"
  mfq_check_release_paths \
    "${mfq_packaged_app}" "${mfq_release_dmg}" \
    "${mfq_private_prefix_args[@]}" \
    --python-cli "${mfq_packaged_app}/Contents/Resources/tyloinfer-cli/tyloinfer-cli" \
    || fail "packaged Studio contains a private build path"
  TYLOINFER_MLX_METALLIB="${mfq_packaged_app}/Contents/Resources/mlx.metallib" \
    "${mfq_packaged_app}/Contents/MacOS/mfq-decode-metal" --self-test-metal
  "${mfq_packaged_app}/Contents/Resources/tyloinfer-cli/tyloinfer-cli" --version >/dev/null
  hdiutil detach "${mfq_mount_dir}" >/dev/null
  mfq_mounted=false
  rmdir "${mfq_mount_dir}"
  mfq_mount_dir=""
  mfq_publish_dmg "${mfq_release_dmg}"
  mfq_release_succeeded=true
  printf 'built %s\n' "${mfq_output_dir}/$(basename "${mfq_release_dmg}")"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  mfq_main "$@"
fi
