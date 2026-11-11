






import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
STUDIO = ROOT / "TyloInferStudio"
TAURI = STUDIO / "src-tauri"
RUST = (TAURI / "src" / "main.rs").read_text(encoding="utf-8")
BUILD = (TAURI / "build.rs").read_text(encoding="utf-8")
STUDIO_BRIDGE = (STUDIO / "src" / "studio.ts").read_text(encoding="utf-8")
PLATFORM_BRIDGE = STUDIO / "src" / "shared" / "platform" / "studio.ts"
if PLATFORM_BRIDGE.exists():
    STUDIO_BRIDGE += "\n" + PLATFORM_BRIDGE.read_text(encoding="utf-8")
MODEL_BROWSER = (
    STUDIO / "src" / "features" / "models" / "ModelBrowser.tsx"
).read_text(encoding="utf-8")
UPDATE_MANAGER = (
    STUDIO / "src" / "features" / "settings" / "UpdateManager.tsx"
).read_text(encoding="utf-8")
UPDATER = (TAURI / "src" / "updater.rs").read_text(encoding="utf-8")
RELEASE_SCRIPT = (ROOT / "packaging" / "build_release_mac.sh").read_text(encoding="utf-8")


def test_studio_uses_one_package_for_web_and_desktop_clients():
    config = json.loads((TAURI / "tauri.conf.json").read_text(encoding="utf-8"))
    release_config = json.loads(
        (TAURI / "tauri.release-macos.conf.json").read_text(encoding="utf-8")
    )
    package = json.loads((STUDIO / "package.json").read_text(encoding="utf-8"))
    assert not (STUDIO / "web").exists()
    assert not (STUDIO / "desktop").exists()
    assert package["name"] == "@tyloinfer/studio"
    assert package["scripts"]["tauri"] == "tauri"
    assert config["build"]["frontendDist"] == "../dist"
    assert config["build"]["beforeBuildCommand"] == "npm run build"
    assert config["identifier"] == "com.tylogi.tyloinfer"
    assert "icons/icon.ico" in config["bundle"]["icon"]
    assert "icons/icon.icns" in config["bundle"]["icon"]
    assert "IconDir::new" in BUILD
    assert "IconFamily::new" in BUILD
    assert "media-src 'self' asset: data: blob:" in config["app"]["security"]["csp"]
    assert release_config["bundle"]["macOS"]["hardenedRuntime"] is False
    assert 'if [[ "${mfq_signing_identity}" != "-" ]]' in RELEASE_SCRIPT
    assert '"hardenedRuntime":true' in RELEASE_SCRIPT
    assert "mfq_add_release_path()" in RELEASE_SCRIPT
    assert "mfq_configure_release_paths" in RELEASE_SCRIPT
    assert '-DCMAKE_C_FLAGS="${CFLAGS}"' in RELEASE_SCRIPT
    assert '-DCMAKE_CXX_FLAGS="${CXXFLAGS}"' in RELEASE_SCRIPT
    assert '-DCMAKE_OBJCXX_FLAGS="${OBJCXXFLAGS}"' in RELEASE_SCRIPT
    assert '--additional-hooks-dir "${mfq_hook_dir}"' in RELEASE_SCRIPT
    assert 'mfq_write_sysconfig_hook "${mfq_hook_dir}"' in RELEASE_SCRIPT
    assert '"${mfq_packaged_app}" "${mfq_release_dmg}"' in RELEASE_SCRIPT
    assert '--python-cli "${mfq_packaged_app}/Contents/Resources/tyloinfer-cli/tyloinfer-cli"' in RELEASE_SCRIPT
    assert "packaged Studio contains a private build path" in RELEASE_SCRIPT
    assert 'mfq-decode-metal" --self-test-metal' in RELEASE_SCRIPT


def test_studio_macos_icons_have_platform_padding():
    assert "const MACOS_ICON_SCALE: f64 = 0.79;" in BUILD
    assert 'std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos")' in BUILD
    assert "render(size, MACOS_ICON_SCALE)" in BUILD
    assert "IconImage::from_rgba_data(size, size, render(size, 1.0))" in BUILD
    for size in (32, 128, 256):
        assert f'write_png(&directory.join("{size}x{size}.png"), {size}, png_scale);' in BUILD


def test_studio_starts_the_unified_local_server_and_bundled_runtime():
    assert 'Some("tyloinfer-server")' in RUST
    assert "Command::new" in RUST
    assert "studio_start_local" in RUST
    assert '.arg("serve")' in RUST
    assert '.arg("--data-dir")' in RUST
    assert '.arg("--db")' not in RUST
    assert 'command.arg("--running-executable")' in RUST
    assert "TYLOINFER_MLX_METALLIB" in RUST
    assert "TYLOINFER_AVFOUNDATION_VIDEO_LIBRARY" in RUST


def test_studio_keeps_web_ui_available_on_the_server_listener():
    assert '.arg("--no-web-ui")' not in RUST
    for platform in ("macos", "windows"):
        config = json.loads(
            (TAURI / f"tauri.release-{platform}.conf.json").read_text(encoding="utf-8")
        )
        assert config["bundle"]["resources"]["../dist"] == "web-ui"


def test_studio_supports_local_and_remote_server_connections_with_voice_controls():
    for command in ('studio_configure', 'studio_start_local', 'studio_select_model_directory'):
        assert command in RUST


def test_studio_can_select_and_load_an_external_mfq_directory_in_local_mode():
    assert 'rfd::AsyncFileDialog::new()' in RUST
    assert '.pick_folder()' in RUST
    assert "tauri.invoke('studio_select_model_directory', { initialDirectory })" in STUDIO_BRIDGE
    assert '.set_directory(initial_directory)' in RUST
    assert 'path: path.to_string()' in RUST


def test_studio_uses_native_confirmation_dialogs_for_destructive_actions():
    assert "fn studio_confirm(message: String) -> bool" in RUST
    assert "rfd::MessageButtons::YesNo" in RUST
    assert "studio_confirm," in RUST
    assert "tauri.invoke<boolean>('studio_confirm', { message })" in STUDIO_BRIDGE


def test_model_hub_resolves_links_and_downloads_selected_variants():
    assert "modelsApi.resolveHubModel(reference, provider)" in MODEL_BROWSER
    assert (
        'destination: `${repositoryDownloadPath(source)}${sharedEngram ? "" : `/${variantPath}`}`'
        in MODEL_BROWSER
    )
    assert "return `models/${source.provider}/${source.repo_id.split('/').map(safeSegment).join('/')}`" in MODEL_BROWSER
    assert "downloadPatterns(variant)" in MODEL_BROWSER
    assert "jobsApi.createJob" in MODEL_BROWSER
    assert 'tr("官方模型", "Official")' in MODEL_BROWSER
    assert 'tr("第三方模型", "Community")' in MODEL_BROWSER
    assert 'tr("内存压力", "Memory pressure")' not in MODEL_BROWSER
    assert "not model capability or quality" not in MODEL_BROWSER
    assert "三星推荐" not in MODEL_BROWSER
    assert "3-star recommendation" not in MODEL_BROWSER
    for symbol in ('"★★★"', '"★★"', '"★"', '"▲"', '"✕"'):
        assert symbol not in MODEL_BROWSER


def test_studio_checks_releases_and_keeps_verified_versions_for_rollback():
    assert "studio_update_status" in RUST
    assert "studio_update_download" in RUST
    assert "studio_update_install" in RUST
    assert "studio_update_delete" in RUST
    assert "studio_update_progress" in RUST
    assert "api.github.com/repos/Tylogi/TyloInfer/releases" in UPDATER
    assert "release SHA-256 verification failed" in UPDATER
    assert "snapshot_current_bundle" in UPDATER
    assert "replace_macos_bundle" in UPDATER
    assert "com.tylogi.tyloinfer" in UPDATER
    assert "studioUpdateStatus(false)" in UPDATE_MANAGER
    assert "tr('自动检查更新', 'Automatically check for updates')" in UPDATE_MANAGER


def _release_bash():
    """Return a native Bash executable for exercising sourceable release helpers."""
    if os.name == 'nt':
        git = shutil.which('git')
        if git:
            candidate = Path(git).resolve().parents[1] / 'bin/bash.exe'
            if candidate.is_file():
                return str(candidate)
        pytest.skip('Git Bash is required for release helper checks on Windows')
    bash = shutil.which('bash')
    if not bash:
        pytest.skip('Bash is required for release helper checks')
    return bash


def _run_macos_release_helper(tmp_path, command):
    """Exercise the actual helper functions using an isolated packaging tree."""
    packaging = tmp_path / 'packaging'
    packaging.mkdir()
    python = packaging / 'venv/bin/python'
    python.parent.mkdir(parents=True)
    python.write_text(
        '#!/usr/bin/env bash\nexec "' + Path(sys.executable).as_posix() + '" "$@"\n',
        encoding='utf-8',
    )
    python.chmod(0o755)
    setup = r'''
source "$1"
mfq_script_dir="$2"
mfq_venv_dir="$2/venv"
mfq_build_id='mac.test'
mfq_stage_root="$2/build/mac.test/stage"
mfq_published_targets=()
mfq_published_backups=()
for name in sidecars Frameworks Resources; do
  mkdir -p "$mfq_stage_root/$name" "$mfq_script_dir/$name"
  printf old > "$mfq_script_dir/$name/old.txt"
  printf new > "$mfq_stage_root/$name/new.txt"
done
'''
    return subprocess.run(
        [_release_bash(), '-euc', setup + command, '--',
         (ROOT / 'packaging/build_release_mac.sh').as_posix(), packaging.as_posix()],
        capture_output=True, text=True, timeout=30,
    ), packaging


def test_macos_release_script_can_be_sourced_without_starting_a_build():
    """Importing helper functions must not run platform checks or build commands."""
    result = subprocess.run(
        [_release_bash(), '-euc', 'source "$1"; declare -F mfq_main mfq_write_sysconfig_hook',
         '--', (ROOT / 'packaging/build_release_mac.sh').as_posix()],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.splitlines() == ['mfq_main', 'mfq_write_sysconfig_hook']


def test_macos_resource_publication_can_restore_previous_trees(tmp_path):
    """A failed downstream build can recover all old resources without deleting either version."""
    result, packaging = _run_macos_release_helper(tmp_path, r'''
mfq_publish_resources
for name in sidecars Frameworks Resources; do
  test -f "$mfq_script_dir/$name/new.txt"
  test -f "$mfq_script_dir/$name.previous-mac.test/old.txt"
done
mfq_rollback_resources
''')
    assert result.returncode == 0, result.stdout + result.stderr
    for name in ('sidecars', 'Frameworks', 'Resources'):
        assert (packaging / name / 'old.txt').read_text() == 'old'
        assert (packaging / f'{name}.failed-mac.test/new.txt').read_text() == 'new'


def test_macos_partial_publication_restores_only_touched_resources(tmp_path):
    """An interrupted directory replacement retains untouched old resources and rolls moved ones back."""
    result, packaging = _run_macos_release_helper(tmp_path, r'''
mv() {
  if [[ "$1" == "$mfq_stage_root/Frameworks" ]]; then
    return 1
  fi
  command mv "$@"
}
if mfq_publish_resources; then
  printf 'publication unexpectedly succeeded\n' >&2
  exit 1
fi
mfq_rollback_resources
''')
    assert result.returncode == 0, result.stdout + result.stderr
    for name in ('sidecars', 'Frameworks', 'Resources'):
        assert (packaging / name / 'old.txt').read_text() == 'old'
    assert (packaging / 'sidecars.failed-mac.test/new.txt').is_file()
    assert (packaging / 'build/mac.test/stage/Frameworks/new.txt').is_file()
    assert (packaging / 'build/mac.test/stage/Resources/new.txt').is_file()


def test_macos_exit_trap_restores_published_resources(tmp_path):
    """A real failing process must restore old resources through the production EXIT trap."""
    result, packaging = _run_macos_release_helper(tmp_path, r'''
mfq_mounted=false
mfq_mount_dir=''
mfq_release_succeeded=false
mfq_build_root="$mfq_script_dir/build/mac.test"
trap 'mfq_cleanup_release "$?"' EXIT
mfq_publish_resources
false
''')
    assert result.returncode != 0
    assert 'release build retained at' in result.stderr
    for name in ('sidecars', 'Frameworks', 'Resources'):
        assert (packaging / name / 'old.txt').read_text() == 'old'
        assert (packaging / f'{name}.failed-mac.test/new.txt').read_text() == 'new'


def test_macos_resource_publication_rejects_escaping_paths(tmp_path):
    """Resolved paths outside packaging must be rejected before any resource is moved."""
    result, packaging = _run_macos_release_helper(tmp_path, r'''
mfq_stage_root="$2/../outside"
if mfq_publish_resources; then
  printf 'escaping path unexpectedly accepted\n' >&2
  exit 1
fi
''')
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'outside packaging' in result.stderr
    for name in ('sidecars', 'Frameworks', 'Resources'):
        assert (packaging / name / 'old.txt').read_text() == 'old'


def test_macos_verified_dmg_publication_preserves_previous_artifact(tmp_path):
    """Only a validated candidate is published, retaining the prior same-name DMG."""
    result, packaging = _run_macos_release_helper(tmp_path, r'''
mfq_output_dir="$2/dist"
mkdir -p "$mfq_output_dir"
printf old > "$mfq_output_dir/candidate.dmg"
printf verified > "$2/build/candidate.dmg"
mfq_publish_dmg "$2/build/candidate.dmg"
''')
    assert result.returncode == 0, result.stdout + result.stderr
    assert (packaging / 'dist/candidate.dmg').read_text() == 'verified'
    assert (packaging / 'dist/candidate.dmg.previous-mac.test').read_text() == 'old'


def test_macos_failed_dmg_publication_restores_previous_artifact(tmp_path):
    """A failed final rename preserves the previous release and the new partial file."""
    result, packaging = _run_macos_release_helper(tmp_path, r'''
mfq_output_dir="$2/dist"
mkdir -p "$mfq_output_dir"
printf old > "$mfq_output_dir/candidate.dmg"
printf verified > "$2/build/candidate.dmg"
mv() {
  if [[ "$1" == "$mfq_output_dir/candidate.dmg.partial-mac.test" ]]; then
    return 1
  fi
  command mv "$@"
}
if mfq_publish_dmg "$2/build/candidate.dmg"; then
  printf 'publication unexpectedly succeeded\n' >&2
  exit 1
fi
''')
    assert result.returncode == 0, result.stdout + result.stderr
    assert (packaging / 'dist/candidate.dmg').read_text() == 'old'
    assert (packaging / 'dist/candidate.dmg.partial-mac.test').read_text() == 'verified'
