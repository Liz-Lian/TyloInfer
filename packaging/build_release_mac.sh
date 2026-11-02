#!/usr/bin/env bash
# Prepare native macOS resources only; block inference releases until the server integrates the Metal engine.
set -euo pipefail

mfq_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mfq_project_dir="$(cd -- "${mfq_script_dir}/.." && pwd)"
tyloinfer_dir="${mfq_project_dir}/TyloInferStudio"
mfq_venv_dir="${TYLOINFER_RELEASE_VENV:-${mfq_script_dir}/.venv}"
mfq_framework_dir="${mfq_script_dir}/Frameworks"
mfq_resource_dir="${mfq_script_dir}/Resources"
mfq_server_dir="${mfq_resource_dir}/tyloinfer-server"
mfq_macos_deployment_target="${TYLOINFER_RELEASE_MACOS_DEPLOYMENT_TARGET:-26.2}"

# /** Stops preparation with an actionable build error. */
fail() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

[[ "$(uname -s)" == Darwin ]] || fail "this release target requires macOS"
[[ "$(uname -m)" == arm64 ]] || fail "this release target requires native Apple Silicon"
if [[ "${1:-}" != --prepare-only || $# != 1 ]]; then
  fail "macOS inference release blocked: tyloinfer-server has no Metal engine factory. --prepare-only stages diagnostic resources; it does not produce a DMG or an inference release."
fi
for mfq_command in cmake ninja codesign install_name_tool lipo; do
  command -v "${mfq_command}" >/dev/null 2>&1 || fail "${mfq_command} is required"
done
[[ -x "${mfq_venv_dir}/bin/python" ]] || fail "set TYLOINFER_RELEASE_VENV to an existing build-only Python environment with MLX"
xcrun --find metal >/dev/null 2>&1 || fail "Xcode Metal Toolchain is required"
xcrun --find metallib >/dev/null 2>&1 || fail "Xcode Metal Toolchain is required"
# Refuse to overwrite existing mapped resources to avoid mixing in older Python or different-architecture artifacts.
for mfq_destination in "${mfq_server_dir}" "${mfq_framework_dir}" "${mfq_resource_dir}/mfq-metal"; do
  [[ ! -e "${mfq_destination}" ]] || fail "preserve or manually remove existing resources before preparation: ${mfq_destination}"
done
mkdir -p "${mfq_script_dir}/build" "${mfq_resource_dir}"
mfq_native_build_dir="$(mktemp -d "${mfq_script_dir}/build/native-macos.XXXXXX")"
source "${mfq_script_dir}/macos_release_paths.sh"
mfq_configure_release_paths
mfq_mlx_root="$("${mfq_venv_dir}/bin/python" -c 'import mlx; print(next(iter(mlx.__path__)))')"
for mfq_file in libmlx.dylib libjaccl.dylib mlx.metallib; do
  [[ -f "${mfq_mlx_root}/lib/${mfq_file}" ]] || fail "missing MLX build resource: ${mfq_file}"
done

cmake -S "${mfq_project_dir}/cpp_runtime" -B "${mfq_native_build_dir}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
  -DCMAKE_C_FLAGS="${CFLAGS}" -DCMAKE_CXX_FLAGS="${CXXFLAGS}" -DCMAKE_OBJCXX_FLAGS="${OBJCXXFLAGS}" \
  -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET="${mfq_macos_deployment_target}" \
  -DTYLOINFER_BUILD_SERVER=ON -DTYLOINFER_BUILD_RUNTIME_COMMUNICATION=ON \
  -DTYLOINFER_BUILD_CUDA_RUNTIME=OFF -DTYLOINFER_BUILD_HIP_RUNTIME=OFF \
  -DTYLOINFER_BUILD_METAL_RUNTIME=ON -DTYLOINFER_BUILD_PRECOMPILED_METAL=ON \
  -DTYLOINFER_ENABLE_MFQ_COMPAT="${TYLOINFER_ENABLE_MFQ_COMPAT:-OFF}" \
  -DTYLOINFER_MLX_ROOT="${mfq_mlx_root}" -DTYLOINFER_MLX_METALLIB_DEFAULT:STRING=
cmake --build "${mfq_native_build_dir}" --target tyloinfer-server mfq-decode-metal \
  --parallel "${TYLOINFER_RELEASE_JOBS:-$(sysctl -n hw.ncpu)}"

mfq_server_source="${mfq_native_build_dir}/server/tyloinfer-server"
[[ -x "${mfq_server_source}" ]] || fail "native build did not create tyloinfer-server"
[[ -s "${mfq_native_build_dir}/metal/lib/mfq-metal/manifest.json" ]] || fail "missing precompiled Metal libraries"
mkdir -p "${mfq_server_dir}" "${mfq_framework_dir}" "${mfq_resource_dir}/licenses"
install -m 755 "${mfq_server_source}" "${mfq_server_dir}/tyloinfer-server"
for mfq_file in libmlx.dylib libjaccl.dylib; do
  install -m 644 "${mfq_mlx_root}/lib/${mfq_file}" "${mfq_framework_dir}/${mfq_file}"
done
install -m 755 "${mfq_native_build_dir}/metal/libmfq_avfoundation_video.dylib" "${mfq_framework_dir}/libmfq_avfoundation_video.dylib"
install -m 644 "${mfq_mlx_root}/lib/mlx.metallib" "${mfq_resource_dir}/mlx.metallib"
cp -R "${mfq_native_build_dir}/metal/lib/mfq-metal" "${mfq_resource_dir}/mfq-metal"
install -m 644 "${mfq_project_dir}/LICENSE" "${mfq_resource_dir}/licenses/MFQ-Apache-2.0.txt"
install -m 644 "${mfq_project_dir}/NOTICE" "${mfq_resource_dir}/licenses/NOTICE.txt"
install -m 644 "${mfq_project_dir}"/LICENSES/*.txt "${mfq_resource_dir}/licenses/"
install_name_tool -add_rpath "@executable_path/../../Frameworks" "${mfq_server_dir}/tyloinfer-server"
mfq_signing_identity="${TYLOINFER_RELEASE_SIGNING_IDENTITY:-${APPLE_SIGNING_IDENTITY:--}}"
for mfq_file in "${mfq_server_dir}/tyloinfer-server" "${mfq_framework_dir}"/*.dylib; do
  lipo -archs "${mfq_file}" | tr ' ' '\n' | grep -qx arm64 || fail "not arm64: ${mfq_file}"
  codesign --force --sign "${mfq_signing_identity}" "${mfq_file}"
  codesign --verify --strict --verbose=2 "${mfq_file}"
done
"${mfq_server_dir}/tyloinfer-server" --version
"${mfq_server_dir}/tyloinfer-server" --help
"${mfq_venv_dir}/bin/python" "${mfq_script_dir}/check_release_paths.py" \
  "${mfq_server_dir}" "${mfq_framework_dir}" "${mfq_resource_dir}/mlx.metallib" \
  "${mfq_resource_dir}/mfq-metal" "${mfq_private_prefix_args[@]}"
printf '%s\n' 'Prepared diagnostic resources only. Metal server inference, transitive dylib closure, Studio signing and DMG verification remain blocked; no release was generated.'
