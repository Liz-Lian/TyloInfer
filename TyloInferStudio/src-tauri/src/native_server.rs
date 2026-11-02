use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

pub const EXECUTABLE_OVERRIDE: &str = "TYLOINFER_SERVER_EXECUTABLE";

#[cfg(windows)]
const EXECUTABLE_NAME: &str = "tyloinfer-server.exe";
#[cfg(not(windows))]
const EXECUTABLE_NAME: &str = "tyloinfer-server";

/// Rejects missing files and aliases of Studio itself before starting a backend.
fn usable_program(path: &Path) -> bool {
    let Ok(candidate) = path.canonicalize() else { return false; };
    let Ok(current) = std::env::current_exe().and_then(fs::canonicalize) else { return false; };
    candidate.is_file() && candidate != current
}

/// Finds only the native server, honoring an explicit override without fallback on errors.
pub fn discover(
    resource_dir: Option<&Path>,
    repository: &Path,
    development: bool,
    explicit: Option<&OsStr>,
) -> Result<PathBuf, String> {
    if let Some(explicit) = explicit {
        let path = PathBuf::from(explicit);
        return if usable_program(&path) {
            path.canonicalize().map_err(|error| error.to_string())
        } else {
            Err(format!("{EXECUTABLE_OVERRIDE} must point to a native {EXECUTABLE_NAME} executable: {}", path.display()))
        };
    }
    let mut candidates = Vec::new();
    if let Some(resource_dir) = resource_dir {
        candidates.push(resource_dir.join("tyloinfer-server").join(EXECUTABLE_NAME));
    }
    if development {
        let build = repository.join("build/native-server/server");
        candidates.push(build.join(EXECUTABLE_NAME));
        for configuration in ["Release", "RelWithDebInfo", "Debug", "MinSizeRel"] {
            candidates.push(build.join(configuration).join(EXECUTABLE_NAME));
        }
    }
    if let Some(candidate) = candidates.iter().find(|candidate| usable_program(candidate)) {
        return candidate.canonicalize().map_err(|error| error.to_string());
    }
    Err(format!("Native TyloInfer Server was not found. Build the native server or set {EXECUTABLE_OVERRIDE}. Searched:\n{}",
        candidates.iter().map(|path| path.display().to_string()).collect::<Vec<_>>().join("\n")))
}

/// Builds the native CLI contract; credentials remain in the environment and OS keyring.
pub fn command(
    executable: &Path,
    data_dir: &Path,
    port: u16,
    instance_id: &str,
    web_root: Option<&Path>,
) -> Result<Command, String> {
    let mut command = Command::new(executable);
    command
        .arg("--exit-on-stdin-close")
        .arg("--studio-instance-id").arg(instance_id)
        .arg("--data-dir").arg(data_dir)
        .arg("--host").arg("127.0.0.1")
        .arg("--port").arg(port.to_string())
        .arg("--api-key-env").arg("TYLOINFER_SERVER_API_KEY")
        .current_dir(data_dir);
    if let Some(web_root) = web_root {
        command.arg("--web-root").arg(web_root);
    }
    #[cfg(windows)]
    {
        // 原生服务及其 DLL 位于同一目录，同时保留开发工具链的搜索路径。
        let directory = executable.parent().ok_or("Native server executable has no parent directory")?;
        let mut paths = vec![directory.to_path_buf()];
        if let Some(existing) = std::env::var_os("PATH") {
            paths.extend(std::env::split_paths(&existing));
        }
        let path = std::env::join_paths(paths).map_err(|error| format!("invalid native server DLL search path: {error}"))?;
        command.env("PATH", path);
    }
    Ok(command)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discovery_uses_the_resource_layout_and_cmake_configuration_directories() {
        let directory = std::env::temp_dir().join(format!("tyloinfer-discovery-{}", uuid::Uuid::new_v4()));
        let bundled = directory.join("resources/tyloinfer-server").join(EXECUTABLE_NAME);
        let development = directory.join("build/native-server/server/Release").join(EXECUTABLE_NAME);
        fs::create_dir_all(bundled.parent().unwrap()).unwrap();
        fs::create_dir_all(development.parent().unwrap()).unwrap();
        fs::write(&development, b"native server fixture").unwrap();
        let resource_dir = directory.join("resources");
        assert_eq!(discover(Some(&resource_dir), &directory, true, None).unwrap(), development.canonicalize().unwrap());
        assert!(discover(Some(&resource_dir), &directory, false, None).is_err());
        fs::write(&bundled, b"bundled native server fixture").unwrap();
        assert_eq!(discover(Some(&resource_dir), &directory, true, None).unwrap(), bundled.canonicalize().unwrap());
        assert_eq!(discover(Some(&resource_dir), &directory, false, None).unwrap(), bundled.canonicalize().unwrap());
        assert_eq!(discover(Some(&resource_dir), &directory, false, Some(development.as_os_str())).unwrap(), development.canonicalize().unwrap());
        assert!(discover(Some(&resource_dir), &directory, true, Some(OsStr::new("missing-native-server"))).is_err());
        assert!(discover(Some(&resource_dir), &directory, true, Some(std::env::current_exe().unwrap().as_os_str())).is_err());
        fs::remove_file(bundled).unwrap();
        fs::remove_file(development).unwrap();
        for path in ["resources/tyloinfer-server", "resources", "build/native-server/server/Release", "build/native-server/server", "build/native-server", "build", ""] {
            fs::remove_dir(directory.join(path)).unwrap();
        }
    }

    #[test]
    fn native_command_uses_supported_arguments_and_keeps_secrets_off_the_command_line() {
        let executable = std::env::current_exe().unwrap();
        let data_dir = executable.parent().unwrap();
        let web_root = data_dir.join("web-ui");
        let command = command(&executable, data_dir, 8091, "studio-instance", Some(&web_root)).unwrap();
        let arguments = command.get_args().map(|value| value.to_string_lossy().to_string()).collect::<Vec<_>>();
        assert_eq!(arguments, ["--exit-on-stdin-close", "--studio-instance-id", "studio-instance",
            "--data-dir", &data_dir.to_string_lossy(), "--host", "127.0.0.1", "--port", "8091",
            "--api-key-env", "TYLOINFER_SERVER_API_KEY", "--web-root", &web_root.to_string_lossy()]);
        assert_eq!(command.get_current_dir(), Some(data_dir));
        #[cfg(windows)]
        {
            let (_, path) = command.get_envs().find(|(name, _)| *name == OsStr::new("PATH")).unwrap();
            assert_eq!(std::env::split_paths(path.unwrap()).next().unwrap(), data_dir);
        }
    }
}
