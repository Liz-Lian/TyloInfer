#![cfg_attr(all(target_os = "windows", not(debug_assertions)), windows_subsystem = "windows")]

mod native_server;
mod updater;
#[cfg(target_os = "windows")]
mod windows_icon;

use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use tauri::{AppHandle, Manager, State};
use tokio::sync::Mutex;
use url::Url;

const CONFIG_FILE: &str = "studio.json";
const LOG_FILE: &str = "tyloinfer-server.log";
const LOG_ROTATE_BYTES: u64 = 16 * 1024 * 1024;
const CREDENTIAL_SERVICE: &str = "TyloInfer";
const CREDENTIAL_ACCOUNT: &str = "tyloinfer-server-api-key";
const SERVER_STOP_TIMEOUT: Duration = Duration::from_secs(30);

/// Reports whether the frontend should render the Windows titlebar controls.
#[tauri::command]
fn studio_custom_titlebar() -> bool {
    cfg!(target_os = "windows")
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
enum RuntimeMode {
    Local,
    Remote,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
struct StudioConfig {
    mode: RuntimeMode,
    remote_url: String,
    local_service_port: u16,
}

impl Default for StudioConfig {
    fn default() -> Self {
        Self {
            mode: RuntimeMode::Local,
            remote_url: "http://127.0.0.1:8090".into(),
            local_service_port: 8090,
        }
    }
}

#[derive(Debug, Serialize)]
struct StudioStatus {
    config: StudioConfig,
    service_url: String,
    reachable: bool,
    managed_pid: Option<u32>,
    connection: &'static str,
    server: Option<ServerIdentity>,
    connection_error: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(default)]
struct ServerIdentity {
    service: String,
    protocol_version: String,
    studio_protocol_version: String,
    version: String,
    instance_id: String,
    workspace_id: String,
    pid: u32,
    launch_mode: String,
}

impl ServerIdentity {
    fn validate(&self) -> Result<(), String> {
        if self.service != "tyloinfer-server" || self.protocol_version != "1.0"
            || self.studio_protocol_version != "1.0" || self.version != env!("TYLOINFER_SERVER_VERSION")
        {
            return Err(format!("Incompatible TyloInfer Server (version {}). Update the server or stop it so Studio can start its bundled backend.",
                if self.version.is_empty() { "unknown" } else { &self.version }));
        }
        if uuid::Uuid::parse_str(&self.instance_id).is_err() || self.pid == 0
            || self.workspace_id.len() != 64 || !self.workspace_id.bytes().all(|byte| byte.is_ascii_hexdigit())
            || !matches!(self.launch_mode.as_str(), "cli" | "studio")
        {
            return Err("TyloInfer Server returned an invalid instance identity. Update or restart the server.".into());
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
struct RegisteredModel {
    name: String,
}

#[derive(Debug, Deserialize)]
struct RegisteredModelList {
    data: Vec<RegisteredModel>,
}

#[derive(Serialize)]
struct SelectedModelDirectory {
    path: String,
    names: Vec<String>,
}

#[derive(Default)]
struct StudioState {
    start_lock: Mutex<()>,
    managed_server: std::sync::Mutex<Option<ManagedServer>>,
    external_server: std::sync::Mutex<Option<(String, ServerIdentity)>>,
    shutting_down: AtomicBool,
}

/// Keeps the process handle after lifecycle EOF until the child has actually exited.
struct ManagedServer {
    child: Child,
    input: Option<ChildStdin>,
    instance_id: String,
    port: u16,
}

impl StudioState {
    /// Releases ownership only after observing process exit, including failed launches.
    fn reap_exited_server(&self) {
        if let Ok(mut server) = self.managed_server.lock() {
            if let Some(process) = server.as_mut() {
                if process.child.try_wait().ok().flatten().is_some() {
                    server.take();
                }
            }
        }
    }

    /// Registers a child before readiness polling so cancellation and cleanup retain its handle.
    fn spawn_server(&self, command: &mut Command, instance_id: &str, port: u16) -> Result<u32, String> {
        let mut server = self.managed_server.lock().map_err(|error| error.to_string())?;
        if self.shutting_down.load(Ordering::SeqCst) {
            return Err("Studio is shutting down; no backend will be started.".into());
        }
        if server.is_some() {
            return Err("Studio's server is already starting or stopping. Retry after it exits; a second backend will not be launched.".into());
        }
        let mut child = command.spawn().map_err(|error| format!("failed to start native TyloInfer Server: {error}"))?;
        let pid = child.id();
        let input = child.stdin.take();
        let has_input = input.is_some();
        *server = Some(ManagedServer { child, input, instance_id: instance_id.into(), port });
        if !has_input { return Err("TyloInfer Server lifecycle pipe is unavailable".into()); }
        Ok(pid)
    }

    /// Checks the original child rather than treating another listener as a successful launch.
    fn check_starting_server(&self, pid: u32) -> Result<(), String> {
        let mut server = self.managed_server.lock().map_err(|error| error.to_string())?;
        let process = server.as_mut().filter(|process| process.child.id() == pid)
            .ok_or("TyloInfer Server stopped before startup completed.")?;
        if let Some(status) = process.child.try_wait().map_err(|error| error.to_string())? {
            server.take();
            return Err(format!("TyloInfer Server could not start ({status})."));
        }
        if process.input.is_none() { return Err("TyloInfer Server is stopping.".into()); }
        Ok(())
    }

    /// Closes only Studio's lifecycle pipe and waits before updates or port changes.
    fn stop_owned_server(&self) -> Result<(), String> {
        self.stop_owned_server_with_timeout(SERVER_STOP_TIMEOUT)
    }

    /// Preserves a timed-out child's handle, preventing a second backend on retry.
    fn stop_owned_server_with_timeout(&self, timeout: Duration) -> Result<(), String> {
        let target = {
            let mut server = self.managed_server.lock().map_err(|error| error.to_string())?;
            let Some(process) = server.as_mut() else { return Ok(()); };
            process.input.take();
            (process.child.id(), process.instance_id.clone())
        };
        let deadline = Instant::now() + timeout;
        loop {
            {
                let mut server = self.managed_server.lock().map_err(|error| error.to_string())?;
                let Some(process) = server.as_mut() else { return Ok(()); };
                if process.child.id() != target.0 || process.instance_id != target.1 { return Ok(()); }
                if process.child.try_wait().map_err(|error| format!("could not wait for TyloInfer Server: {error}"))?.is_some() {
                    server.take();
                    return Ok(());
                }
            }
            if Instant::now() >= deadline { break; }
            std::thread::sleep(Duration::from_millis(50).min(deadline.saturating_duration_since(Instant::now())));
        }
        Err(format!("TyloInfer Server did not stop within {} seconds. Studio retained its process handle; finish active requests and retry after it exits.", timeout.as_secs()))
    }

    /// Returns the port of any tracked child, including a child still draining after EOF.
    fn managed_port(&self) -> Result<Option<u16>, String> {
        Ok(self.managed_server.lock().map_err(|error| error.to_string())?.as_ref().map(|server| server.port))
    }

    /// Distinguishes an unusable draining process from an active managed connection.
    fn server_stopping(&self) -> bool {
        self.managed_server.lock().ok().is_some_and(|server| server.as_ref().is_some_and(|server| server.input.is_none()))
    }

    /// Identifies a draining owned process so it cannot be accepted as an external service.
    fn tracks(&self, server: &ServerIdentity) -> bool {
        self.managed_server.lock().ok().is_some_and(|process| process.as_ref().is_some_and(
            |process| process.child.id() == server.pid && process.instance_id == server.instance_id && server.launch_mode == "studio"))
    }

    /// Requires the spawned PID, instance ID, launch mode, and an open lifecycle pipe.
    fn owns(&self, server: &ServerIdentity) -> bool {
        self.managed_server.lock().ok().is_some_and(|process| process.as_ref().is_some_and(
            |process| process.child.id() == server.pid && process.instance_id == server.instance_id
                && process.input.is_some() && server.launch_mode == "studio"))
    }

    /// Restricts external-service consent to the exact endpoint and verified identity.
    fn accepts(&self, url: &str, server: &ServerIdentity) -> bool {
        self.external_server.lock().ok().is_some_and(|accepted| accepted.as_ref().is_some_and(
            |(endpoint, identity)| endpoint == url && identity == server))
    }

    /// Prevents further spawns before closing and draining Studio's lifecycle pipe on exit.
    fn shutdown(&self) -> Result<(), String> {
        self.shutting_down.store(true, Ordering::SeqCst);
        self.stop_owned_server()
    }
}

fn app_data_dir(app: &AppHandle) -> Result<PathBuf, String> {
    let path = app
        .path()
        .app_local_data_dir()
        .map_err(|error| error.to_string())?;
    fs::create_dir_all(&path).map_err(|error| error.to_string())?;
    Ok(path)
}

fn normalized_http_url(value: &str, field: &str) -> Result<String, String> {
    let trimmed = value.trim().trim_end_matches('/');
    let parsed = Url::parse(trimmed).map_err(|error| format!("invalid {field}: {error}"))?;
    if !matches!(parsed.scheme(), "http" | "https") || parsed.host_str().is_none() {
        return Err(format!("{field} must be an HTTP or HTTPS URL"));
    }
    if !parsed.username().is_empty() || parsed.password().is_some() {
        return Err(format!("{field} must not contain credentials"));
    }
    Ok(trimmed.to_string())
}

fn validate_config(mut config: StudioConfig) -> Result<StudioConfig, String> {
    config.remote_url = normalized_http_url(&config.remote_url, "remote_url")?;
    if config.local_service_port == 0 {
        return Err("local_service_port must be greater than zero".into());
    }
    Ok(config)
}

fn config_path(app: &AppHandle) -> Result<PathBuf, String> {
    Ok(app_data_dir(app)?.join(CONFIG_FILE))
}

fn load_config(app: &AppHandle) -> Result<StudioConfig, String> {
    let path = config_path(app)?;
    if !path.exists() {
        return Ok(StudioConfig::default());
    }
    let bytes = fs::read(path).map_err(|error| error.to_string())?;
    let config = serde_json::from_slice(&bytes).map_err(|error| error.to_string())?;
    validate_config(config)
}

fn save_config(app: &AppHandle, config: &StudioConfig) -> Result<(), String> {
    save_config_at(&config_path(app)?, config)
}

/// Replaces configuration atomically so failed writes preserve the previous endpoint.
fn save_config_at(path: &Path, config: &StudioConfig) -> Result<(), String> {
    let bytes = serde_json::to_vec_pretty(config).map_err(|error| error.to_string())?;
    let temporary = path.with_extension(format!("{}.tmp", uuid::Uuid::new_v4()));
    let result = (|| -> std::io::Result<()> {
        let mut output = OpenOptions::new().create_new(true).write(true).open(&temporary)?;
        output.write_all(&bytes)?;
        output.sync_all()?;
        drop(output);
        fs::rename(&temporary, path)
    })();
    if result.is_err() { let _ = fs::remove_file(temporary); }
    result.map_err(|error| format!("could not save Studio configuration: {error}"))
}

fn local_service_url(config: &StudioConfig) -> String {
    format!("http://127.0.0.1:{}", config.local_service_port)
}

fn service_url(config: &StudioConfig) -> String {
    match config.mode {
        RuntimeMode::Local => local_service_url(config),
        RuntimeMode::Remote => config.remote_url.clone(),
    }
}

fn is_loopback_service(base_url: &str) -> bool {
    let Ok(url) = Url::parse(base_url) else { return false; };
    match url.host() {
        Some(url::Host::Domain(host)) => host.eq_ignore_ascii_case("localhost"),
        Some(url::Host::Ipv4(address)) => address.is_loopback(),
        Some(url::Host::Ipv6(address)) => address.is_loopback(),
        None => false,
    }
}

fn server_client(base_url: &str, timeout: Duration) -> Result<reqwest::Client, String> {
    let mut builder = reqwest::Client::builder().timeout(timeout);
    if is_loopback_service(base_url) {
        builder = builder.no_proxy();
    }
    builder.build().map_err(|error| error.to_string())
}

/// Checks the single listener shared by the native OpenAI and Anthropic APIs.
fn check_local_port(port: u16) -> Result<(), String> {
    std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, port))
        .map_err(|error| match error.kind() {
            std::io::ErrorKind::AddrInUse => format!("The API port {port} is already in use. Close the other service or choose a different port."),
            std::io::ErrorKind::PermissionDenied => format!("Permission denied for the API port {port}. Choose a different port."),
            _ => format!("Could not use the API port {port}: {error}"),
        })?;
    Ok(())
}

fn startup_failure(log_path: &Path, offset: u64, summary: &str) -> String {
    let detail = (|| -> std::io::Result<String> {
        let mut log = File::open(log_path)?;
        let end = log.metadata()?.len();
        log.seek(SeekFrom::Start(offset.max(end.saturating_sub(8192))))?;
        let mut bytes = Vec::new();
        log.take(8192).read_to_end(&mut bytes)?;
        Ok(String::from_utf8_lossy(&bytes).trim().to_string())
    })().unwrap_or_default();
    if detail.is_empty() {
        format!("{summary}\nServer log: {}", log_path.display())
    } else {
        format!("{summary}\n\n{detail}\n\nServer log: {}", log_path.display())
    }
}

async fn is_mfq_server(client: &reqwest::Client, base_url: &str) -> bool {
    let response = match client.get(format!("{base_url}/health")).send().await {
        Ok(response) if response.status().is_success() => response,
        _ => return false,
    };
    let payload = match response.json::<serde_json::Value>().await {
        Ok(payload) => payload,
        Err(_) => return false,
    };
    payload.get("service").and_then(|value| value.as_str()) == Some("tyloinfer-server")
}

async fn probe_server(client: &reqwest::Client, base_url: &str) -> Result<Option<ServerIdentity>, String> {
    if !is_mfq_server(client, base_url).await { return Ok(None); }
    let response = client.get(format!("{base_url}/health/identity")).send().await
        .map_err(|_| "Could not verify the TyloInfer Server instance. Restart or update the server.".to_string())?;
    if !response.status().is_success() {
        return Err("This TyloInfer Server does not support Studio's instance handshake. Update it or stop it so Studio can start its bundled backend.".into());
    }
    let server: ServerIdentity = response.json().await
        .map_err(|_| "TyloInfer Server returned an invalid instance identity.".to_string())?;
    server.validate()?;
    Ok(Some(server))
}

fn with_server_credential(request: reqwest::RequestBuilder, token: &str) -> reqwest::RequestBuilder {
    let token = token.trim();
    if token.is_empty() { request } else { request.bearer_auth(token) }
}

fn open_log(path: &Path) -> Result<File, String> {
    if path
        .metadata()
        .map(|metadata| metadata.len() >= LOG_ROTATE_BYTES)
        .unwrap_or(false)
    {
        let backup = path.with_extension("log.1");
        if backup.exists() {
            fs::remove_file(&backup).map_err(|error| error.to_string())?;
        }
        fs::rename(path, backup).map_err(|error| error.to_string())?;
    }
    OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|error| error.to_string())
}

fn configure_detached(command: &mut Command) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.creation_flags(CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW);
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
}

/// Resolves the native server from the release resource layout or a development build.
fn server_program(app: &AppHandle) -> Result<PathBuf, String> {
    let resource_dir = app.path().resource_dir().ok();
    let repository = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let explicit = std::env::var_os(native_server::EXECUTABLE_OVERRIDE);
    native_server::discover(resource_dir.as_deref(), &repository, cfg!(debug_assertions), explicit.as_deref())
}

/// Uses the packaged web UI, or the built frontend in a development checkout when present.
fn server_web_root(app: &AppHandle) -> Option<PathBuf> {
    if let Ok(resource_dir) = app.path().resource_dir() {
        let web_root = resource_dir.join("web-ui");
        if web_root.is_dir() { return Some(web_root); }
    }
    let web_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../dist");
    (cfg!(debug_assertions) && web_root.join("index.html").is_file()).then_some(web_root)
}

fn packaged_resource(app: &AppHandle, name: &str) -> Option<PathBuf> {
    let resource_dir = app.path().resource_dir().ok()?;
    let candidate = resource_dir.join(name);
    candidate.is_file().then_some(candidate)
}

#[cfg(target_os = "macos")]
fn packaged_framework(name: &str) -> Option<PathBuf> {
    let executable = std::env::current_exe().ok()?;
    let contents = executable.parent()?.parent()?;
    let candidate = contents.join("Frameworks").join(name);
    candidate.is_file().then_some(candidate)
}

async fn status_for(app: &AppHandle, config: StudioConfig) -> Result<StudioStatus, String> {
    app.state::<StudioState>().reap_exited_server();
    let service_url = service_url(&config);
    let client = server_client(&service_url, Duration::from_secs(2))?;
    let probe = probe_server(&client, &service_url).await;
    let (server, mut connection_error) = match probe {
        Ok(server) => (server, None),
        Err(error) => (None, Some(error)),
    };
    let state = app.state::<StudioState>();
    if server.as_ref().is_some_and(|server| state.tracks(server)) && state.server_stopping() {
        connection_error = Some("Studio's server is stopping. Wait for it to exit before reconnecting.".into());
    }
    let owned = server.as_ref().is_some_and(|server| state.owns(server));
    let accepted = server.as_ref().is_some_and(|server| state.accepts(&service_url, server));
    let connection = if connection_error.is_some() { "incompatible" }
        else if owned { "managed" }
        else if server.is_some() && accepted { "external" }
        else if server.is_some() { "confirmation_required" }
        else { "offline" };
    let reachable = connection_error.is_none() && (owned || accepted);
    let managed_pid = server.as_ref().filter(|_| owned).map(|server| server.pid);
    Ok(StudioStatus {
        config,
        service_url,
        reachable,
        managed_pid,
        connection,
        server,
        connection_error,
    })
}

/// Waits off the async executor for lifecycle EOF to drain an owned native process.
async fn stop_local(app: &AppHandle) -> Result<(), String> {
    let app = app.clone();
    tokio::task::spawn_blocking(move || app.state::<StudioState>().stop_owned_server())
        .await.map_err(|error| error.to_string())?
}

/// Starts only the native server and accepts readiness from the exact spawned instance.
async fn start_local(app: &AppHandle, config: StudioConfig) -> Result<StudioStatus, String> {
    let state = app.state::<StudioState>();
    state.reap_exited_server();
    let service_url = local_service_url(&config);
    let client = server_client(&service_url, Duration::from_millis(500))?;
    if let Some(server) = probe_server(&client, &service_url).await? {
        if !state.owns(&server) {
            return Err("An independently managed or stopping TyloInfer Server is already running. Connect to an external service explicitly, or wait for Studio's backend to stop.".into());
        }
        return status_for(app, config).await;
    }
    if state.managed_port()?.is_some() {
        return Err("Studio's server is already starting or stopping. Retry after it exits; a second backend will not be launched.".into());
    }

    let data_dir = app_data_dir(app)?;
    check_local_port(config.local_service_port)?;
    let program = server_program(app)?;
    let web_root = server_web_root(app);
    let instance_id = uuid::Uuid::new_v4().to_string();
    let mut command = native_server::command(&program, &data_dir, config.local_service_port,
        &instance_id, web_root.as_deref())?;
    if let Some(metallib) = packaged_resource(app, "mlx.metallib") {
        command.env("TYLOINFER_MLX_METALLIB", metallib);
    }
    #[cfg(target_os = "macos")]
    if let Some(video_library) = packaged_framework("libmfq_avfoundation_video.dylib") {
        command.env("TYLOINFER_AVFOUNDATION_VIDEO_LIBRARY", video_library);
    }
    let log_path = data_dir.join(LOG_FILE);
    let stdout = open_log(&log_path)?;
    let log_offset = stdout.metadata().map_err(|error| error.to_string())?.len();
    let stderr = stdout.try_clone().map_err(|error| error.to_string())?;
    command.stdin(Stdio::piped()).stdout(Stdio::from(stdout)).stderr(Stdio::from(stderr));
    configure_detached(&mut command);
    let result: Result<StudioStatus, String> = async {
        let pid = state.spawn_server(&mut command, &instance_id, config.local_service_port)?;
        let deadline = tokio::time::Instant::now() + Duration::from_secs(30 * 60);
        loop {
            state.check_starting_server(pid)?;
            if let Some(server) = probe_server(&client, &service_url).await? {
                if !state.owns(&server) {
                    return Err("Another server appeared on the port during startup. Studio did not connect to it.".into());
                }
                let status = status_for(app, config.clone()).await?;
                if status.connection == "managed" { return Ok(status); }
                return Err("TyloInfer Server stopped or changed identity before startup completed.".into());
            }
            if tokio::time::Instant::now() >= deadline {
                return Err("TyloInfer Server did not become ready within 30 minutes.".into());
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
    }.await;
    match result {
        Ok(status) => Ok(status),
        Err(error) => {
            let cleanup = stop_local(app).await;
            let summary = match cleanup {
                Ok(()) => error,
                Err(cleanup) => format!("{error}\nCleanup: {cleanup}"),
            };
            Err(startup_failure(&log_path, log_offset, &summary))
        }
    }
}

/// Applies endpoint changes and rolls an owned process back if startup or persistence fails.
async fn apply_config(app: &AppHandle, config: StudioConfig, start_requested: bool) -> Result<StudioStatus, String> {
    let previous = load_config(app)?;
    if matches!(config.mode, RuntimeMode::Remote) {
        let status = status_for(app, config.clone()).await?;
        save_config(app, &config)?;
        return Ok(status);
    }
    let state = app.state::<StudioState>();
    state.reap_exited_server();
    let owned_port = state.managed_port()?;
    if previous.local_service_port != config.local_service_port && owned_port.is_none() {
        let previous_url = local_service_url(&previous);
        let client = server_client(&previous_url, Duration::from_secs(2))?;
        if probe_server(&client, &previous_url).await?.is_some() {
            return Err("This server is managed outside Studio. Change its port through its own service settings.".into());
        }
    }
    let restart = owned_port.is_some_and(|port| port != config.local_service_port || state.server_stopping());
    if restart {
        if owned_port != Some(config.local_service_port) { check_local_port(config.local_service_port)?; }
        server_program(app)?;
        stop_local(app).await?;
    }
    let had_server = state.managed_port()?.is_some();
    let result: Result<StudioStatus, String> = async {
        let current = status_for(app, config.clone()).await?;
        let status = if restart || start_requested || current.connection == "offline" {
            start_local(app, config.clone()).await?
        } else { current };
        save_config(app, &config)?;
        Ok(status)
    }.await;
    match result {
        Ok(status) => Ok(status),
        Err(error) => {
            if !had_server && !state.server_stopping() {
                if let Err(cleanup) = stop_local(app).await {
                    return Err(format!("{error}\nConfiguration was not changed. Cleanup: {cleanup}"));
                }
            }
            if restart {
                if state.managed_port()?.is_some() {
                    return Err(format!("{error}\nConfiguration was not changed. The previous endpoint can be restored after the stopping process exits; retry the port change then."));
                }
                let rollback = StudioConfig { mode: RuntimeMode::Local,
                    local_service_port: owned_port.unwrap(), ..previous };
                return match start_local(app, rollback).await {
                    Ok(_) => Err(format!("{error}\nConfiguration was not changed; Studio restored its server on port {}.", owned_port.unwrap())),
                    Err(rollback) => Err(format!("{error}\nConfiguration was not changed. Could not restore the previous server: {rollback}\nRetry starting the local server after resolving the error.")),
                };
            }
            Err(format!("{error}\nConfiguration was not changed."))
        }
    }
}

/// Returns the verified connection state without granting consent to an external instance.
#[tauri::command]
async fn studio_status(app: AppHandle) -> Result<StudioStatus, String> {
    status_for(&app, load_config(&app)?).await
}

/// Persists settings only after a local restart succeeds; external services are never reconfigured.
#[tauri::command]
async fn studio_configure(
    app: AppHandle,
    state: State<'_, StudioState>,
    config: StudioConfig,
) -> Result<StudioStatus, String> {
    let config = validate_config(config)?;
    let _guard = state.start_lock.lock().await;
    apply_config(&app, config, false).await
}

/// Confirms a particular external instance after the user reviews its identity and workspace.
#[tauri::command]
async fn studio_connect_existing(app: AppHandle, state: State<'_, StudioState>,
    instance_id: String, workspace_id: String) -> Result<StudioStatus, String> {
    let _guard = state.start_lock.lock().await;
    let config = load_config(&app)?;
    let endpoint = service_url(&config);
    let client = server_client(&endpoint, Duration::from_secs(2))?;
    let server = probe_server(&client, &endpoint).await?
        .ok_or("The server stopped before connecting. Refresh and try again.")?;
    if server.instance_id != instance_id || server.workspace_id != workspace_id {
        return Err("The server instance changed. Refresh and confirm the new server before connecting.".into());
    }
    if state.tracks(&server) && state.server_stopping() {
        return Err("Studio's server is stopping. Wait for it to exit before reconnecting.".into());
    }
    *state.external_server.lock().map_err(|error| error.to_string())? = Some((endpoint, server));
    status_for(&app, config).await
}

/// Starts or reconnects to Studio's owned native process and saves local mode on success.
#[tauri::command]
async fn studio_start_local(
    app: AppHandle,
    state: State<'_, StudioState>,
) -> Result<StudioStatus, String> {
    let _guard = state.start_lock.lock().await;
    let mut config = load_config(&app)?;
    config.mode = RuntimeMode::Local;
    apply_config(&app, config, true).await
}

/// Registers a selected folder with the confirmed local server using the stored credential.
#[tauri::command]
async fn studio_select_model_directory(
    app: AppHandle,
    initial_directory: String,
) -> Result<Option<SelectedModelDirectory>, String> {
    let config = load_config(&app)?;
    if !matches!(config.mode, RuntimeMode::Local) {
        return Err("the native directory picker is available only for the local server".into());
    }
    let selected = rfd::AsyncFileDialog::new()
        .set_title("Select a folder containing TQ models")
        .set_directory(initial_directory)
        .pick_folder()
        .await;
    let Some(directory) = selected else {
        return Ok(None);
    };
    let path = directory
        .path()
        .to_str()
        .ok_or_else(|| "selected model directory is not valid UTF-8".to_string())?;
    let client = server_client(&local_service_url(&config), Duration::from_secs(300))?;
    let status = status_for(&app, config.clone()).await?;
    if !status.reachable { return Err("Connect to and confirm the server before registering a model directory.".into()); }
    let server = status.server.ok_or("The server is unavailable.")?;
    let token = studio_credential_get()?;
    let request = client
        .post(format!(
            "{}/api/v1/models/directories/register",
            local_service_url(&config)
        ))
        .header("X-TyloInfer-Instance-ID", server.instance_id)
        .json(&serde_json::json!({"path": path}));
    let response = with_server_credential(request, &token)
        .send()
        .await
        .map_err(|error| format!("failed to register model directory: {error}"))?;
    if !response.status().is_success() {
        let status = response.status();
        let detail = response.text().await.unwrap_or_default();
        return Err(format!(
            "model directory registration failed ({status}): {detail}"
        ));
    }
    let registered = response
        .json::<RegisteredModelList>()
        .await
        .map_err(|error| format!("invalid model registration response: {error}"))?;
    Ok(Some(SelectedModelDirectory {
        path: path.to_string(),
        names: registered
            .data
            .into_iter()
            .map(|model| model.name)
            .collect(),
    }))
}

/// Shows a native confirmation dialog for frontend operations.
#[tauri::command]
fn studio_confirm(message: String) -> bool {
    matches!(
        rfd::MessageDialog::new()
            .set_title("TyloInfer")
            .set_description(message)
            .set_buttons(rfd::MessageButtons::YesNo)
            .show(),
        rfd::MessageDialogResult::Yes
    )
}

/// Reads the server credential from the platform credential store.
#[tauri::command]
fn studio_credential_get() -> Result<String, String> {
    let entry = keyring::Entry::new(CREDENTIAL_SERVICE, CREDENTIAL_ACCOUNT)
        .map_err(|error| error.to_string())?;
    match entry.get_password() {
        Ok(value) => Ok(value),
        Err(keyring::Error::NoEntry) => Ok(String::new()),
        Err(error) => Err(error.to_string()),
    }
}

/// Saves or clears the server credential without writing secrets to Studio configuration.
#[tauri::command]
fn studio_credential_set(token: String) -> Result<(), String> {
    let entry = keyring::Entry::new(CREDENTIAL_SERVICE, CREDENTIAL_ACCOUNT)
        .map_err(|error| error.to_string())?;
    let token = token.trim();
    if token.is_empty() {
        match entry.delete_credential() {
            Ok(()) | Err(keyring::Error::NoEntry) => Ok(()),
            Err(error) => Err(error.to_string()),
        }
    } else {
        entry.set_password(token).map_err(|error| error.to_string())
    }
}

/// Opens a validated HTTPS model or release link with the platform browser.
#[tauri::command]
fn studio_open_external(url: String) -> Result<(), String> {
    let parsed = Url::parse(&url).map_err(|error| format!("invalid external URL: {error}"))?;
    let host = parsed.host_str().unwrap_or_default().to_ascii_lowercase();
    if parsed.scheme() != "https"
        || !parsed.username().is_empty()
        || parsed.password().is_some()
        || parsed.port_or_known_default() != Some(443)
        || !matches!(
            host.as_str(),
            "github.com"
                | "www.github.com"
                | "huggingface.co"
                | "www.huggingface.co"
                | "modelscope.cn"
                | "www.modelscope.cn"
                | "www.deepseek.com"
                | "opencode.ai"
                | "code.claude.com"
                | "openclaw.ai"
        )
    {
        return Err("external links are limited to trusted TyloInfer model and release hosts".into());
    }
    #[cfg(target_os = "macos")]
    let mut command = Command::new("/usr/bin/open");
    #[cfg(target_os = "windows")]
    let mut command = {
        let mut value = Command::new("rundll32.exe");
        value.arg("url.dll,FileProtocolHandler");
        value
    };
    #[cfg(all(unix, not(target_os = "macos")))]
    let mut command = Command::new("xdg-open");
    command
        .arg(url)
        .spawn()
        .map(|_| ())
        .map_err(|error| format!("could not open external link: {error}"))
}

fn main() {
    if updater::maybe_run_update_helper() {
        return;
    }
    tauri::Builder::default()
        .manage(StudioState::default())
        .manage(updater::UpdateState::default())
        .setup(|app| {
            #[cfg(target_os = "windows")]
            if let Some(window) = app.get_webview_window("main") {
                windows_icon::set(&window)?;
                window.set_decorations(false)?;
            }
            #[cfg(not(target_os = "windows"))]
            let _ = app;
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            studio_custom_titlebar,
            studio_status,
            studio_configure,
            studio_start_local,
            studio_connect_existing,
            studio_select_model_directory,
            studio_confirm,
            studio_credential_get,
            studio_credential_set,
            studio_open_external,
            updater::studio_update_status,
            updater::studio_update_progress,
            updater::studio_update_set_automatic,
            updater::studio_update_set_download,
            updater::studio_update_download,
            updater::studio_update_install,
            updater::studio_update_delete
        ])
        .build(tauri::generate_context!())
        .expect("TyloInfer failed")
        .run(|app, event| {
            if let tauri::RunEvent::Exit = event {
                let state = app.state::<StudioState>();
                if let Err(error) = state.shutdown() {
                    eprintln!("{error}");
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    fn pipe_command() -> Command {
        let mut command = Command::new("/bin/cat");
        command.stdin(Stdio::piped()).stdout(Stdio::null());
        command
    }

    #[cfg(windows)]
    fn pipe_command() -> Command {
        let root = std::env::var_os("SystemRoot").expect("Windows system directory");
        let mut command = Command::new(PathBuf::from(root).join("System32/sort.exe"));
        command.stdin(Stdio::piped()).stdout(Stdio::null());
        configure_detached(&mut command);
        command
    }

    fn wait_for_pipe_child(child: &mut Child) {
        for _ in 0..100 {
            if let Some(status) = child.try_wait().unwrap() {
                assert!(status.success());
                return;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        let _ = child.kill();
        let _ = child.wait();
        panic!("owned server did not receive lifecycle EOF");
    }

    #[test]
    fn update_shutdown_waits_for_owned_server_and_preserves_external_server() {
        let state = StudioState::default();
        let mut external = pipe_command().spawn().unwrap();
        let pid = state.spawn_server(&mut pipe_command(), "owned-instance", 8090).unwrap();
        assert!(state.managed_server.lock().unwrap().as_ref().unwrap().child.stdin.is_none());
        state.check_starting_server(pid).unwrap();
        assert!(state.spawn_server(&mut pipe_command(), "duplicate", 8092).is_err());
        state.stop_owned_server().unwrap();
        assert!(state.managed_server.lock().unwrap().is_none());
        state.stop_owned_server().unwrap();
        assert!(external.try_wait().unwrap().is_none());
        drop(external.stdin.take());
        wait_for_pipe_child(&mut external);
    }

    #[test]
    fn exit_shutdown_prevents_new_launches() {
        let state = StudioState::default();
        state.spawn_server(&mut pipe_command(), "owned-instance", 8090).unwrap();
        state.shutdown().unwrap();
        assert!(state.managed_port().unwrap().is_none());
        assert!(state.spawn_server(&mut pipe_command(), "late-start", 8090).unwrap_err().contains("shutting down"));
    }

    #[test]
    #[ignore = "subprocess fixture for the lifecycle timeout test"]
    fn delayed_exit_fixture() {
        let mut input = Vec::new();
        std::io::stdin().read_to_end(&mut input).unwrap();
        std::thread::sleep(Duration::from_millis(500));
    }

    #[test]
    fn failed_startup_retains_a_draining_child_and_blocks_duplicates_until_exit() {
        let state = StudioState::default();
        let instance = uuid::Uuid::new_v4().to_string();
        let mut command = Command::new(std::env::current_exe().unwrap());
        command.args(["--exact", "tests::delayed_exit_fixture", "--ignored"])
            .stdin(Stdio::piped()).stdout(Stdio::null()).stderr(Stdio::null());
        configure_detached(&mut command);
        let pid = state.spawn_server(&mut command, &instance, 8090).unwrap();
        let server = ServerIdentity { pid, instance_id: instance, launch_mode: "studio".into(), ..test_identity() };
        assert!(state.owns(&server));
        let error = state.stop_owned_server_with_timeout(Duration::ZERO).unwrap_err();
        assert!(error.contains("retained its process handle"));
        assert_eq!(state.managed_port().unwrap(), Some(8090));
        assert!(state.tracks(&server));
        assert!(!state.owns(&server));
        assert!(state.server_stopping());
        assert!(state.check_starting_server(pid).is_err());
        assert!(state.spawn_server(&mut pipe_command(), "duplicate", 8092).is_err());
        state.stop_owned_server().unwrap();
        state.spawn_server(&mut pipe_command(), "retry", 8092).unwrap();
        state.stop_owned_server().unwrap();
    }

    fn startup_test_dir() -> PathBuf {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let nonce = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let sequence = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("mfq-startup-test-{}-{nonce}-{sequence}", std::process::id()));
        fs::create_dir(&path).unwrap();
        path
    }

    async fn launch_native_smoke(state: &StudioState, executable: &Path, directory: &Path, port: u16) -> Result<ServerIdentity, String> {
        let instance = uuid::Uuid::new_v4().to_string();
        let mut command = native_server::command(executable, directory, port, &instance, None)?;
        let log_path = directory.join(LOG_FILE);
        let log = open_log(&log_path)?;
        let offset = log.metadata().map_err(|error| error.to_string())?.len();
        let stderr = log.try_clone().map_err(|error| error.to_string())?;
        command.env("TYLOINFER_SERVER_API_KEY", "studio-smoke-test-key")
            .stdin(Stdio::piped()).stdout(Stdio::from(log)).stderr(Stdio::from(stderr));
        configure_detached(&mut command);
        let pid = state.spawn_server(&mut command, &instance, port)?;
        let endpoint = format!("http://127.0.0.1:{port}");
        let client = server_client(&endpoint, Duration::from_millis(500))?;
        let result: Result<ServerIdentity, String> = async {
            let deadline = tokio::time::Instant::now() + Duration::from_secs(20);
            loop {
                state.check_starting_server(pid)?;
                if let Some(server) = probe_server(&client, &endpoint).await? {
                    if !state.owns(&server) { return Err("native smoke instance identity mismatch".into()); }
                    return Ok(server);
                }
                if tokio::time::Instant::now() >= deadline { return Err("native smoke startup timed out".into()); }
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        }.await;
        result.map_err(|error| startup_failure(&log_path, offset, &error))
    }

    #[test]
    #[ignore = "requires TYLOINFER_TEST_SERVER_EXECUTABLE pointing to a built native server"]
    fn native_server_lifecycle_smoke() {
        let executable = PathBuf::from(std::env::var_os("TYLOINFER_TEST_SERVER_EXECUTABLE").expect("built native server executable"));
        let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("target")
            .join(format!("native-server-smoke-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&directory).unwrap();
        let available_port = || {
            std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap().local_addr().unwrap().port()
        };
        let first_port = available_port();
        let state = StudioState::default();
        let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        let result: Result<(), String> = runtime.block_on(async {
            let first = launch_native_smoke(&state, &executable, &directory, first_port).await?;
            let endpoint = format!("http://127.0.0.1:{first_port}");
            let client = server_client(&endpoint, Duration::from_secs(2))?;
            let unauthorized = client.get(format!("{endpoint}/api/v1/models")).send().await.map_err(|error| error.to_string())?;
            if unauthorized.status() != reqwest::StatusCode::UNAUTHORIZED { return Err("native server did not enforce the inherited root credential".into()); }
            let authenticated = with_server_credential(client.get(format!("{endpoint}/api/v1/models")), "studio-smoke-test-key")
                .header("X-TyloInfer-Instance-ID", &first.instance_id).send().await.map_err(|error| error.to_string())?;
            if !authenticated.status().is_success() { return Err("native server rejected the authenticated Studio request".into()); }
            state.stop_owned_server()?;
            check_local_port(first_port)?;
            let old_port = std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, first_port)).map_err(|error| error.to_string())?;
            let second_port = available_port();
            let second = launch_native_smoke(&state, &executable, &directory, second_port).await?;
            drop(old_port);
            if first.instance_id == second.instance_id || first.workspace_id != second.workspace_id {
                return Err("native port restart did not preserve workspace identity with a fresh instance".into());
            }
            state.stop_owned_server()?;
            check_local_port(second_port)?;
            Ok(())
        });
        let cleanup = state.stop_owned_server();
        eprintln!("Native smoke artifacts: {}", directory.display());
        result.unwrap();
        cleanup.unwrap();
    }

    fn test_identity() -> ServerIdentity {
        ServerIdentity { service: "tyloinfer-server".into(), protocol_version: "1.0".into(),
            studio_protocol_version: "1.0".into(), version: env!("TYLOINFER_SERVER_VERSION").into(),
            instance_id: uuid::Uuid::new_v4().to_string(), workspace_id: "a".repeat(64), pid: 123,
            launch_mode: "cli".into() }
    }

    #[test]
    fn identity_requires_compatible_versions_and_real_instance_fields() {
        let valid = test_identity();
        valid.validate().unwrap();
        for invalid in [ServerIdentity { version: "0.0.0".into(), ..valid.clone() },
            ServerIdentity { protocol_version: "2.0".into(), ..valid.clone() },
            ServerIdentity { studio_protocol_version: String::new(), ..valid.clone() },
            ServerIdentity { instance_id: String::new(), ..valid.clone() },
            ServerIdentity { workspace_id: "invalid".into(), ..valid.clone() },
            ServerIdentity { pid: 0, ..valid.clone() },
            ServerIdentity { launch_mode: "unknown".into(), ..valid.clone() },
            ServerIdentity::default()] {
            assert!(invalid.validate().is_err(), "{invalid:?}");
        }
    }

    #[test]
    fn external_confirmation_is_bound_to_endpoint_instance_and_workspace() {
        let state = StudioState::default();
        let server = test_identity();
        let url = "http://127.0.0.1:8090";
        assert!(!state.accepts(url, &server));
        *state.external_server.lock().unwrap() = Some((url.into(), server.clone()));
        assert!(state.accepts(url, &server));
        assert!(!state.accepts("http://127.0.0.1:8092", &server));
        assert!(!state.accepts(url, &ServerIdentity { instance_id: uuid::Uuid::new_v4().to_string(), ..server.clone() }));
        assert!(!state.accepts(url, &ServerIdentity { workspace_id: "b".repeat(64), ..server.clone() }));
        assert!(!state.owns(&server));
    }

    #[test]
    fn ownership_needs_both_spawned_pid_and_instance_id_and_clears_after_exit() {
        let state = StudioState::default();
        let mut server = test_identity();
        server.launch_mode = "studio".into();
        server.pid = state.spawn_server(&mut pipe_command(), &server.instance_id, 8090).unwrap();
        assert!(state.owns(&server));
        assert!(!state.owns(&ServerIdentity { instance_id: uuid::Uuid::new_v4().to_string(), ..server.clone() }));
        assert!(!state.owns(&ServerIdentity { pid: server.pid.wrapping_add(1), ..server.clone() }));
        assert!(!state.owns(&ServerIdentity { launch_mode: "cli".into(), ..server.clone() }));
        {
            let mut owned = state.managed_server.lock().unwrap();
            let child = &mut owned.as_mut().unwrap().child;
            child.kill().unwrap();
            child.wait().unwrap();
        }
        state.reap_exited_server();
        assert!(!state.owns(&server));
        assert!(state.managed_port().unwrap().is_none());
    }

    #[test]
    fn only_loopback_server_connections_bypass_proxies() {
        for url in ["http://localhost:8090", "https://LOCALHOST:8090", "http://127.0.0.1:8090", "http://127.0.0.2:8090", "http://[::1]:8090"] {
            assert!(is_loopback_service(url), "{url}");
        }
        for url in ["https://example.com", "http://localhost.example.com", "http://192.168.1.2:8090", "http://[::2]:8090", "not a URL"] {
            assert!(!is_loopback_service(url), "{url}");
        }
    }

    #[test]
    fn startup_failure_reports_only_this_launch_and_bounds_log_output() {
        let directory = startup_test_dir();
        let path = directory.join(LOG_FILE);
        let previous = "previous-launch-private-text\n";
        fs::write(&path, format!("{previous}new launch: library not found\n")).unwrap();
        let error = startup_failure(&path, previous.len() as u64, "Server exited.");
        assert!(error.contains("Server exited."));
        assert!(error.contains("library not found"));
        assert!(!error.contains("previous-launch-private-text"));
        fs::write(&path, format!("{}\nlast error", "x".repeat(20000))).unwrap();
        let error = startup_failure(&path, 0, "Server exited.");
        assert!(error.contains("last error"));
        assert!(error.len() < 8500);
        assert!(startup_failure(&path, 30000, "Server exited.").contains("Server log:"));
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn startup_failure_keeps_the_summary_when_the_log_is_missing() {
        let directory = startup_test_dir();
        let error = startup_failure(&directory.join(LOG_FILE), 0, "Server exited.");
        assert!(error.starts_with("Server exited.\nServer log:"));
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn startup_failure_after_log_rotation_does_not_expose_the_previous_log() {
        use std::io::Write;
        let directory = startup_test_dir();
        let path = directory.join(LOG_FILE);
        fs::write(&path, "previous-launch-private-text").unwrap();
        OpenOptions::new().write(true).open(&path).unwrap().set_len(LOG_ROTATE_BYTES).unwrap();
        let mut log = open_log(&path).unwrap();
        let offset = log.metadata().unwrap().len();
        assert_eq!(offset, 0);
        writeln!(log, "current launch error").unwrap();
        drop(log);
        let error = startup_failure(&path, offset, "Server exited.");
        assert!(error.contains("current launch error"));
        assert!(!error.contains("previous-launch-private-text"));
        assert_eq!(path.with_extension("log.1").metadata().unwrap().len(), LOG_ROTATE_BYTES);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn startup_checks_only_the_single_native_api_port() {
        let occupied = std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap();
        let occupied_port = occupied.local_addr().unwrap().port();
        let available = std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap();
        let available_port = available.local_addr().unwrap().port();
        drop(available);
        check_local_port(available_port).unwrap();
        let error = check_local_port(occupied_port).unwrap_err();
        assert!(error.contains(&format!("API port {occupied_port} is already in use")));
    }

    #[test]
    fn configuration_replace_keeps_one_complete_endpoint_file() {
        let directory = startup_test_dir();
        let path = directory.join(CONFIG_FILE);
        let previous = StudioConfig::default();
        save_config_at(&path, &previous).unwrap();
        let next = StudioConfig { local_service_port: 8091, ..previous };
        save_config_at(&path, &next).unwrap();
        assert_eq!(serde_json::from_slice::<StudioConfig>(&fs::read(&path).unwrap()).unwrap(), next);
        assert_eq!(fs::read_dir(&directory).unwrap().count(), 1);
        fs::remove_file(path).unwrap();
        fs::remove_dir(directory).unwrap();
    }

    #[cfg(windows)]
    #[test]
    fn configuration_save_failure_keeps_previous_settings_and_removes_the_temporary_file() {
        use std::os::windows::fs::OpenOptionsExt;
        let directory = startup_test_dir();
        let path = directory.join(CONFIG_FILE);
        let previous = StudioConfig::default();
        save_config_at(&path, &previous).unwrap();
        let locked = OpenOptions::new().read(true).share_mode(0).open(&path).unwrap();
        let next = StudioConfig { local_service_port: 8091, ..previous.clone() };
        assert!(save_config_at(&path, &next).is_err());
        drop(locked);
        assert_eq!(serde_json::from_slice::<StudioConfig>(&fs::read(&path).unwrap()).unwrap(), previous);
        assert_eq!(fs::read_dir(&directory).unwrap().count(), 1);
        save_config_at(&path, &next).unwrap();
        fs::remove_file(path).unwrap();
        fs::remove_dir(directory).unwrap();
    }

    #[cfg(any(target_os = "macos", target_os = "windows"))]
    #[test]
    fn desktop_credentials_use_a_native_store_instead_of_the_mock_backend() {
        let builder = keyring::default::default_credential_builder();
        assert!(!builder.as_any().is::<keyring::mock::MockCredentialBuilder>());
    }

    #[test]
    fn server_requests_use_the_trimmed_credential_without_empty_auth_headers() {
        let client = reqwest::Client::new();
        for path in ["/api/v1/models/directories/register"] {
            let url = format!("http://127.0.0.1:8090{path}");
            let authenticated = with_server_credential(client.post(&url), " test-credential ").build().unwrap();
            assert_eq!(authenticated.headers()[reqwest::header::AUTHORIZATION], "Bearer test-credential");
            assert!(authenticated.headers()[reqwest::header::AUTHORIZATION].is_sensitive());
            for empty in ["", " \n "] {
                let anonymous = with_server_credential(client.post(&url), empty).build().unwrap();
                assert!(!anonymous.headers().contains_key(reqwest::header::AUTHORIZATION));
            }
        }
    }
}
