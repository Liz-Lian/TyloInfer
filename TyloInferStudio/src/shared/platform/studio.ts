
export type StudioRuntimeMode = 'local' | 'remote';


export interface StudioConfig {
  mode: StudioRuntimeMode;
  remote_url: string;
  local_service_port: number;
}


export interface StudioStatus {
  config: StudioConfig;
  service_url: string;
  reachable: boolean;
  managed_pid: number | null;
  connection?: 'managed' | 'external' | 'confirmation_required' | 'incompatible' | 'offline';
  server?: {
    version: string;
    instance_id: string;
    workspace_id: string;
    pid: number;
    launch_mode: 'studio' | 'cli';
  } | null;
  connection_error?: string | null;
}


export interface StudioReleaseAsset {
  name: string;
  byte_size: number;
  sha256?: string | null;
  download_url: string;
}


export interface StudioRelease {
  version: string;
  tag: string;
  name: string;
  notes: string;
  published_at?: string | null;
  page_url: string;
  prerelease: boolean;
  asset: StudioReleaseAsset | null;
}


export interface InstalledStudioVersion {
  version: string;
  tag: string;
  current: boolean;
  ready: boolean;
  byte_size: number;
  installed_at_epoch_seconds: number;
  notes?: string;
  prerelease?: boolean;
}


export interface StudioUpdateStatus {
  current_version: string;
  current_release: boolean;
  automatic_check: boolean;
  automatic_download: boolean;
  checked_at_epoch_seconds?: number | null;
  update_available: boolean;
  platform_supported: boolean;
  releases_page: string;
  latest?: StudioRelease | null;
  releases: StudioRelease[];
  installed_versions: InstalledStudioVersion[];
  error?: string | null;
}


export interface StudioUpdateProgress {
  tag: string;
  stage: 'downloading' | 'preparing' | string;
  received_bytes: number;
  total_bytes: number;
}

interface TauriInternals {

  invoke<T>(command: string, args?: Record<string, unknown>): Promise<T>;
}

declare global {
  interface Window {
    __TAURI_INTERNALS__?: TauriInternals;
  }
}

function internals(): TauriInternals | null {
  return window.__TAURI_INTERNALS__ ?? null;
}


export function isStudio(): boolean {
  return internals() !== null;
}


export async function studioCustomTitlebar(): Promise<boolean> {
  const tauri = internals();
  return tauri ? tauri.invoke<boolean>('studio_custom_titlebar') : false;
}


export async function studioStatus(): Promise<StudioStatus | null> {
  const tauri = internals();
  return tauri ? tauri.invoke<StudioStatus>('studio_status') : null;
}


export async function configureStudio(config: StudioConfig): Promise<StudioStatus> {
  const tauri = internals();
  if (!tauri) throw new Error('TyloInfer runtime is unavailable');
  return tauri.invoke<StudioStatus>('studio_configure', { config });
}


export async function startLocalStudio(): Promise<StudioStatus> {
  const tauri = internals();
  if (!tauri) throw new Error('TyloInfer runtime is unavailable');
  return tauri.invoke<StudioStatus>('studio_start_local');
}

export async function connectExistingStudio(instanceId: string, workspaceId: string): Promise<StudioStatus> {
  const tauri = internals();
  if (!tauri) throw new Error('TyloInfer runtime is unavailable');
  return tauri.invoke<StudioStatus>('studio_connect_existing', { instanceId, workspaceId });
}


export async function selectLocalModelDirectory(initialDirectory: string): Promise<{ path: string; names: string[] } | null> {
  const tauri = internals();
  if (!tauri) throw new Error('TyloInfer runtime is unavailable');
  return tauri.invoke('studio_select_model_directory', { initialDirectory });
}


export async function studioConfirm(message: string): Promise<boolean> {
  const tauri = internals();
  return tauri
    ? tauri.invoke<boolean>('studio_confirm', { message })
    : window.confirm(message);
}


export async function studioCredential(): Promise<string> {
  const tauri = internals();
  return tauri ? tauri.invoke<string>('studio_credential_get') : '';
}


export async function saveStudioCredential(token: string): Promise<void> {
  const tauri = internals();
  if (!tauri) return;
  await tauri.invoke<void>('studio_credential_set', { token });
}

function trustedExternalUrl(value: string): string {
  const parsed = new URL(value);
  const trustedHosts = new Set([
    'github.com',
    'www.github.com',
    'huggingface.co',
    'www.huggingface.co',
    'modelscope.cn',
    'www.modelscope.cn',
    'www.deepseek.com',
    'opencode.ai',
    'code.claude.com',
    'openclaw.ai',
  ]);
  if (
    parsed.protocol !== 'https:' ||
    parsed.username ||
    parsed.password ||
    (parsed.port && parsed.port !== '443') ||
    !trustedHosts.has(parsed.hostname.toLowerCase())
  ) {
    throw new Error('External links are limited to trusted TyloInfer model and release hosts');
  }
  return parsed.toString();
}


export async function openStudioExternal(url: string): Promise<void> {
  const trusted = trustedExternalUrl(url);
  const tauri = internals();
  if (tauri) {
    await tauri.invoke<void>('studio_open_external', { url: trusted });
    return;
  }
  window.open(trusted, '_blank', 'noopener,noreferrer');
}


export async function studioUpdateStatus(force = false): Promise<StudioUpdateStatus | null> {
  const tauri = internals();
  return tauri ? tauri.invoke<StudioUpdateStatus>('studio_update_status', { force }) : null;
}


export async function studioUpdateProgress(): Promise<StudioUpdateProgress | null> {
  const tauri = internals();
  return tauri
    ? tauri.invoke<StudioUpdateProgress | null>('studio_update_progress')
    : null;
}


export async function setAutomaticStudioUpdates(
  enabled: boolean,
): Promise<StudioUpdateStatus> {
  const tauri = internals();
  if (!tauri) throw new Error('TyloInfer runtime is unavailable');
  return tauri.invoke<StudioUpdateStatus>('studio_update_set_automatic', { enabled });
}

export async function setAutomaticStudioDownloads(enabled: boolean): Promise<StudioUpdateStatus> {
  const tauri = internals();
  if (!tauri) throw new Error('TyloInfer runtime is unavailable');
  return tauri.invoke<StudioUpdateStatus>('studio_update_set_download', { enabled });
}


export async function downloadStudioVersion(tag: string): Promise<InstalledStudioVersion> {
  const tauri = internals();
  if (!tauri) throw new Error('TyloInfer runtime is unavailable');
  return tauri.invoke<InstalledStudioVersion>('studio_update_download', { tag });
}


export async function installStudioVersion(version: string): Promise<void> {
  const tauri = internals();
  if (!tauri) throw new Error('TyloInfer runtime is unavailable');
  await tauri.invoke<void>('studio_update_install', { version });
}


export async function deleteStudioVersion(version: string): Promise<StudioUpdateStatus> {
  const tauri = internals();
  if (!tauri) throw new Error('TyloInfer runtime is unavailable');
  return tauri.invoke<StudioUpdateStatus>('studio_update_delete', { version });
}
