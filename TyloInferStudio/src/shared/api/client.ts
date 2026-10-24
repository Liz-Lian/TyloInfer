
import type { ApiErrorBody } from './types';

export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly retryable: boolean;
  readonly details: Record<string, unknown>;

  constructor(status: number, body: ApiErrorBody) {
    super(body.error.message);
    this.name = 'ApiError';
    this.status = status;
    this.code = body.error.code;
    this.retryable = body.error.retryable;
    this.details = body.error.details;
  }
}

let apiBaseUrl = '';
let apiToken = '';
let apiServerInstance = '';
const authenticationListeners = new Set<() => void>();
const instanceChangeListeners = new Set<() => void>();
const BROWSER_SERVICE_KEY = 'tyloinfer.studio.service-url';
let browserUrlOverride: string | undefined;

export function onAuthenticationRequired(listener: () => void): () => void {
  authenticationListeners.add(listener);
  return () => { authenticationListeners.delete(listener); };
}

export function onServerInstanceChanged(listener: () => void): () => void {
  instanceChangeListeners.add(listener);
  return () => { instanceChangeListeners.delete(listener); };
}

export function setApiServerInstance(value: string): void {
  apiServerInstance = value;
}

export async function verifyApiToken(token: string, signal?: AbortSignal): Promise<void> {
  const headers = new Headers({ Authorization: `Bearer ${token.trim()}` });
  if (apiServerInstance) headers.set('X-TyloInfer-Instance-ID', apiServerInstance);
  const response = await fetch(apiUrl('/api/v1/runtime/instances'), {
    headers,
    cache: 'no-store',
    signal,
  });
  if (!response.ok) throw await errorFromResponse(response);
}

export function browserServiceUrl(): string {
  if (browserUrlOverride !== undefined) return browserUrlOverride;
  try {
    const value = localStorage.getItem(BROWSER_SERVICE_KEY) ?? '';
    if (!value) return '';
    const url = new URL(value);
    return ['http:', 'https:'].includes(url.protocol) && !url.username && !url.password ? value : '';
  } catch { return ''; }
}

export function setBrowserServiceUrl(value: string): void {
  try {
    if (value) localStorage.setItem(BROWSER_SERVICE_KEY, value);
    else localStorage.removeItem(BROWSER_SERVICE_KEY);
    browserUrlOverride = undefined;
  } catch { browserUrlOverride = value; }
}


export function getApiToken(): string {
  return apiToken;
}


export function setApiBaseUrl(value: string): void {
  apiBaseUrl = value.trim().replace(/\/+$/, '');
}


export function setApiToken(value: string): void {
  apiToken = value.trim();
}


export function getApiBaseUrl(): string {
  return apiBaseUrl;
}


export function apiUrl(path: string): string {
  return `${apiBaseUrl}${path}`;
}


export function runtimeRealtimeUrl(): string {
  const base = apiBaseUrl || window.location.origin;
  const url = new URL('/api/v1/runtime/realtime?mode=audio', base);
  if (apiToken) url.searchParams.set('access_token', apiToken);
  if (apiServerInstance) url.searchParams.set('server_instance', apiServerInstance);
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
  return url.toString();
}


export async function errorFromResponse(response: Response): Promise<ApiError> {
  try {
    return new ApiError(response.status, (await response.json()) as ApiErrorBody);
  } catch {
    return new ApiError(response.status, {
      error: {
        code: `http_${response.status}`,
        message: response.statusText || 'Request failed',
        retryable: false,
        details: {},
      },
    });
  }
}


export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const endpoint = apiBaseUrl;
  const token = apiToken;
  const instance = apiServerInstance;
  const headers = authorizedHeaders(init?.headers);
  if (!headers.has('Content-Type')) headers.set('Content-Type', 'application/json');
  const response = await fetch(apiUrl(path), {
    ...init,
    headers,
  });
  if (!response.ok) {
    const error = await errorFromResponse(response);
    if (error.code === 'server_instance_changed' && endpoint === apiBaseUrl && instance === apiServerInstance && !init?.signal?.aborted) {
      instanceChangeListeners.forEach((listener) => listener());
    }
    if (error.status === 401 && endpoint === apiBaseUrl && token === apiToken && !init?.signal?.aborted) {
      authenticationListeners.forEach((listener) => listener());
    }
    throw error;
  }
  return (await response.json()) as T;
}


export function authorizedHeaders(headers?: HeadersInit): Headers {
  const result = new Headers(headers);
  if (apiToken) result.set('Authorization', `Bearer ${apiToken}`);
  if (apiServerInstance) result.set('X-TyloInfer-Instance-ID', apiServerInstance);
  return result;
}
