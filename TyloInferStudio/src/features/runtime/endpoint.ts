import { getApiBaseUrl } from '../../shared/api/client';
import { isStudio } from '../../shared/platform/studio';

function serviceEndpointUrl(serviceUrl?: string | null): URL | null {
  try {
    const configured = serviceUrl?.trim() || getApiBaseUrl();
    if (!configured && isStudio()) return null;
    const url = new URL(configured || window.location.origin);
    return ['http:', 'https:'].includes(url.protocol) && !url.username && !url.password ? url : null;
  } catch { return null; }
}

/** Build an SDK base URL from the connected HTTP service or the browser's same-origin server. */
export function openAIEndpoint(serviceUrl?: string | null): string | null {
  const url = serviceEndpointUrl(serviceUrl);
  if (!url) return null;
  const path = url.pathname.replace(/\/+$/, '');
  url.pathname = path.endsWith('/v1') ? path : `${path}/v1`;
  url.search = '';
  url.hash = '';
  return url.toString().replace(/\/$/, '');
}

/** Build a supported separate-port Anthropic endpoint without exposing desktop resource origins. */
export function anthropicEndpoint(serviceUrl: string | null | undefined, port: number | null | undefined): string | null {
  if (port == null || !Number.isInteger(port) || port < 1 || port > 65535) return null;
  const url = serviceEndpointUrl(serviceUrl);
  if (!url) return null;
  url.port = String(port);
  const path = url.pathname.replace(/\/+$/, '').replace(/\/v1$/, '');
  url.pathname = `${path}/v1/messages`;
  url.search = '';
  url.hash = '';
  return url.toString().replace(/\/$/, '');
}
