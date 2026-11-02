import { afterEach, expect, it, vi } from 'vitest';
import { anthropicEndpoint, openAIEndpoint } from './endpoint';
import { setApiBaseUrl } from '../../shared/api/client';

afterEach(() => { setApiBaseUrl(''); vi.unstubAllGlobals(); });

it.each([
  [null, `${window.location.origin}/v1`],
  ['http://localhost:8090/', 'http://localhost:8090/v1'],
  ['https://example.com/v1/', 'https://example.com/v1'],
  ['https://example.com/tyloinfer///', 'https://example.com/tyloinfer/v1'],
  ['https://example.com/tyloinfer/v1?x=1#settings', 'https://example.com/tyloinfer/v1'],
  ['http://[::1]:8090', 'http://[::1]:8090/v1'],
])('produces an SDK base URL from %s without duplicate /v1', (source, expected) => {
  expect(openAIEndpoint(source)).toBe(expected);
});

it('keeps OpenAI /v1 while Anthropic uses the actual separate port and full request path', () => {
  expect(openAIEndpoint('http://127.0.0.1:9000/v1/')).toBe('http://127.0.0.1:9000/v1');
  expect(anthropicEndpoint('http://127.0.0.1:9000/v1/', 9001)).toBe('http://127.0.0.1:9001/v1/messages');
  expect(anthropicEndpoint('https://remote.example/tyloinfer/v1?key=discard#fragment', 9443)).toBe('https://remote.example:9443/tyloinfer/v1/messages');
  expect(anthropicEndpoint('http://[::1]:8090', 8091)).toBe('http://[::1]:8091/v1/messages');
});

it.each([null, undefined, 0, 65536, 1.5])('never invents an endpoint for an invalid or unavailable port: %s', (port) => {
  expect(anthropicEndpoint('', port)).toBeNull();
});

it.each([
  [null, 8091, `${new URL(window.location.origin).protocol}//${new URL(window.location.origin).hostname}:8091/v1/messages`],
  ['http://127.0.0.1:9000/v1/', 9001, 'http://127.0.0.1:9001/v1/messages'],
  ['https://remote.example/tyloinfer/v1?key=discard#fragment', 9443, 'https://remote.example:9443/tyloinfer/v1/messages'],
  ['http://[::1]:8090', 8091, 'http://[::1]:8091/v1/messages'],
])('also exposes the complete Anthropic request URL for %s', (service, port, expected) => {
  expect(anthropicEndpoint(service, port)).toBe(expected);
});

it('uses the actual same-origin browser port when no API base is configured', () => {
  vi.stubGlobal('window', { location: { origin: 'http://127.0.0.1:59421' } });
  expect(openAIEndpoint()).toBe('http://127.0.0.1:59421/v1');
});

it('prefers configured desktop HTTP services and never advertises Tauri resource origins', () => {
  vi.stubGlobal('window', { location: { origin: 'tauri://localhost' } });
  expect(openAIEndpoint()).toBeNull();
  expect(anthropicEndpoint(null, 8091)).toBeNull();
  vi.stubGlobal('window', { __TAURI_INTERNALS__: { invoke: vi.fn() }, location: { origin: 'http://tauri.localhost' } });
  expect(openAIEndpoint()).toBeNull();
  setApiBaseUrl('http://127.0.0.1:59421');
  expect(openAIEndpoint()).toBe('http://127.0.0.1:59421/v1');
  expect(openAIEndpoint('https://remote.example:9443/tyloinfer')).toBe('https://remote.example:9443/tyloinfer/v1');
  expect(openAIEndpoint('tauri://localhost')).toBeNull();
});
