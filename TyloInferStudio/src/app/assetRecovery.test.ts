import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { installAssetRecovery, isAssetLoadError } from './assetRecovery';

let remove: (() => void) | undefined;
const failure = (message = 'Failed to fetch dynamically imported module: /assets/Overview-old.js') => {
  const event = new Event('vite:preloadError', { cancelable: true });
  Object.assign(event, { payload: new TypeError(message) });
  window.dispatchEvent(event);
  return event;
};

beforeEach(() => {
  sessionStorage.clear();
  vi.spyOn(Date, 'now').mockReturnValue(100_000);
});
afterEach(() => { remove?.(); remove = undefined; vi.restoreAllMocks(); });

it.each([
  'Failed to fetch dynamically imported module: /assets/page.js',
  'error loading dynamically imported module: /assets/page.js',
  'Importing a module script failed.',
  'Unable to preload CSS for /assets/page.css',
  'Loading chunk 5 failed.',
  'ChunkLoadError: page',
])('recognizes asset loading failures: %s', message => {
  expect(isAssetLoadError(new Error(message))).toBe(true);
});

it.each(['ChatProvider is missing', 'page render failed', 'http_500', null, undefined])('does not reload for ordinary application errors: %s', error => {
  expect(isAssetLoadError(error)).toBe(false);
});

it('reloads once for a missing route asset and cancels the preload error', () => {
  const reload = vi.fn();
  remove = installAssetRecovery(window, reload);
  expect(failure().defaultPrevented).toBe(true);
  failure();
  expect(reload).toHaveBeenCalledOnce();
});

it('retains the guard across reloads and does not create a refresh loop', () => {
  const reload = vi.fn();
  remove = installAssetRecovery(window, reload);
  failure();
  remove();
  remove = installAssetRecovery(window, reload);
  expect(failure().defaultPrevented).toBe(false);
  expect(reload).toHaveBeenCalledOnce();
  vi.mocked(Date.now).mockReturnValue(131_000);
  expect(failure().defaultPrevented).toBe(true);
  expect(reload).toHaveBeenCalledTimes(2);
});

it('keeps error handling available while offline or session storage is blocked', () => {
  const reload = vi.fn();
  remove = installAssetRecovery(window, reload);
  const online = vi.spyOn(navigator, 'onLine', 'get').mockReturnValue(false);
  expect(failure().defaultPrevented).toBe(false);
  online.mockReturnValue(true);
  vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new Error('blocked'); });
  expect(failure().defaultPrevented).toBe(false);
  expect(reload).not.toHaveBeenCalled();
});

it('leaves actual rendering errors alone and removes its event listener on disposal', () => {
  const reload = vi.fn();
  remove = installAssetRecovery(window, reload);
  expect(failure('ChatProvider is missing').defaultPrevented).toBe(false);
  remove();
  failure();
  expect(reload).not.toHaveBeenCalled();
});
