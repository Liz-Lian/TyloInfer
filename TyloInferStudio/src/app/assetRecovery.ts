const reloadKey = 'tyloinfer.asset-reload';
const reloadInterval = 30_000;

export function isAssetLoadError(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error);
  return /(?:failed to fetch|error loading) dynamically imported module|importing a module script failed|unable to preload css|loading (?:css )?chunk .+ failed|chunkloaderror/i.test(message);
}

export function installAssetRecovery(target: Window = window, reload = () => target.location.reload()): () => void {
  let reloading = false;
  const onError = (event: Event) => {
    if (reloading || !target.navigator.onLine || !isAssetLoadError((event as Event & { payload?: unknown }).payload)) return;
    try {
      const previous = Number(target.sessionStorage.getItem(reloadKey));
      const now = Date.now();
      if (previous > 0 && now >= previous && now - previous < reloadInterval) return;
      target.sessionStorage.setItem(reloadKey, String(now));
    } catch { return; }
    reloading = true;
    event.preventDefault();
    reload();
  };
  target.addEventListener('vite:preloadError', onError);
  return () => target.removeEventListener('vite:preloadError', onError);
}
