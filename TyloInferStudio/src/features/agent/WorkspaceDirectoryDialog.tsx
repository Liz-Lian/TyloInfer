import { useEffect, useRef, useState, type RefObject } from 'react';
import { CaretRightIcon, FolderIcon } from '@phosphor-icons/react';
import { Dialog } from '../../shared/ui/Dialog';
import { modelsApi } from '../../shared/api/resources/models';
import { ApiError } from '../../shared/api/client';
import type { ModelDirectoryList } from '../../shared/api/types';
import { errorMessage } from '../../app/formatters';
import { useSettings } from '../settings/SettingsProvider';

export function WorkspaceDirectoryDialog({ initialPath, onSelect, onClose, trigger }: {
  initialPath: string; onSelect(path: string): Promise<void>; onClose(): void; trigger: RefObject<HTMLButtonElement | null>;
}) {
  const { tr } = useSettings();
  const [listing, setListing] = useState<ModelDirectoryList | null>(null);
  const [path, setPath] = useState(initialPath);
  const [error, setError] = useState('');
  const [selectionError, setSelectionError] = useState('');
  const [loading, setLoading] = useState(true);
  const [adding, setAdding] = useState(false);
  const controller = useRef<AbortController | null>(null);
  const mounted = useRef(true);
  async function browse(directoryId?: string | null, target?: string) {
    controller.current?.abort();
    const next = new AbortController();
    controller.current = next;
    setLoading(true); setError(''); setSelectionError('');
    try {
      const result = await modelsApi.modelDirectories(directoryId, target, next.signal);
      if (next.signal.aborted) return;
      setListing(result); setPath(result.current_path || '');
    } catch (cause) {
      if (next.signal.aborted) return;
      setError(cause instanceof ApiError && cause.code === 'model_directory_not_found'
        ? tr('无法打开此目录，请检查路径和访问权限。', 'Cannot open this folder. Check its path and permissions.') : errorMessage(cause, tr));
      setPath(listing?.current_path || target || initialPath);
    } finally { if (!next.signal.aborted) setLoading(false); }
  }
  useEffect(() => {
    mounted.current = true;
    void browse(null, initialPath);
    return () => { mounted.current = false; controller.current?.abort(); };
  }, []);
  async function select() {
    if (!listing?.current_path || path.trim() !== listing.current_path || loading || adding || error) return;
    setAdding(true); setSelectionError('');
    try { await onSelect(listing.current_path); }
    catch (cause) { if (mounted.current) setSelectionError(errorMessage(cause, tr)); }
    finally { if (mounted.current) setAdding(false); }
  }
  return <Dialog open onOpenChange={open => { if (!open && !adding) onClose(); }} returnFocusRef={trigger}
    title={tr('选择工作区目录', 'Choose workspace folder')} description={tr('浏览 TyloInfer 服务所在设备的目录。', 'Browse folders on the TyloInfer server host.')}
    closeLabel={tr('关闭', 'Close')} className="model-browser-dialog workspace-directory-dialog">
    <form className="model-browser-location" onSubmit={event => { event.preventDefault(); if (path.trim()) void browse(null, path.trim()); }}>
      <button type="button" disabled={adding || loading || !listing?.parent_id} onClick={() => void browse(listing?.parent_id)}>{tr('上一级', 'Up')}</button>
      <input aria-label={tr('当前目录', 'Current directory')} value={path} onChange={event => setPath(event.target.value)} disabled={adding} spellCheck={false} />
      <button type="submit" disabled={adding || !path.trim()}>{tr('前往', 'Go')}</button>
    </form>
    <div className="model-browser-list" aria-busy={loading}>
      {loading ? <p role="status">{tr('正在读取目录…', 'Loading folder…')}</p> : <>
        {(error || selectionError) && <p className="workspace-directory-error" role="alert">{error || selectionError}</p>}
        {listing?.data.map(directory => <button type="button" key={directory.id} disabled={adding} onClick={() => void browse(directory.id)}>
          <FolderIcon size={17} /><span>{directory.name}</span><CaretRightIcon size={12} />
        </button>)}
        {listing && !listing.data.length && !error && <p>{tr('此目录没有子文件夹，可以直接选择。', 'This folder has no subfolders. You can select it directly.')}</p>}
      </>}
    </div>
    <footer><button type="button" disabled={adding} onClick={onClose}>{tr('取消', 'Cancel')}</button>
      <button className="primary" type="button" disabled={loading || adding || !listing?.current_path || path.trim() !== listing.current_path || Boolean(error)} onClick={() => void select()}>
        {adding ? tr('正在添加…', 'Adding…') : tr('选择此目录', 'Select folder')}
      </button>
    </footer>
  </Dialog>;
}
