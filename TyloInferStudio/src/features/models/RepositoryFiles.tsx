import { useState } from 'react';
import { Icon } from '../../app/display';
import { DownloadSimpleIcon } from '@phosphor-icons/react';
import type { HubModelInfo, HubModelVariant } from '../../shared/api/types';
import type { DownloadOrigin } from './ModelBrowser';
import { downloadSize, repositoryFileGroups } from './repositoryFileGroups';

type HubModelFile = HubModelInfo['files'][number];

export function RepositoryFiles({ files, variants = [], disabled, onDownload, tr, expanded = false }: {
  files: HubModelFile[];
  variants?: HubModelVariant[];
  disabled: boolean;
  onDownload(files: HubModelFile[], label: string, origin: DownloadOrigin): void;
  tr(zh: string, en: string): string;
  expanded?: boolean;
}) {
  const [folder, setFolder] = useState('');
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const contained = files.filter((file) => file.name.startsWith(folder));
  const folders = Array.from(new Set(contained.map((file) => file.name.slice(folder.length))
    .filter((name) => name.includes('/')).map((name) => name.split('/')[0]))).sort();
  const groups = repositoryFileGroups(files, variants).filter(group => group.label.startsWith(folder) && !group.label.slice(folder.length).includes('/'));
  const chosen = files.filter((file) => selected.has(file.name));
  function submit(event: React.MouseEvent<HTMLButtonElement>, items: HubModelFile[], label: string) {
    const rect = event.currentTarget.getBoundingClientRect();
    onDownload(items, label, { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 });
  }
  return <details className="repository-files" open={expanded || undefined}>
    <summary>{tr('文件与文件夹', 'Files and folders')} · {files.length}</summary>
    <div className="repository-file-toolbar">
      <button disabled={!folder} onClick={() => setFolder(folder.replace(/[^/]+\/$/, ''))} type="button">{tr('上一级', 'Up')}</button>
      <code>/{folder}</code>
      <button disabled={disabled || !contained.length} onClick={(event) => submit(event, contained, folder.replace(/\/$/, '') || 'repository')} type="button">{tr('下载此文件夹', 'Download folder')}</button>
    </div>
    <div className="repository-file-list">
      {folders.map((name) => {
        const items = contained.filter(file => file.name.startsWith(`${folder}${name}/`));
        return <div className="repository-folder-row" key={name}>
          <button className="repository-folder" type="button" aria-label={`${tr('打开文件夹', 'Open folder')} ${name}`} onClick={() => setFolder(`${folder}${name}/`)}>
            <Icon name="folder" size={17} /><span>{name}/</span><small>{items.length} {tr('个文件', 'files')} · {downloadSize(items.reduce((sum, file) => sum + file.byte_size, 0))}</small>
          </button>
          <button type="button" className="repository-folder-download" aria-label={`${tr('下载文件夹', 'Download folder')} ${name}`} title={tr('下载整个文件夹', 'Download entire folder')} disabled={disabled}
            onClick={event => submit(event, items, `${folder}${name}`)}><DownloadSimpleIcon size={17} /></button>
        </div>;
      })}
      {groups.map((group) => <label key={group.id}>
        <input type="checkbox" checked={group.files.every(file => selected.has(file.name))} onChange={(event) => setSelected((current) => {
          const next = new Set(current);
          for (const file of group.files) {
            if (event.target.checked) next.add(file.name); else next.delete(file.name);
          }
          return next;
        })} />
        <span>{group.label.slice(folder.length)}{group.expected && <small className="repository-shard-count">{group.files.length}/{group.expected} {tr('分片', 'shards')}{group.missing > 0 ? ` · ${tr('分片不全', 'Incomplete')}` : ''}</small>}</span><small>{downloadSize(group.files.reduce((sum, file) => sum + file.byte_size, 0))}</small>
      </label>)}
    </div>
    <button disabled={disabled || !chosen.length} onClick={(event) => submit(event, chosen, 'selected-files')} type="button">
      {tr(`下载所选 ${chosen.length} 个文件`, `Download ${chosen.length} selected files`)}
    </button>
  </details>;
}
