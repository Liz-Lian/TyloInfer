import { useEffect, useRef, useState } from 'react';
import { CheckIcon, SlidersHorizontalIcon } from '@phosphor-icons/react';
import { useSettings } from '../settings/SettingsProvider';
import type { WorkspaceView } from './workspaceView';

export function WorkspaceViewMenu({ view, onChange }: { view: WorkspaceView; onChange(value: WorkspaceView): void }) {
  const { tr } = useSettings();
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    if (!open) return;
    root.current?.querySelector<HTMLButtonElement>('[role=menuitemradio][aria-checked=true]')?.focus();
    const outside = (event: PointerEvent) => { if (!root.current?.contains(event.target as Node)) setOpen(false); };
    const escape = (event: KeyboardEvent) => { if (event.key === 'Escape') { event.stopPropagation(); setOpen(false); trigger.current?.focus(); } };
    document.addEventListener('pointerdown', outside); document.addEventListener('keydown', escape);
    return () => { document.removeEventListener('pointerdown', outside); document.removeEventListener('keydown', escape); };
  }, [open]);
  const groups: { field: keyof WorkspaceView; title: string; options: [string, string][] }[] = [
    { field: 'groupBy', title: tr('分组方式', 'Group by'), options: [['workspace', tr('按工作区', 'By workspace')], ['workspace-tree', tr('按工作区树', 'Workspace tree')], ['flat', tr('单列表', 'In one list')]] },
    { field: 'orderBy', title: tr('排序方式', 'Order by'), options: [['manual', tr('原始顺序', 'Workspace order')], ['updated', tr('最近更新', 'Last updated')]] },
    { field: 'archivedFilter', title: tr('筛选会话', 'Filter conversations'), options: [['default', tr('隐藏已归档', 'Hide archived')], ['show', tr('全部对话', 'All conversations')], ['only', tr('仅显示已归档', 'Archived only')]] },
  ];
  return <div ref={root} className="agent-workspace-view">
    <button ref={trigger} className="agent-workspace-action" type="button" aria-label={tr('视图选项', 'View options')} title={tr('视图选项', 'View options')} aria-haspopup="menu" aria-expanded={open} onClick={() => setOpen(value => !value)}><SlidersHorizontalIcon size={16} /></button>
    {open && <div role="menu" aria-label={tr('视图选项', 'View options')} className="agent-workspace-view-menu" style={{ left: Math.max(10, Math.min(window.innerWidth - 230, trigger.current?.getBoundingClientRect().left || 10)), top: (trigger.current?.getBoundingClientRect().bottom || 0) + 5, maxHeight: Math.max(80, window.innerHeight - (trigger.current?.getBoundingClientRect().bottom || 0) - 15) }}
      onKeyDown={event => {
        if (!['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) return;
        const items = [...event.currentTarget.querySelectorAll<HTMLButtonElement>('[role=menuitemradio]')];
        const index = items.indexOf(document.activeElement as HTMLButtonElement);
        const next = event.key === 'Home' ? 0 : event.key === 'End' ? items.length - 1 : (index + (event.key === 'ArrowUp' ? -1 : 1) + items.length) % items.length;
        event.preventDefault(); items[next]?.focus();
      }}>{groups.map(group => <div key={group.field}><small>{group.title}</small>{group.options.map(([value, label]) => <button type="button" key={value} role="menuitemradio" aria-checked={view[group.field] === value} onClick={() => { onChange({ ...view, [group.field]: value }); setOpen(false); trigger.current?.focus(); }}><span>{label}</span>{view[group.field] === value && <CheckIcon size={13} />}</button>)}</div>)}</div>}
  </div>;
}
