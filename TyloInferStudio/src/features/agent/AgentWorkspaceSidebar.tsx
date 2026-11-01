import { useEffect, useRef, useState } from 'react';
import { ArchiveIcon, ArrowCounterClockwiseIcon, CaretRightIcon, FolderIcon, FolderPlusIcon, MagnifyingGlassIcon, NotePencilIcon, XIcon } from '@phosphor-icons/react';
import { useSearchParams } from 'react-router';
import { useSettings } from '../settings/SettingsProvider';
import { request } from '../../shared/api/client';
import { agentRpc, useAgent, type AgentSession, type AgentWorkspace } from './AgentProvider';
import { WorkspaceDirectoryDialog } from './WorkspaceDirectoryDialog';
import { WorkspaceViewMenu } from './WorkspaceViewMenu';
import { orderWorkspaceSessions, readWorkspaceView, workspaceParents, workspaceViewKey } from './workspaceView';
import './agent.css';

export function AgentWorkspaceSidebar({ onOpen }: { onOpen(path: string): void }) {
  const agent = useAgent();
  const { tr } = useSettings();
  const [search] = useSearchParams();
  const [closed, setClosed] = useState<string[]>([]);
  const [adding, setAdding] = useState(false);
  const [view, setView] = useState(readWorkspaceView);
  const [searching, setSearching] = useState(false);
  const [query, setQuery] = useState('');
  const [restoring, setRestoring] = useState('');
  const [error, setError] = useState('');
  const [retry, setRetry] = useState(0);
  const [remote, setRemote] = useState<{ query: string; items: { sessionId: string; snippet: string }[]; pending: boolean; hasMore: boolean; failed: boolean }>({ query: '', items: [], pending: false, hasMore: false, failed: false });
  const trigger = useRef<HTMLButtonElement>(null);
  const searchTrigger = useRef<HTMLButtonElement>(null);
  const searchInput = useRef<HTMLInputElement>(null);
  const normalizedQuery = query.trim();
  useEffect(() => { try { localStorage.setItem(workspaceViewKey, JSON.stringify(view)); } catch {} }, [view]);
  useEffect(() => { if (searching) searchInput.current?.focus(); }, [searching]);
  useEffect(() => {
    if (!normalizedQuery || !agent?.state.running) {
      setRemote({ query: normalizedQuery, items: [], pending: false, hasMore: false, failed: Boolean(normalizedQuery) });
      return;
    }
    const controller = new AbortController();
    setRemote({ query: normalizedQuery, items: [], pending: true, hasMore: false, failed: false });
    const timer = setTimeout(() => {
      void request<{ items: { sessionId: string; snippet: string }[]; hasMore: boolean }>(`/api/v1/agent/search?query=${encodeURIComponent(normalizedQuery)}`, { signal: controller.signal })
        .then(value => { if (!controller.signal.aborted) setRemote({ query: normalizedQuery, ...value, pending: false, failed: false }); })
        .catch(() => { if (!controller.signal.aborted) setRemote({ query: normalizedQuery, items: [], pending: false, hasMore: false, failed: true }); });
    }, 200);
    return () => { clearTimeout(timer); controller.abort(); };
  }, [normalizedQuery, agent?.state.running, retry]);
  if (!agent) return null;
  async function add(path: string) {
    const result = await agentRpc<{ workspace: { workspaceId: string } }>('workspace/create', { request: { path } });
    await agent!.refresh(); setAdding(false);
    onOpen(`/agent?workspace=${encodeURIComponent(result.workspace.workspaceId)}`);
  }
  const byId = new Map(agent.state.sessions.map(item => [item.sessionId, item]));
  const archived = new Set(agent.state.archivedSessionIds || []);
  const workspaceBySession = new Map<string, AgentWorkspace>();
  for (const workspace of agent.state.workspaces) for (const id of workspace.sessionIds) if (!workspaceBySession.has(id)) workspaceBySession.set(id, workspace);
  const visible = (item: AgentSession) => item.origin !== 'subagent' && (view.archivedFilter === 'show' || (view.archivedFilter === 'only' ? archived.has(item.sessionId) : !archived.has(item.sessionId)));
  const sessions = agent.state.sessions.filter(visible);
  const titleOf = (item: AgentSession) => item.blank ? tr('新建会话', 'New conversation') : item.projections?.values?.title || tr('未命名对话', 'Untitled conversation');
  async function restore(id: string) {
    setRestoring(id); setError('');
    try { await agentRpc('workspace/unarchiveSession', { request: { sessionId: id } }); await agent!.refresh(); }
    catch { setError(tr('无法恢复该对话，请重试。', 'Could not restore this conversation. Please retry.')); }
    finally { setRestoring(''); }
  }
  function sessionRow(session: AgentSession, context = false, snippet = '') {
    const workspace = workspaceBySession.get(session.sessionId);
    const title = titleOf(session);
    const isArchived = archived.has(session.sessionId);
    return <div className={`agent-workspace-session-row${isArchived ? ' archived' : ''}`} key={session.sessionId}>
      <button title={title} className={search.get('session') === session.sessionId ? 'active' : ''} type="button" disabled={isArchived}
        aria-label={session.blank ? `${tr('打开空白会话', 'Open blank conversation')} · ${workspace?.title || tr('未分组', 'Ungrouped')}` : undefined}
        onClick={() => onOpen(`/agent?${new URLSearchParams(workspace ? { workspace: workspace.workspaceId, session: session.sessionId } : { session: session.sessionId })}`)}>
        {isArchived ? <ArchiveIcon size={12} /> : <i className={session.running ? 'running' : ''} />}<span><span>{title}</span>{context && <small>{workspace?.title || tr('未分组', 'Ungrouped')}</small>}{snippet && <small className="agent-workspace-result-snippet">{snippet}</small>}</span>
      </button>{isArchived && <button type="button" className="agent-unarchive-session" disabled={Boolean(restoring)} aria-label={`${tr('恢复对话', 'Unarchive conversation')} · ${title}`} title={tr('恢复对话', 'Unarchive conversation')} onClick={() => void restore(session.sessionId)}><ArrowCounterClockwiseIcon size={14} /></button>}
    </div>;
  }
  const members = (workspace: AgentWorkspace) => orderWorkspaceSessions(workspace.sessionIds.flatMap(id => { const item = byId.get(id); return item && visible(item) ? [item] : []; }), view.orderBy);
  const parents = workspaceParents(agent.state.workspaces);
  const children = (id: string | null) => agent.state.workspaces.filter(item => parents.get(item.workspaceId) === id);
  function workspaceNode(workspace: AgentWorkspace, depth = 0) {
    const rows = members(workspace);
    const nested = view.groupBy === 'workspace-tree' ? children(workspace.workspaceId).map(child => workspaceNode(child, depth + 1)).filter(Boolean) : [];
    if (view.archivedFilter === 'only' && !rows.length && !nested.length) return null;
    const expanded = !closed.includes(workspace.workspaceId);
    return <div className="agent-workspace-node" key={workspace.workspaceId} style={{ marginLeft: depth ? 10 : 0 }}>
      <div className="agent-workspace-heading">
        <button type="button" aria-expanded={expanded} aria-label={`${tr('展开工作区', 'Expand workspace')} ${workspace.title}`}
          onClick={() => setClosed(values => values.includes(workspace.workspaceId) ? values.filter(value => value !== workspace.workspaceId) : [...values, workspace.workspaceId])}><CaretRightIcon size={9} weight="fill" /></button>
        <button type="button" className={search.get('workspace') === workspace.workspaceId ? 'selected' : ''} title={workspace.path}
          onClick={() => onOpen(`/agent?workspace=${encodeURIComponent(workspace.workspaceId)}`)}><FolderIcon size={14} /><span>{workspace.title}</span></button>
        <button type="button" className="agent-workspace-new-conversation" aria-label={`${tr('新建会话', 'New conversation')} · ${workspace.title}`}
          title={tr('新建会话', 'New conversation')} onClick={() => onOpen(`/agent?workspace=${encodeURIComponent(workspace.workspaceId)}`)}><NotePencilIcon size={14} /></button>
      </div>
      {expanded && <><div className="agent-workspace-sessions">{rows.map(item => sessionRow(item))}</div>{nested}</>}
    </div>;
  }
  const currentRemote = remote.query === normalizedQuery ? remote : null;
  const matching = new Map(orderWorkspaceSessions(sessions.filter(item => `${titleOf(item)}\n${workspaceBySession.get(item.sessionId)?.title || ''}`.toLowerCase().includes(normalizedQuery.toLowerCase())), 'updated').map(item => [item.sessionId, item]));
  for (const match of currentRemote?.items || []) { const item = byId.get(match.sessionId); if (item && visible(item)) matching.set(item.sessionId, item); }
  const results = [...matching.values()].slice(0, 20);
  const snippets = new Map(currentRemote?.items.map(item => [item.sessionId, item.snippet]));
  const ungrouped = orderWorkspaceSessions(sessions.filter(item => !workspaceBySession.has(item.sessionId)), view.orderBy);
  return <div className="agent-workspace-tree">
    <div className="agent-workspace-controls">
    <div className="agent-workspace-toolbar"><span>{tr('工作区', 'Workspaces')}</span>
      <div className="agent-workspace-toolbar-actions"><button ref={searchTrigger} className="agent-workspace-action" type="button" disabled={!agent.state.running} aria-label={tr('搜索会话', 'Search conversations')} title={tr('搜索会话', 'Search conversations')} aria-expanded={searching} onClick={() => { setSearching(value => !value); setQuery(''); }}><MagnifyingGlassIcon size={16} /></button>
      <WorkspaceViewMenu view={view} onChange={setView} />
      <button ref={trigger} className="agent-add-workspace agent-workspace-action" type="button" disabled={!agent.state.running}
        aria-label={tr('添加工作区', 'Add workspace')} title={tr('添加工作区', 'Add workspace')} onClick={() => setAdding(true)}><FolderPlusIcon size={16} /></button>
      </div>
    </div>
    {searching && <div className="agent-workspace-search"><MagnifyingGlassIcon size={13} /><input ref={searchInput} type="search" aria-label={tr('搜索会话', 'Search conversations')} placeholder={tr('搜索名称或内容…', 'Search names or content…')} value={query} maxLength={500} onChange={event => setQuery(event.target.value)} onKeyDown={event => { if (event.key === 'Escape') { event.stopPropagation(); setQuery(''); setSearching(false); searchTrigger.current?.focus(); } }} /><button type="button" aria-label={tr('清除搜索', 'Clear search')} onClick={() => { setQuery(''); searchInput.current?.focus(); }}><XIcon size={12} /></button></div>}
    {error && <p className="agent-workspace-notice" role="alert">{error}</p>}
    </div>
    <div className="agent-workspace-list" tabIndex={0} role="region" aria-label={tr('工作区列表', 'Workspace list')}>
    {normalizedQuery ? <div className="agent-workspace-search-results" aria-label={tr('搜索结果', 'Search results')}>
      <div className="agent-workspace-sessions">{results.map(item => sessionRow(item, true, snippets.get(item.sessionId)))}</div>
      {(!currentRemote || currentRemote.pending) && <p className="agent-workspace-notice" role="status">{tr('正在搜索会话历史…', 'Searching conversation history…')}</p>}
      {currentRemote?.failed && <p className="agent-workspace-notice" role="alert">{tr('无法搜索历史内容。', 'Could not search conversation history.')}<button type="button" onClick={() => setRetry(value => value + 1)}>{tr('重试', 'Retry')}</button></p>}
      {currentRemote && !currentRemote.pending && !currentRemote.failed && !results.length && <p className="agent-workspace-notice">{tr('无匹配对话', 'No matching conversations')}</p>}
      {(matching.size > 20 || currentRemote?.hasMore) && <p className="agent-workspace-notice">{tr('仅显示前 20 条，请缩小搜索范围。', 'Showing the first 20 results. Narrow your search.')}</p>}
    </div> : view.groupBy === 'flat' ? <div className="agent-workspace-sessions agent-workspace-flat">{orderWorkspaceSessions(sessions, view.orderBy).map(item => sessionRow(item, true))}{!sessions.length && <p className="agent-workspace-notice">{tr('暂无对话', 'No conversations yet')}</p>}</div>
      : <>{(view.groupBy === 'workspace-tree' ? children(null) : agent.state.workspaces).map(item => workspaceNode(item))}{ungrouped.length > 0 && <><div className="agent-workspace-ungrouped">{tr('未分组', 'Ungrouped')}</div><div className="agent-workspace-sessions">{ungrouped.map(item => sessionRow(item))}</div></>}{view.archivedFilter === 'only' && !sessions.length && <p className="agent-workspace-notice">{tr('暂无已归档对话', 'No archived conversations')}</p>}</>}
    </div>
    {adding && <WorkspaceDirectoryDialog initialPath={agent.state.workspaces.find(item => item.workspaceId === search.get('workspace'))?.path || '/'}
      onSelect={add} onClose={() => setAdding(false)} trigger={trigger} />}
  </div>;
}
