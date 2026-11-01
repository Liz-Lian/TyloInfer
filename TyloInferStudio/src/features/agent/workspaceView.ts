import type { AgentSession, AgentWorkspace } from './AgentProvider';

export interface WorkspaceView { groupBy: 'workspace' | 'workspace-tree' | 'flat'; orderBy: 'manual' | 'updated'; archivedFilter: 'default' | 'show' | 'only' }
export const workspaceViewKey = 'tyloinfer.studio.agent.workspace-view.v1';
export const defaultWorkspaceView: WorkspaceView = { groupBy: 'workspace', orderBy: 'updated', archivedFilter: 'default' };

export function readWorkspaceView(): WorkspaceView {
  try {
    const value = JSON.parse(localStorage.getItem(workspaceViewKey) || '{}');
    return { groupBy: ['workspace', 'workspace-tree', 'flat'].includes(value.groupBy) ? value.groupBy : 'workspace',
      orderBy: ['manual', 'updated'].includes(value.orderBy) ? value.orderBy : 'updated',
      archivedFilter: ['default', 'show', 'only'].includes(value.archivedFilter) ? value.archivedFilter : 'default' };
  } catch { return { ...defaultWorkspaceView }; }
}

export function orderWorkspaceSessions(sessions: AgentSession[], order: WorkspaceView['orderBy']): AgentSession[] {
  return order === 'manual' ? sessions : [...sessions].sort((left, right) => (right.updatedAt || 0) - (left.updatedAt || 0));
}

export function workspaceParents(workspaces: AgentWorkspace[]): Map<string, string | null> {
  const paths = new Map(workspaces.map(item => {
    const path = item.path.replace(/\\/g, '/').replace(/\/+$/, '') || '/';
    return [item.workspaceId, /^[a-z]:/i.test(path) ? path.toLowerCase() : path];
  }));
  return new Map(workspaces.map(item => {
    const path = paths.get(item.workspaceId)!;
    const parent = workspaces.filter(other => other.workspaceId !== item.workspaceId && path !== paths.get(other.workspaceId)
      && path.startsWith(paths.get(other.workspaceId) === '/' ? '/' : `${paths.get(other.workspaceId)}/`))
      .sort((left, right) => paths.get(right.workspaceId)!.length - paths.get(left.workspaceId)!.length)[0];
    return [item.workspaceId, parent?.workspaceId || null];
  }));
}
