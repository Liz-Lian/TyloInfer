import { afterEach, expect, it } from 'vitest';
import { defaultWorkspaceView, orderWorkspaceSessions, readWorkspaceView, workspaceParents, workspaceViewKey } from './workspaceView';
import type { AgentSession, AgentWorkspace } from './AgentProvider';

afterEach(() => localStorage.removeItem(workspaceViewKey));

it('validates saved view preferences and tolerates corrupt or inaccessible storage', () => {
  expect(readWorkspaceView()).toEqual(defaultWorkspaceView);
  localStorage.setItem(workspaceViewKey, 'invalid json');
  expect(readWorkspaceView()).toEqual(defaultWorkspaceView);
  localStorage.setItem(workspaceViewKey, JSON.stringify({ groupBy: 'flat', orderBy: 'random', archivedFilter: 'only' }));
  expect(readWorkspaceView()).toEqual({ groupBy: 'flat', orderBy: 'updated', archivedFilter: 'only' });
});

it('sorts recent sessions without changing workspace order or tying unstamped sessions arbitrarily', () => {
  const sessions: AgentSession[] = [{ sessionId: 'older', updatedAt: 1, blank: false, running: false }, { sessionId: 'newer', updatedAt: 2, blank: false, running: false }, { sessionId: 'blank', blank: true, running: false }];
  expect(orderWorkspaceSessions(sessions, 'updated').map(item => item.sessionId)).toEqual(['newer', 'older', 'blank']);
  expect(orderWorkspaceSessions(sessions, 'manual')).toBe(sessions);
  expect(sessions[0].sessionId).toBe('older');
});

it('builds workspace trees by directory boundaries on POSIX and Windows paths', () => {
  const workspaces = [['root', '/projects'], ['child', '/projects/app'], ['nested', '/projects/app/src'], ['other', '/projects-other'], ['win', 'C:\\Work'], ['win-child', 'c:\\work\\app']]
    .map(([workspaceId, path]) => ({ workspaceId, path, title: workspaceId, sessionIds: [] } satisfies AgentWorkspace));
  const parents = workspaceParents(workspaces);
  expect(Object.fromEntries(parents)).toEqual({ root: null, child: 'root', nested: 'child', other: null, win: null, 'win-child': 'win' });
});
