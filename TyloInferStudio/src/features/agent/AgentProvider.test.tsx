import { act, cleanup, renderHook } from '@testing-library/react';
import type { ReactNode } from 'react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { AgentProvider, useAgent, type AgentState } from './AgentProvider';
import type { RuntimeStatus } from '../../shared/api/types';

const runtime = vi.hoisted(() => ({ ready: true, connectionRevision: 1, instances: [],
  runtime: null as RuntimeStatus | null }));
vi.mock('../../app/RuntimeProvider', () => ({ useRuntime: () => runtime }));

const idle: AgentState = { running: false, workspaces: [], sessions: [], models: [], approvals: [] };
const wrapper = ({ children }: { children: ReactNode }) => <AgentProvider>{children}</AgentProvider>;
beforeEach(() => { runtime.runtime = {}; runtime.ready = true; runtime.connectionRevision = 1; });
afterEach(() => { cleanup(); vi.useRealTimers(); vi.unstubAllGlobals(); });

it.each([null, { source: 'native-instances' } as RuntimeStatus])('does not poll or start Agent APIs on a native or uninitialized connection: %j', async status => {
  vi.useFakeTimers();
  runtime.runtime = status;
  const fetchMock = vi.fn();
  vi.stubGlobal('fetch', fetchMock);
  const { result } = renderHook(useAgent, { wrapper });
  expect(result.current?.available).toBe(false);
  await act(async () => { await vi.advanceTimersByTimeAsync(7500); });
  await act(async () => { await result.current!.refresh(); await result.current!.start(); });
  await expect(result.current!.connectWorkspace('workspace-a')).rejects.toThrow('Use text chat instead');
  expect(result.current?.error).toContain('does not provide Agent workspaces');
  expect(result.current?.state.running).toBe(false);
  expect(fetchMock).not.toHaveBeenCalled();
});

it('preserves supported services polling, startup and workspace connection behavior', async () => {
  vi.useFakeTimers();
  const fetchMock = vi.fn(async (url: string) => new Response(JSON.stringify(
    url.endsWith('/session') ? { sessionId: 'agent-session' }
      : url.endsWith('/start') ? { ...idle, running: true } : idle,
  )));
  vi.stubGlobal('fetch', fetchMock);
  const { result } = renderHook(useAgent, { wrapper });
  await act(async () => {});
  expect(result.current?.available).toBe(true);
  expect(fetchMock.mock.calls.map(([url]) => url)).toEqual(['/api/v1/agent/state']);
  await act(async () => { await vi.advanceTimersByTimeAsync(2500); });
  expect(fetchMock.mock.calls.map(([url]) => url)).toEqual(['/api/v1/agent/state', '/api/v1/agent/state']);
  await act(async () => { await result.current!.start(); });
  expect(result.current?.state.running).toBe(true);
  await expect(result.current!.connectWorkspace('workspace-a')).resolves.toEqual({ sessionId: 'agent-session' });
  expect(fetchMock.mock.calls.map(([url]) => url)).toContain('/api/v1/agent/start');
  expect(fetchMock.mock.calls.map(([url]) => url)).toContain('/api/v1/agent/workspaces/workspace-a/session');
});

it('stops old Agent polling and discards pending state when switching to a native connection', async () => {
  vi.useFakeTimers();
  let finish!: (response: Response) => void;
  const fetchMock = vi.fn(() => new Promise<Response>(resolve => { finish = resolve; }));
  vi.stubGlobal('fetch', fetchMock);
  const { result, rerender } = renderHook(useAgent, { wrapper });
  expect(fetchMock).toHaveBeenCalledOnce();
  runtime.runtime = { source: 'native-instances' };
  runtime.connectionRevision++;
  rerender();
  await act(async () => {
    finish(new Response(JSON.stringify({ ...idle, running: true, workspaces: [{ workspaceId: 'old' }] })));
    await vi.advanceTimersByTimeAsync(7500);
  });
  expect(result.current?.available).toBe(false);
  expect(result.current?.state).toEqual(idle);
  expect(fetchMock).toHaveBeenCalledOnce();
});
