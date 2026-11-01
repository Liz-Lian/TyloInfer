import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from 'react';
import { request } from '../../shared/api/client';
import { useRuntime } from '../../app/RuntimeProvider';

export interface AgentWorkspace { workspaceId: string; title: string; path: string; sessionIds: string[] }
export interface AgentSession { sessionId: string; running: boolean; blank: boolean; updatedAt?: number; cwd?: string; origin?: string; projections?: { values?: { title?: string | null } } }
export interface AgentApproval { eventId: string; agentId: string; request?: { toolName?: string; reason?: string } }
export interface AgentQuestion { eventId: string; request: { questions: { id: string; question: string; detail?: string; options?: { label: string; description?: string }[]; multiSelect?: boolean }[] } }
export interface AgentModel { id: string; name: string; contextWindow?: number; input?: string[]; reasoningEfforts?: Record<string, string> | false; thinkingToggleOnly?: boolean; mtpAvailable?: boolean }
export interface AgentState { running: boolean; error?: string | null; workspaces: AgentWorkspace[]; sessions: AgentSession[]; models: AgentModel[]; approvals: AgentApproval[]; questions?: AgentQuestion[]; archivedSessionIds?: string[] }
const empty: AgentState = { running: false, workspaces: [], sessions: [], models: [], approvals: [] };

export const agentRpc = <T,>(method: string, args: Record<string, unknown> = {}) =>
  request<T>('/api/v1/agent/rpc', { method: 'POST', body: JSON.stringify({ method, args }) });

const AgentContext = createContext<{ state: AgentState; starting: boolean; error: string; refresh(): Promise<void>; start(): Promise<void>; connectWorkspace(id: string): Promise<{ sessionId: string }> } | null>(null);

export function AgentProvider({ children }: { children: ReactNode }) {
  const { ready, connectionRevision, instances } = useRuntime();
  const [state, setState] = useState(empty);
  const [error, setError] = useState('');
  const [startError, setStartError] = useState('');
  const [starting, setStarting] = useState(false);
  const generation = useRef(0);
  const inFlight = useRef<Promise<void> | null>(null);
  const workspaceConnections = useRef(new Map<string, Promise<{ sessionId: string }>>());
  const modelRoutes = JSON.stringify(instances.filter(item => item.state === 'ready').map(item => [item.model, item.context_size || 32768]));
  const serverRoutes = JSON.stringify(state.models.map(item => [item.id, item.contextWindow || 32768]));
  const syncAttempt = useRef(0);
  const connectWorkspace = useCallback((id: string) => {
    const key = `${generation.current}:${id}`;
    const existing = workspaceConnections.current.get(key);
    if (existing) return existing;
    const pending = request<{ sessionId: string }>(`/api/v1/agent/workspaces/${encodeURIComponent(id)}/session`, { method: 'POST' })
      .finally(() => { if (workspaceConnections.current.get(key) === pending) workspaceConnections.current.delete(key); });
    workspaceConnections.current.set(key, pending);
    return pending;
  }, []);
  const refresh = useCallback(async () => {
    const revision = generation.current;
    try {
      const result = await request<AgentState>('/api/v1/agent/state');
      if (generation.current === revision) { setState(result); setError(''); }
    } catch (cause) {
      if (generation.current === revision) setError(cause instanceof Error ? cause.message : String(cause));
    }
  }, []);
  const start = useCallback(async () => {
    if (inFlight.current) return inFlight.current;
    const revision = generation.current;
    setStarting(true); setStartError('');
    const operation = (async () => { try {
      const result = await request<AgentState>('/api/v1/agent/start', { method: 'POST', body: '{}' });
      if (generation.current === revision) setState(result);
    } catch (cause) {
      if (generation.current === revision) setStartError(cause instanceof Error ? cause.message : String(cause));
    } finally { if (generation.current === revision) setStarting(false); inFlight.current = null; } })();
    inFlight.current = operation;
    return operation;
  }, []);
  const hasRunningTurn = state.sessions.some(item => item.running);
  useEffect(() => {
    if (!ready || !state.running || starting || hasRunningTurn || serverRoutes === modelRoutes) return;
    const timer = setTimeout(() => { syncAttempt.current = Date.now(); void start(); }, Math.max(200, 5000 - (Date.now() - syncAttempt.current)));
    return () => clearTimeout(timer);
  }, [ready, state.running, starting, hasRunningTurn, modelRoutes, serverRoutes, start]);
  useEffect(() => {
    generation.current += 1;
    syncAttempt.current = 0;
    setState(empty); setError(''); setStartError(''); setStarting(false);
    if (!ready) return;
    void refresh();
    const timer = setInterval(() => { if (!document.hidden) void refresh(); }, 2500);
    return () => { generation.current += 1; clearInterval(timer); };
  }, [ready, connectionRevision, refresh]);
  return <AgentContext.Provider value={{ state, starting, error: startError || error, refresh, start, connectWorkspace }}>{children}</AgentContext.Provider>;
}

export function useAgent() { return useContext(AgentContext); }
