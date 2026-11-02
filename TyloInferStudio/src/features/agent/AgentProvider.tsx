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

/** Invoke a workspace RPC from Agent views on services that implement the Agent API. */
export const agentRpc = <T,>(method: string, args: Record<string, unknown> = {}) =>
  request<T>('/api/v1/agent/rpc', { method: 'POST', body: JSON.stringify({ method, args }) });

const unavailableMessage = 'The native server does not provide Agent workspaces. Use text chat instead.';
interface AgentContextValue {
  /** Whether this connection is allowed to use the existing Agent API. */
  available: boolean;
  /** Whether connection-scoped state has been initialized before page startup effects run. */
  ready: boolean;
  state: AgentState;
  starting: boolean;
  error: string;
  /** Refresh supported Agent state and surface failures in the shared context. */
  refresh(): Promise<void>;
  /** Start or resynchronize a supported Agent runtime, deduplicating concurrent requests. */
  start(): Promise<void>;
  /** Open a workspace session or reject an unsupported native operation explicitly. */
  connectWorkspace(id: string): Promise<{ sessionId: string }>;
}
const AgentContext = createContext<AgentContextValue | null>(null);

/** Preserve Agent state across routes and poll only services that support Agent workspaces. */
export function AgentProvider({ children }: { children: ReactNode }) {
  const { ready, connectionRevision, instances, runtime } = useRuntime();
  const available = runtime?.source !== 'native-instances';
  const [state, setState] = useState(empty);
  const [error, setError] = useState('');
  const [startError, setStartError] = useState('');
  const [starting, setStarting] = useState(false);
  const [initializedRevision, setInitializedRevision] = useState<number | null>(null);
  const generation = useRef(0);
  const inFlight = useRef<Promise<void> | null>(null);
  const workspaceConnections = useRef(new Map<string, Promise<{ sessionId: string }>>());
  const modelRoutes = JSON.stringify(instances.filter(item => item.state === 'ready').map(item => [item.model, item.context_size || 32768]));
  const serverRoutes = JSON.stringify(state.models.map(item => [item.id, item.contextWindow || 32768]));
  const syncAttempt = useRef(0);
  const connectWorkspace = useCallback((id: string) => {
    if (!available) return Promise.reject(new Error(unavailableMessage));
    const key = `${generation.current}:${id}`;
    const existing = workspaceConnections.current.get(key);
    if (existing) return existing;
    const pending = request<{ sessionId: string }>(`/api/v1/agent/workspaces/${encodeURIComponent(id)}/session`, { method: 'POST' })
      .finally(() => { if (workspaceConnections.current.get(key) === pending) workspaceConnections.current.delete(key); });
    workspaceConnections.current.set(key, pending);
    return pending;
  }, [available]);
  const refresh = useCallback(async () => {
    if (!available) { setError(unavailableMessage); return; }
    const revision = generation.current;
    try {
      const result = await request<AgentState>('/api/v1/agent/state');
      if (generation.current === revision) { setState(result); setError(''); }
    } catch (cause) {
      if (generation.current === revision) setError(cause instanceof Error ? cause.message : String(cause));
    }
  }, [available]);
  const start = useCallback(async () => {
    if (!available) { setStartError(unavailableMessage); return; }
    if (inFlight.current) return inFlight.current;
    const revision = generation.current;
    setStarting(true); setStartError('');
    const operation = (async () => { try {
      const result = await request<AgentState>('/api/v1/agent/start', { method: 'POST', body: '{}' });
      if (generation.current === revision) setState(result);
    } catch (cause) {
      if (generation.current === revision) setStartError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      if (generation.current === revision) { setStarting(false); inFlight.current = null; }
    } })();
    inFlight.current = operation;
    return operation;
  }, [available]);
  const hasRunningTurn = state.sessions.some(item => item.running);
  useEffect(() => {
    if (!available || !ready || !state.running || starting || hasRunningTurn || serverRoutes === modelRoutes) return;
    const timer = setTimeout(() => { syncAttempt.current = Date.now(); void start(); }, Math.max(200, 5000 - (Date.now() - syncAttempt.current)));
    return () => clearTimeout(timer);
  }, [available, ready, state.running, starting, hasRunningTurn, modelRoutes, serverRoutes, start]);
  useEffect(() => {
    syncAttempt.current = 0;
    inFlight.current = null;
    workspaceConnections.current.clear();
    setState(empty); setError(''); setStartError(''); setStarting(false);
    setInitializedRevision(available && ready ? connectionRevision : null);
    if (!available || !ready) return;
    void refresh();
    const timer = setInterval(() => { if (!document.hidden) void refresh(); }, 2500);
    return () => { generation.current += 1; clearInterval(timer); };
  }, [available, ready, connectionRevision, refresh]);
  return <AgentContext.Provider value={{ available, ready: available && ready && initializedRevision === connectionRevision,
    state, starting, error: startError || error, refresh, start, connectWorkspace }}>{children}</AgentContext.Provider>;
}

/** Read Agent state and explicit service availability without starting the Agent. */
export function useAgent() { return useContext(AgentContext); }
