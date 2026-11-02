import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { MemoryRouter, useLocation } from 'react-router';
import { StrictMode, type ReactNode } from 'react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import App from '../src/App';
import { SettingsProvider } from '../src/features/settings/SettingsProvider';
import { DEFAULT_SETTINGS, SETTINGS_KEY } from '../src/features/settings/configuration';
import { useConversationStore } from '../src/features/chat/state/conversationStore';
import { useDraftStore } from '../src/features/chat/state/draftStore';
import { setApiBaseUrl } from '../src/shared/api/client';
import type { Message, ResponseResource, RuntimeInstance, RuntimeModel, RuntimeStatus, Session } from '../src/shared/api/types';

const runtime = vi.hoisted(() => ({
  runtime: { source: 'native-instances', model: 'Native Model', runtime_state: 'ready', max_context: 8192 } as RuntimeStatus,
  models: [{ id: 'Native Model' }] as RuntimeModel[],
  instances: [{ id: 'instance-a', model: 'Native Model', state: 'ready', devices: [], context_size: 8192 }] as RuntimeInstance[],
  artifactNames: {}, selectedModel: 'Native Model', setSelectedModel: vi.fn(),
  capabilities: null, realtime: null, voiceComponent: null, studio: null,
  ready: true, loading: false, connectionRevision: 1, connectionError: null, refreshError: null,
  jobStreamErrors: {}, reloadingInstances: {}, authenticationRequired: false,
  refreshRuntime: vi.fn(async () => true), reloadService: vi.fn(), retryJobStreams: vi.fn(), addJob: vi.fn(),
}));

// Keep the real App, shell, both feature providers and pages; isolate runtime state sources and unrelated update checks.
vi.mock('../src/app/RuntimeProvider', () => ({
  RuntimeProvider: ({ children }: { children: ReactNode }) => children,
  useRuntime: () => runtime,
}));
vi.mock('../src/features/settings/UpdateManager', () => ({
  StudioUpdateProvider: ({ children }: { children: ReactNode }) => children,
  useStudioUpdateContext: () => ({ status: null, busy: null }),
}));

const sessionId = '11111111-1111-4111-8111-111111111111';
const responseId = '22222222-2222-4222-8222-222222222222';
const timestamp = '2026-11-02T00:00:00+00:00';
let sessions: Session[];
let messages: Message[];
let responses: ResponseResource[];
let stream: ReadableStreamDefaultController<Uint8Array>;
let generationRequest: Record<string, unknown>;
let generationSignal: AbortSignal | null;
let requests: { path: string; method: string }[];

function savedSession(): Session {
  return { id: sessionId, model: 'Native Model', mode: 'text', state: 'idle', revision: 0,
    title: 'Native conversation', runtime_instance_id: null, created_at: timestamp, updated_at: timestamp, metadata: {} };
}

function frame(sequence: number, payload: unknown): Uint8Array {
  return new TextEncoder().encode(`data: ${JSON.stringify({ protocol_version: '1.0', session_id: sessionId,
    sequence, timestamp, payload })}\n\n`);
}

function CurrentPath() {
  const location = useLocation();
  return <output aria-label="Current path">{location.pathname}{location.search}</output>;
}

function renderApp(path: string, strict = false) {
  const app = <MemoryRouter initialEntries={[path]}><SettingsProvider><App /><CurrentPath /></SettingsProvider></MemoryRouter>;
  return render(strict ? <StrictMode>{app}</StrictMode> : app);
}

beforeEach(() => {
  useConversationStore.getState().reset();
  useDraftStore.setState({ drafts: {} });
  setApiBaseUrl('');
  localStorage.setItem(SETTINGS_KEY, JSON.stringify({ ...DEFAULT_SETTINGS, language: 'en' }));
  runtime.runtime = { source: 'native-instances', model: 'Native Model', runtime_state: 'ready', max_context: 8192 };
  runtime.models = [{ id: 'Native Model' }];
  runtime.instances = [{ id: 'instance-a', model: 'Native Model', state: 'ready', devices: [], context_size: 8192 }];
  runtime.selectedModel = 'Native Model';
  sessions = [savedSession()]; messages = []; responses = []; requests = []; generationSignal = null;
  vi.stubGlobal('ResizeObserver', class { observe() {} unobserve() {} disconnect() {} });
  vi.stubGlobal('matchMedia', vi.fn((query: string) => ({ matches: query.includes('min-width'),
    media: query, addEventListener: vi.fn(), removeEventListener: vi.fn() })));
  Object.defineProperty(HTMLElement.prototype, 'scrollIntoView', { configurable: true, value: vi.fn() });
  vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
    const path = new URL(url, window.location.origin).pathname;
    const method = init?.method ?? 'GET';
    requests.push({ path, method });
    const json = (value: unknown) => new Response(JSON.stringify(value));
    if (path === '/api/v1/auth/policy') return json({ enabled: false, has_key: false });
    if (path === '/api/v1/runtime/model-aliases') return json({ aliases: {} });
    if (path === '/api/v1/jobs/kinds') return json({ data: [] });
    if (path === '/api/v1/hub/download-directory') return json({ actual_directory: 'D:/isolated/models', default_directory: 'D:/isolated/models', evict_prefix_cache: false });
    if (runtime.runtime.source !== 'native-instances' && ['/api/v1/agent/state', '/api/v1/agent/start'].includes(path)) {
      return json({ running: path.endsWith('/start'), workspaces: [], sessions: [], models: [], approvals: [] });
    }
    if (path === '/api/v1/sessions' && method === 'GET') return json({ data: sessions });
    if (path === '/api/v1/sessions' && method === 'POST') {
      sessions = [savedSession()]; return json(sessions[0]);
    }
    if (path === `/api/v1/sessions/${sessionId}/messages`) return json({ data: messages });
    if (path === `/api/v1/sessions/${sessionId}/responses` && method === 'GET') return json({ data: responses });
    if (path === `/api/v1/sessions/${sessionId}`) return json(sessions[0]);
    if (path === `/api/v1/sessions/${sessionId}/responses` && method === 'POST') {
      generationRequest = JSON.parse(init!.body as string);
      generationSignal = init?.signal ?? null;
      const body = new ReadableStream<Uint8Array>({ start(controller) { stream = controller; } });
      return new Response(body, { headers: { 'Content-Type': 'text/event-stream' } });
    }
    throw new Error(`Unexpected service request: ${method} ${path}`);
  }));
});

afterEach(() => { cleanup(); vi.useRealTimers(); setApiBaseUrl(''); vi.unstubAllGlobals(); });

it.each([
  ['/resources', 'resource monitoring or cache management'],
  ['/analysis', 'architecture or tensor analysis'],
  ['/evaluations', 'datasets, quality evaluations or benchmark jobs'],
])('keeps %s as an unavailable entry without mounting requests or polling', async (path, detail) => {
  renderApp(path);
  expect(await screen.findByText(new RegExp(detail))).toBeInTheDocument();
  vi.useFakeTimers();
  await act(async () => { await vi.advanceTimersByTimeAsync(12000); });
  expect(requests.filter(request => !request.path.startsWith('/api/v1/sessions'))).toEqual([]);
});

it('retains native authentication and aliases while all runtime settings tabs avoid unsupported APIs', async () => {
  setApiBaseUrl('http://127.0.0.1:59421');
  renderApp('/runtime');
  expect(await screen.findByRole('spinbutton', { name: 'OpenAI port' })).toBeDisabled();
  expect(screen.getByRole('spinbutton', { name: 'OpenAI port' })).toHaveValue(59421);
  await waitFor(() => expect(screen.getByRole('combobox', { name: 'Authentication' })).toBeEnabled());
  fireEvent.click(screen.getByRole('tab', { name: 'Cache' }));
  expect(screen.getByRole('tabpanel')).toHaveTextContent('Cache settings are unavailable');
  fireEvent.click(screen.getByRole('tab', { name: 'Advanced' }));
  expect(screen.getByRole('tabpanel')).toHaveTextContent('Cluster nodes and remote routing are unavailable');
  vi.useFakeTimers();
  await act(async () => { await vi.advanceTimersByTimeAsync(12000); });
  expect(requests.filter(request => !['/api/v1/sessions', '/api/v1/runtime/model-aliases', '/api/v1/auth/policy'].includes(request.path))).toEqual([]);
});

it('keeps native downloads and future recommendations visible without automatic keyword or catalog requests', async () => {
  renderApp('/model-hub');
  const search = await screen.findByRole('textbox', { name: 'Search models' });
  expect(search).toHaveAttribute('placeholder', 'owner/repo or repository URL');
  expect(screen.getByRole('button', { name: 'Search' })).toBeDisabled();
  const reclaim = screen.getByRole('checkbox', { name: 'Reclaim SSD prefix budget' });
  expect(reclaim).toBeDisabled();
  fireEvent.change(search, { target: { value: 'keyword' } });
  expect(screen.getByRole('button', { name: 'Search' })).toBeDisabled();
  fireEvent.click(screen.getByRole('tab', { name: 'Recommended models' }));
  expect(screen.getByText(/does not provide recommendations/)).toBeInTheDocument();
  vi.useFakeTimers();
  await act(async () => { await vi.advanceTimersByTimeAsync(12000); });
  expect(requests.filter(request => !['/api/v1/sessions', '/api/v1/jobs/kinds', '/api/v1/hub/download-directory'].includes(request.path))).toEqual([]);
});

it.each(['/chat', '/chat/'])('renders the real native chat page at %s without redirecting or calling Agent APIs', async path => {
  renderApp(path);
  await waitFor(() => expect(screen.getByRole('textbox', { name: 'Message' })).toBeEnabled());
  expect(screen.getByLabelText('Current path')).toHaveTextContent(path);
  expect(screen.getByRole('heading', { name: 'Native conversation' })).toBeInTheDocument();
  expect(requests.some(request => request.path.startsWith('/api/v1/agent'))).toBe(false);
  expect(screen.getByRole('button', { name: 'Text chat' })).toHaveAttribute('aria-current', 'page');
});

it('explains native Agent unavailability and opens the actual text chat route', async () => {
  renderApp('/agent?workspace=old-workspace&session=old-session');
  expect(await screen.findByRole('heading', { name: 'Agent is unavailable on this server' })).toBeInTheDocument();
  expect(screen.queryByRole('combobox', { name: 'Workspace' })).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Add workspace' })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Open text chat' }));
  await waitFor(() => expect(screen.getByRole('textbox', { name: 'Message' })).toBeEnabled());
  expect(screen.getByLabelText('Current path').textContent).toBe('/chat');
  expect(requests.some(request => request.path.startsWith('/api/v1/agent'))).toBe(false);
});

it('retains the real Agent page, startup and workspace entry on services supporting Agent', async () => {
  runtime.runtime = { model: 'Native Model', runtime_state: 'ready' };
  renderApp('/agent', true);
  expect(await screen.findByRole('heading', { name: 'Start something new' })).toBeInTheDocument();
  await waitFor(() => expect(screen.getByRole('button', { name: 'Add workspace' })).toBeEnabled());
  expect(screen.getByRole('button', { name: 'New conversation' })).toBeInTheDocument();
  expect(screen.queryByRole('heading', { name: 'Agent is unavailable on this server' })).not.toBeInTheDocument();
  expect(requests).toEqual(expect.arrayContaining([
    { path: '/api/v1/agent/state', method: 'GET' }, { path: '/api/v1/agent/start', method: 'POST' },
  ]));
});

it('allows a native idle server to open and create text chats, then guides model selection', async () => {
  runtime.runtime = { source: 'native-instances', runtime_state: 'idle' };
  runtime.models = []; runtime.instances = []; runtime.selectedModel = '';
  sessions = [];
  renderApp('/chat');
  fireEvent.click(await screen.findByRole('button', { name: 'Start chat' }));
  expect(await screen.findByRole('heading', { name: 'No model loaded' })).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Choose model' })).toBeEnabled();
  expect(screen.getByRole('textbox', { name: 'Message' })).toBeEnabled();
  expect(screen.getByRole('button', { name: 'Send' })).toBeDisabled();
  expect(requests.filter(request => request.path === '/api/v1/sessions' && request.method === 'POST')).toHaveLength(1);
});

it('keeps a real text stream alive across App routes and synchronizes persisted output on return', async () => {
  renderApp('/chat', true);
  await waitFor(() => expect(screen.getByRole('textbox', { name: 'Message' })).toBeEnabled());
  fireEvent.change(screen.getByRole('textbox', { name: 'Message' }), { target: { value: 'Hello native' } });
  fireEvent.click(screen.getByRole('button', { name: 'Send' }));
  await screen.findByRole('button', { name: 'Stop generation' });
  await act(async () => {
    stream.enqueue(frame(0, { type: 'session.state', state: 'processing', revision: 1 }));
    stream.enqueue(frame(1, { type: 'response.text.delta', response_id: responseId, delta: 'Persisted answer' }));
  });
  fireEvent.click(screen.getByRole('button', { name: 'Overview' }));
  await screen.findByRole('heading', { name: 'Overview' });
  expect(generationSignal?.aborted).toBe(false);
  fireEvent.click(within(screen.getByRole('navigation', { name: 'Inference' })).getByRole('button', { name: 'Text chat' }));
  await screen.findByRole('button', { name: 'Stop generation' });
  expect(generationSignal?.aborted).toBe(false);
  sessions = [{ ...savedSession(), revision: 2 }];
  messages = [
    { id: 'user-a', role: 'user', parts: [{ type: 'text', text: 'Hello native' }], parent_id: null, created_at: timestamp },
    { id: 'answer-a', role: 'assistant', parts: [{ type: 'text', text: 'Persisted answer' }], parent_id: 'user-a', created_at: timestamp },
  ];
  responses = [{ id: responseId, request_id: generationRequest.request_id as string, session_id: sessionId,
    status: 'completed', output_message_id: 'answer-a', output: messages[1].parts, finish_reason: 'stop', created_at: timestamp }];
  await act(async () => {
    stream.enqueue(frame(2, { type: 'response.completed', response_id: responseId, finish_reason: 'stop' }));
    stream.enqueue(frame(3, { type: 'session.state', state: 'idle', revision: 2 }));
    stream.close();
  });
  await waitFor(() => expect(screen.queryByRole('button', { name: 'Stop generation' })).not.toBeInTheDocument());
  expect(await screen.findByText('Persisted answer')).toBeInTheDocument();
  expect(useConversationStore.getState().messages).toEqual(messages);
  expect(requests.filter(request => request.path.endsWith('/responses') && request.method === 'POST')).toHaveLength(1);
  expect(requests.some(request => request.path.endsWith('/cancel'))).toBe(false);
});

it('opens native chat from the overview and advertises the configured dynamic service port', async () => {
  setApiBaseUrl('http://127.0.0.1:59421');
  renderApp('/');
  await screen.findByRole('heading', { name: 'Overview' });
  expect(screen.getAllByText('http://127.0.0.1:59421/v1').length).toBeGreaterThan(0);
  expect(screen.getByText('http://127.0.0.1:59421/v1/messages')).toBeInTheDocument();
  expect(screen.queryByText('http://127.0.0.1:8090/v1')).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Chat' }));
  await waitFor(() => expect(screen.getByRole('textbox', { name: 'Message' })).toBeEnabled());
  expect(screen.getByLabelText('Current path').textContent).toBe('/chat');
});
