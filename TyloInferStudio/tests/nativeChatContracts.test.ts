import { afterEach, expect, it, vi } from 'vitest';
import { GenerationController } from '../src/features/chat/state/generationController';
import { sessionsApi } from '../src/shared/api/resources/sessions';
import { setApiBaseUrl } from '../src/shared/api/client';
import type { StreamRequest } from '../src/shared/api/types';

const sessionId = '11111111-1111-4111-8111-111111111111';
const responseId = '22222222-2222-4222-8222-222222222222';
const request: StreamRequest = {
  request_id: '33333333-3333-4333-8333-333333333333', model: 'Local Model', expected_revision: 0,
  input: [{ type: 'text', text: 'Hello' }], input_role: 'user', stream: true, include_reasoning_history: true,
  sampling: { max_tokens: 32, temperature: 0.7, top_k: 40, top_p: 0.9, presence_penalty: 0,
    frequency_penalty: 0, repetition_penalty: 1, enable_thinking: false, enable_vision: false, enable_mtp: false },
};
const timestamp = '2026-11-02T00:00:00+00:00';
const session = { id: sessionId, model: 'Local Model', mode: 'text', state: 'idle', revision: 2,
  title: null, runtime_instance_id: 'instance-a', created_at: timestamp, updated_at: timestamp, metadata: {} };
const user = { id: 'user-a', role: 'user', parts: request.input, parent_id: null, created_at: timestamp };
const answer = { id: 'answer-a', role: 'assistant', parts: [{ type: 'text', text: 'Hi' }], parent_id: 'user-a', created_at: timestamp };
const json = (value: unknown) => new Response(JSON.stringify(value));
const frame = (sequence: number, payload: unknown) => `id: ${sequence}\ndata: ${JSON.stringify({
  protocol_version: '1.0', session_id: sessionId, sequence, timestamp, payload,
})}\n\n`;
const response = (status: string) => ({ id: responseId, request_id: request.request_id,
  session_id: sessionId, status, output_message_id: status === 'completed' ? answer.id : null,
  output: status === 'completed' ? answer.parts : [], finish_reason: status === 'completed' ? 'stop' : null,
  usage: null, performance: null, settings: { sampling: request.sampling }, created_at: timestamp, completed_at: timestamp });

afterEach(() => { vi.unstubAllGlobals(); setApiBaseUrl(''); });

it('consumes the native text SSE sequence and replaces live output with persisted history', async () => {
  const events = frame(0, { type: 'session.state', state: 'processing', revision: 1 })
    + frame(1, { type: 'response.reasoning.delta', response_id: responseId, delta: 'Think' })
    + frame(2, { type: 'response.text.delta', response_id: responseId, delta: 'Hi' })
    + frame(3, { type: 'response.completed', response_id: responseId, finish_reason: 'stop', usage: null, performance: null })
    + frame(4, { type: 'session.state', state: 'idle', revision: 2 });
  const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
    if (init?.method === 'POST' && url.endsWith('/responses')) return new Response(events, { headers: { 'Content-Type': 'text/event-stream; charset=utf-8' } });
    if (url.endsWith('/messages')) return json({ data: [user, answer] });
    if (url.includes('/responses?')) return json({ data: [response('completed')] });
    if (url.endsWith(sessionId)) return json(session);
    throw new Error(`Unexpected native route: ${url}`);
  });
  vi.stubGlobal('fetch', fetchMock);
  const synchronized = vi.fn();
  const state = vi.fn();
  const accepted = vi.fn();
  const controller = new GenerationController({ onSynchronized: synchronized, onSessionState: state });
  await controller.start(sessionId, request, accepted);
  expect(controller.getSnapshot()).toMatchObject({ phase: 'completed', live: null, error: null, recoveryNeeded: false });
  expect(accepted).toHaveBeenCalledOnce();
  expect(synchronized).toHaveBeenCalledWith({ session, messages: [user, answer], responses: [response('completed')] });
  expect(state).toHaveBeenLastCalledWith(sessionId, 'idle', 2);
  expect(JSON.parse(fetchMock.mock.calls[0][1]!.body as string)).toEqual(request);
  const restored = await sessionsApi.listMessages(sessionId);
  expect(restored).toEqual([user, answer]);
  expect(fetchMock.mock.calls.filter(([, init]) => init?.method === 'POST')).toHaveLength(1);
  controller.reset();
});

it('stops through the native cancel route, then confirms the persisted cancelled response', async () => {
  let cancelled = false;
  let stream!: ReadableStreamDefaultController<Uint8Array>;
  const body = new ReadableStream<Uint8Array>({ start(controller) { stream = controller; } });
  const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
    if (init?.method === 'POST' && url.endsWith('/responses')) return new Response(body, { headers: { 'Content-Type': 'text/event-stream' } });
    if (url.endsWith('/responses/cancel')) { cancelled = true; return json(response('running')); }
    if (url.endsWith('/messages')) return json({ data: [user] });
    if (url.includes('/responses?')) return json({ data: [response(cancelled ? 'cancelled' : 'running')] });
    if (url.endsWith(sessionId)) return json({ ...session, state: 'interrupted', revision: 1 });
    throw new Error(`Unexpected native route: ${url}`);
  });
  vi.stubGlobal('fetch', fetchMock);
  const synchronized = vi.fn();
  const controller = new GenerationController({ onSynchronized: synchronized, onSessionState: vi.fn() });
  const running = controller.start(sessionId, request);
  stream.enqueue(new TextEncoder().encode(frame(0, { type: 'session.state', state: 'processing', revision: 1 })));
  await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledOnce());
  await controller.stop();
  await running;
  expect(cancelled).toBe(true);
  expect(controller.getSnapshot()).toMatchObject({ phase: 'cancelled', recoveryNeeded: false });
  expect(synchronized.mock.calls[0][0].messages).toEqual([user]);
  expect(fetchMock.mock.calls.filter(([, init]) => init?.method === 'POST').map(([url]) => url))
    .toEqual([`/api/v1/sessions/${sessionId}/responses`, `/api/v1/sessions/${sessionId}/responses/cancel`]);
  controller.reset();
});

it.each([true, false])('shows persisted native failure details with error frame present=%s', async withErrorFrame => {
  const error = { code: 'model_load_failed', message: 'No supported native backend', retryable: false, details: {} };
  vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
    if (init?.method === 'POST') return new Response(
      frame(0, { type: 'session.state', state: 'processing', revision: 1 })
      + (withErrorFrame ? frame(1, { type: 'error', error }) : '')
      + frame(withErrorFrame ? 2 : 1, { type: 'session.state', state: 'error', revision: 1 }),
      { headers: { 'Content-Type': 'text/event-stream' } });
    if (url.endsWith('/messages')) return json({ data: [user] });
    if (url.includes('/responses?')) return json({ data: [{ ...response('failed'), error }] });
    return json({ ...session, state: 'error', revision: 1 });
  }));
  const synchronized = vi.fn();
  const controller = new GenerationController({ onSynchronized: synchronized, onSessionState: vi.fn() });
  await controller.start(sessionId, request);
  expect(controller.getSnapshot()).toMatchObject({ phase: 'failed', error: 'No supported native backend', recoveryNeeded: false });
  expect(synchronized).toHaveBeenCalledOnce();
  controller.reset();
});
