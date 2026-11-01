import { expect, it } from 'vitest';
import { agentEvent, emptyJournal } from './agentEvents';

it('keeps attachment-only user messages in the restored conversation', () => {
  const state = agentEvent(emptyJournal(), { type: 'event', event: { seq: 1, type: 'user/message', data: {
    content: [{ type: 'file', attachment: { name: 'note.txt' } }, { type: 'image', attachment: { name: 'figure.png' } }],
  } } });
  expect(state.messages).toHaveLength(1);
  expect(state.messages[0]).toMatchObject({ role: 'user', attachments: ['note.txt', 'figure.png'] });
});

it('replaces a reconnect snapshot and deduplicates durable events', () => {
  const event = { type: 'event', event: { seq: 1, type: 'user/message', data: { content: [{ type: 'text', text: 'hello' }] } } };
  const state = agentEvent(emptyJournal(), { type: 'snapshot', cursor: 1, records: [event] });
  expect(state.messages).toHaveLength(1);
  expect(agentEvent(state, event)).toBe(state);
  expect(agentEvent(state, { type: 'snapshot', cursor: 1, records: [event] }).messages).toHaveLength(1);
});

it('applies history replacements and exposes failed turns instead of silently stopping', () => {
  let state = agentEvent(emptyJournal(), { type: 'event', event: { seq: 1, type: 'assistant/message', data: { message: { content: [{ type: 'text', text: 'Old answer' }] } } } });
  state = agentEvent(state, { type: 'event', event: { seq: 2, type: 'assistant/message', surfaceOp: { op: 'replace', startSeq: 1, endSeq: 1 }, data: { message: { content: [{ type: 'text', text: 'Replacement' }] } } } });
  expect(state.messages.map(item => item.text)).toEqual(['Replacement']);
  state = agentEvent(state, { type: 'event', event: { seq: 3, type: 'turn/end', data: { reason: { kind: 'error', error: { message: 'Model unavailable' } } } } });
  expect(state.error).toBe('Model unavailable');
  expect(state.running).toBe(false);
});

it('reconciles streamed assistant text with the authoritative message and tracks tool results', () => {
  let state = agentEvent(emptyJournal(), { type: 'assistant-stream', frame: { type: 'chunk', attemptId: 'attempt', chunk: { type: 'text-delta', text: 'hello' } } });
  expect(state.live?.text).toBe('hello');
  state = agentEvent(state, { type: 'event', event: { seq: 2, type: 'assistant/message', data: { message: { content: [{ type: 'text', text: 'hello world' }] } } } });
  expect(state.live).toBeNull();
  expect(state.messages.map(item => item.text)).toEqual(['hello world']);
  state = agentEvent(state, { type: 'event', event: { seq: 3, type: 'tool/call', data: { callId: 'tool', name: 'read', arguments: { path: 'note.txt' } } } });
  state = agentEvent(state, { type: 'event', event: { seq: 4, type: 'tool/result', data: { message: { toolCallId: 'tool', content: [{ type: 'text', text: 'marker' }] } } } });
  expect(state.messages.at(-1)).toMatchObject({ name: 'read', result: 'marker' });
});
