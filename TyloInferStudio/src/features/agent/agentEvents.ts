export interface AgentMessage { id: string; role: 'user' | 'assistant' | 'tool'; text: string; reasoning?: string; name?: string; result?: string; seq?: number; attachments?: string[] }
export interface AgentJournal { messages: AgentMessage[]; live: AgentMessage | null; running: boolean; cursor: number; error: string | null }
export const emptyJournal = (): AgentJournal => ({ messages: [], live: null, running: false, cursor: -1, error: null });
type Part = { type: string; text?: string; name?: string; attachment?: { name?: string } };
type Chunk = { type?: string; text?: string };
export interface AgentFrame {
  type: string; records?: AgentFrame[]; cursor?: number;
  assistantStream?: { activeAttempt?: { attemptId: string; stream: Chunk[] } };
  frame?: { type: string; attemptId: string; chunk?: Chunk };
  event?: { seq: number; type: string; surfaceOp?: 'append' | { op: 'replace'; startSeq: number; endSeq: number }; data?: { content?: Part[]; message?: { content?: Part[]; toolCallId?: string }; callId?: string; name?: string; arguments?: unknown; reason?: { kind: string; error?: { message?: string } } } };
}
const text = (parts: Part[] = [], kind = 'text') => parts.filter(part => part.type === kind).map(part => part.text || '').join('');

export function agentEvent(state: AgentJournal, frame: AgentFrame): AgentJournal {
  if (frame.type === 'snapshot') {
    let next = emptyJournal();
    for (const record of frame.records || []) next = agentEvent(next, record);
    const active = frame.assistantStream?.activeAttempt;
    if (active) for (const chunk of active.stream) next = agentEvent(next, { type: 'assistant-stream', frame: { type: 'chunk', attemptId: active.attemptId, chunk } });
    return { ...next, cursor: frame.cursor ?? next.cursor };
  }
  if (frame.type === 'assistant-stream' && frame.frame?.type === 'chunk') {
    const { chunk, attemptId } = frame.frame;
    if (!chunk?.text || !['text-delta', 'reasoning-delta'].includes(chunk.type || '')) return state;
    const live = state.live?.id === attemptId ? { ...state.live } : { id: attemptId, role: 'assistant' as const, text: '', reasoning: '' };
    if (chunk.type === 'text-delta') live.text += chunk.text; else live.reasoning = (live.reasoning || '') + chunk.text;
    return { ...state, live, running: true };
  }
  const event = frame.event;
  if (frame.type !== 'event' || !event || event.seq <= state.cursor) return state;
  const next = { ...state, cursor: event.seq, messages: [...state.messages] };
  const replacement = event.surfaceOp;
  if (typeof replacement === 'object' && replacement.op === 'replace') next.messages = next.messages.filter(item => item.seq == null || item.seq < replacement.startSeq || item.seq > replacement.endSeq);
  const data = event.data || {};
  if (event.type === 'user/message' || event.type === 'assistant/message') {
    const body = text(data.message?.content || data.content);
    const reasoning = text(data.message?.content || data.content, 'reasoning');
    const attachments = (data.message?.content || data.content || []).filter(part => part.type === 'file' || part.type === 'image').map(part => part.attachment?.name || part.name || part.type);
    if (body || reasoning || attachments.length) next.messages.push({ id: `event-${event.seq}`, seq: event.seq, role: event.type === 'user/message' ? 'user' : 'assistant', text: body, reasoning, ...(attachments.length ? { attachments } : {}) });
    if (event.type === 'assistant/message') next.live = null;
  } else if (event.type === 'tool/call') {
    next.messages.push({ id: data.callId || `event-${event.seq}`, seq: event.seq, role: 'tool', name: data.name,
      text: typeof data.arguments === 'string' ? data.arguments : JSON.stringify(data.arguments ?? {}) });
  } else if (event.type === 'tool/result') {
    next.messages = next.messages.map(item => item.id === data.message?.toolCallId ? { ...item, result: text(data.message?.content).slice(0, 50_000) } : item);
  } else if (event.type === 'turn/start') { next.running = true; next.error = null; }
  else if (event.type === 'turn/end') {
    next.running = false; next.live = null;
    if (data.reason?.kind === 'error') next.error = data.reason.error?.message || 'The agent turn failed.';
  }
  return next;
}
