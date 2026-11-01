import { useEffect, useRef, useState } from 'react';
import { FileIcon, FolderIcon, RobotIcon } from '@phosphor-icons/react';
import { useSearchParams } from 'react-router';
import { apiUrl, authorizedHeaders, errorFromResponse, request } from '../../shared/api/client';
import { useSettings } from '../settings/SettingsProvider';
import { useRuntime } from '../../app/RuntimeProvider';
import { renderMarkdown } from '../chat/MessageMarkdown';
import { agentRpc, useAgent } from './AgentProvider';
import { AgentQuestionCard } from './AgentQuestionCard';
import { AgentComposer, type AgentPromptPart, type AgentSelection } from './AgentComposer';
import { agentEvent, emptyJournal, type AgentFrame, type AgentMessage } from './agentEvents';
import './agent.css';

function Message({ message }: { message: AgentMessage }) {
  const { tr } = useSettings();
  return <article className={`agent-message ${message.role}`}>
    {message.role === 'tool' ? <details><summary>{message.name}</summary><pre>{message.text}</pre>{message.result != null && <pre>{message.result}</pre>}</details>
      : <>{message.attachments?.length && <div className="agent-message-attachments">{message.attachments.map((name, index) => <span key={index}><FileIcon size={14} />{name}</span>)}</div>}{message.reasoning && <details><summary>{tr('推理', 'Reasoning')}</summary>{renderMarkdown(message.reasoning)}</details>}{renderMarkdown(message.text)}</>}
  </article>;
}

export function AgentPage() {
  const agent = useAgent()!;
  const { tr } = useSettings();
  const { ready } = useRuntime();
  const [search, setSearch] = useSearchParams();
  const sessionId = search.get('session');
  const conversationTitle = agent.state.sessions.find(item => item.sessionId === sessionId)?.projections?.values?.title?.trim() || tr('未命名对话', 'Untitled conversation');
  const workspace = agent.state.workspaces.find(item => item.workspaceId === search.get('workspace')) || agent.state.workspaces.find(item => sessionId && item.sessionIds.includes(sessionId));
  const [journal, setJournal] = useState(emptyJournal);
  const [error, setError] = useState('');
  const [opening, setOpening] = useState(false);
  const [reconnect, setReconnect] = useState(0);
  const [controlRevision, setControlRevision] = useState(0);
  const started = useRef(false);
  const bottom = useRef<HTMLDivElement>(null);
  useEffect(() => { if (ready && !started.current) { started.current = true; void agent.start(); } }, [ready, agent.start]);
  const workspaceId = workspace?.workspaceId;
  useEffect(() => {
    setOpening(false);
    if (!agent.state.running || !workspaceId || sessionId) return;
    let cancelled = false;
    setOpening(true);
    void (async () => {
      try {
        const result = await agent.connectWorkspace(workspaceId);
        if (cancelled) return;
        await agent.refresh();
        if (!cancelled) setSearch({ workspace: workspaceId, session: result.sessionId }, { replace: true });
      } catch (cause) { if (!cancelled) setError(cause instanceof Error ? cause.message : String(cause)); }
      finally { if (!cancelled) setOpening(false); }
    })();
    return () => { cancelled = true; };
  }, [agent.state.running, workspaceId, sessionId, reconnect, agent.refresh, agent.connectWorkspace, setSearch]);
  useEffect(() => {
    setJournal(emptyJournal()); setError('');
    if (!sessionId || !agent.state.running) return;
    const controller = new AbortController();
    void (async () => {
      try {
        const response = await fetch(apiUrl(`/api/v1/agent/sessions/${encodeURIComponent(sessionId)}/events`), { headers: authorizedHeaders(), signal: controller.signal });
        if (!response.ok) throw await errorFromResponse(response);
        const reader = response.body!.pipeThrough(new TextDecoderStream()).getReader();
        let pending = '';
        for (;;) {
          const { value, done } = await reader.read();
          if (done) throw new Error(tr('Agent 连接已中断，请重连。', 'The agent connection was interrupted. Reconnect to continue.'));
          pending += value;
          let split;
          while ((split = pending.indexOf('\n')) >= 0) {
            const line = pending.slice(0, split); pending = pending.slice(split + 1);
            if (!line.trim()) continue;
            const frame = JSON.parse(line) as AgentFrame;
            if (frame.type === 'connection-error') throw new Error(tr('Agent 连接已中断，请重连。', 'The agent connection was interrupted. Reconnect to continue.'));
            if (!controller.signal.aborted) {
              setJournal(current => agentEvent(current, frame));
              if (frame.event && ['plan/mode', 'permission/preset', 'model/selection', 'turn/end'].includes(frame.event.type)) setControlRevision(value => value + 1);
            }
          }
        }
      } catch (cause) { if (!controller.signal.aborted) setError(cause instanceof Error ? cause.message : String(cause)); }
    })();
    return () => controller.abort();
  }, [sessionId, agent.state.running, reconnect, tr]);
  useEffect(() => { bottom.current?.scrollIntoView({ block: 'end' }); }, [journal.messages.length, journal.live?.text]);
  const running = journal.running || agent.state.sessions.some(item => item.sessionId === sessionId && item.running);
  async function send(content: AgentPromptPart[], selection: AgentSelection, title: string) {
    if (!workspace) throw new Error(tr('请先选择工作区。', 'Choose a workspace first.'));
    setError('');
    let id = sessionId;
    if (!id) {
      const result = await agent.connectWorkspace(workspace.workspaceId);
      id = result.sessionId;
      setSearch({ workspace: workspace.workspaceId, session: id });
    }
    if (agent.state.sessions.find(item => item.sessionId === id)?.blank) {
      await agentRpc('session/rename', { request: { sessionId: id, title: title.slice(0, 60) } });
    }
    await agentRpc('session/selectModel', { request: { sessionId: id, ...selection } });
    await agentRpc('session/prompt', { request: { sessionId: id, requestId: crypto.randomUUID(), mode: 'queue', content } });
    await agent.refresh();
  }
  return <section className="agent-page">
    <header className="agent-page-header"><div><h1 title={conversationTitle}>{conversationTitle}</h1></div>
      <div><FolderIcon size={14} /><select aria-label={tr('工作区', 'Workspace')} value={workspace?.workspaceId || ''} onChange={event => setSearch({ workspace: event.target.value })}>
        {!workspace && <option value="">{tr('选择工作区', 'Choose a workspace')}</option>}{agent.state.workspaces.map(item => <option key={item.workspaceId} value={item.workspaceId}>{item.title}</option>)}
      </select><button type="button" onClick={() => void agent.start()} disabled={agent.starting}>{tr('重连', 'Reconnect')}</button></div>
    </header>
    <div className="agent-conversation">
      {!journal.messages.length && !journal.live && <div className="agent-welcome"><RobotIcon size={36} /><h2>{agent.starting ? tr('正在启动 Agent…', 'Starting the agent…') : opening ? tr('正在打开会话…', 'Opening the conversation…') : tr('开始一项工作', 'Start something new')}</h2>
        <p>{!agent.state.running ? tr('由 DeepSeek Harness 驱动。', 'Powered by DeepSeek Harness.') : !workspace ? agent.state.workspaces.length ? tr('选择一个工作区开始。', 'Choose a workspace to begin.') : tr('先在左侧添加一个工作区。', 'Add a workspace in the sidebar to begin.') : tr('描述你的任务，Agent 会在当前工作区中执行。', 'Describe a task for the agent to work on in this workspace.')}</p></div>}
      {journal.messages.map(message => <Message key={message.id} message={message} />)}
      {journal.live && <Message message={journal.live} />}
      {agent.state.approvals.map(item => <div className="agent-approval" key={item.eventId}><strong>{item.request?.toolName}</strong><p>{item.request?.reason}</p><div>
        {[true, false].map(allow => <button key={String(allow)} type="button" onClick={() => void request('/api/v1/agent/approvals', { method: 'POST', body: JSON.stringify({ event_id: item.eventId, allow }) }).then(agent.refresh).catch(cause => setError(String(cause)))}>{allow ? tr('允许这一次', 'Allow once') : tr('拒绝', 'Deny')}</button>)}
      </div></div>)}
      {agent.state.questions?.map(item => <AgentQuestionCard key={item.eventId} pending={item} onAnswered={agent.refresh} />)}
      {(error || journal.error || agent.error || agent.state.error) && <div className="agent-error" role="alert">{error || journal.error || agent.error || agent.state.error}<button type="button" onClick={() => { void agent.start(); setReconnect(value => value + 1); }}>{tr('重连', 'Reconnect')}</button></div>}
      <div ref={bottom} />
    </div>
    <AgentComposer key={sessionId || workspaceId || 'unselected'} sessionId={sessionId} locked={!workspace || !agent.state.running || opening} running={running} revision={controlRevision} onSend={send}
      onStop={async () => { await agentRpc('session/cancel', { request: { sessionId } }); }} />
  </section>;
}
