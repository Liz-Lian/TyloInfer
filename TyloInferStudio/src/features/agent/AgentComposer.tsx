import { useEffect, useRef, useState, type ReactNode } from 'react';
import { ArrowUpIcon, BrainIcon, CaretDownIcon, CheckIcon, CircleNotchIcon, FileIcon, LightningIcon, ListChecksIcon, PaperclipIcon, PlusIcon, ShieldCheckIcon, ShieldStarIcon, ShieldWarningIcon, StopIcon, XIcon } from '@phosphor-icons/react';
import { apiUrl, authorizedHeaders, errorFromResponse, request } from '../../shared/api/client';
import { Dialog } from '../../shared/ui/Dialog';
import { useSettings } from '../settings/SettingsProvider';
import { agentRpc, useAgent } from './AgentProvider';

export interface AgentSelection { provider: string; model: string; reasoningEffort?: string }
export type AgentPromptPart = { type: 'text'; text: string } | { type: 'file'; receiptId: string } | { type: 'image'; name: string; mediaType: string; data: string };
interface Command { name: string; description: string; input?: { hint: string; attachments?: boolean } }
interface Controls { values: { modelSelection?: { next: AgentSelection | null }; plan?: { active: boolean; pending: boolean | null }; permissions?: { currentValue: string } }; commands: Command[]; permissions: { value: string; name: string; description?: string }[] }
interface Attachment { id: string; name: string; part: AgentPromptPart }

function PermissionShield({ value }: { value?: string }) {
  const props = { size: 16, weight: 'regular' as const, className: 'agent-permission-shield', 'data-permission': value, 'aria-hidden': true as const };
  if (value === 'danger-full-access') return <ShieldWarningIcon {...props} />;
  if (value === 'auto') return <ShieldStarIcon {...props} />;
  if (value === 'read-only') return <svg width="16" height="16" viewBox="0 0 256 256" fill="none" stroke="currentColor" strokeWidth="16" strokeLinecap="round" strokeLinejoin="round" className={props.className} data-permission={value} aria-hidden="true">
    <path d="M128 24 40 56v64c0 64 88 112 88 112s88-48 88-112V56Z" />
    <rect x="96" y="112" width="64" height="48" rx="8" /><path d="M108 112V96a20 20 0 0 1 40 0v16" />
  </svg>;
  return <ShieldCheckIcon {...props} />;
}

function ComposerMenu({ label, detail, icon, children, disabled = false, align = 'left' }: { label: string; detail?: string; icon?: ReactNode; children: ReactNode; disabled?: boolean; align?: 'left' | 'right' }) {
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    if (!open) return;
    const close = (event: PointerEvent) => { if (!root.current?.contains(event.target as Node)) setOpen(false); };
    const escape = (event: KeyboardEvent) => { if (event.key === 'Escape') { event.stopPropagation(); setOpen(false); trigger.current?.focus(); } };
    document.addEventListener('pointerdown', close);
    document.addEventListener('keydown', escape);
    const first = root.current?.querySelector<HTMLButtonElement>('[role=menu] button:not(:disabled)');
    first?.focus();
    return () => { document.removeEventListener('pointerdown', close); document.removeEventListener('keydown', escape); };
  }, [open]);
  return <div className={`agent-composer-menu ${align}`} ref={root}>
    <button ref={trigger} type="button" className="agent-composer-chip" aria-label={label} title={label} aria-haspopup="menu" aria-expanded={open} disabled={disabled}
      onClick={() => setOpen(value => !value)}>{icon}<span>{label}</span>{detail && <small><BrainIcon size={12} />{detail}</small>}<CaretDownIcon size={10} /></button>
    {open && <div role="menu" aria-label={label} className="agent-composer-popover" style={{ left: Math.max(12, Math.min(window.innerWidth - 292, align === 'right' ? (root.current?.getBoundingClientRect().right || 292) - 280 : root.current?.getBoundingClientRect().left || 12)), bottom: window.innerHeight - (root.current?.getBoundingClientRect().top || window.innerHeight - 12) + 8 }} onClick={event => { if ((event.target as Element).closest('[role^=menuitem]')) { setOpen(false); trigger.current?.focus(); } }}
      onKeyDown={event => {
        if (!['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) return;
        const buttons = [...event.currentTarget.querySelectorAll<HTMLButtonElement>('button:not(:disabled), input')];
        const index = buttons.indexOf(document.activeElement as HTMLButtonElement);
        const next = event.key === 'Home' ? 0 : event.key === 'End' ? buttons.length - 1 : (index + (event.key === 'ArrowUp' ? -1 : 1) + buttons.length) % buttons.length;
        event.preventDefault(); buttons[next]?.focus();
      }}>{children}</div>}
  </div>;
}

export function AgentComposer({ sessionId, locked, running, revision, onSend, onStop }: { sessionId: string | null; locked: boolean; running: boolean; revision: number; onSend(content: AgentPromptPart[], selection: AgentSelection, title: string): Promise<void>; onStop(): Promise<void> }) {
  const agent = useAgent()!;
  const { tr } = useSettings();
  const [controls, setControls] = useState<Controls | null>(null);
  const [draft, setDraft] = useState('');
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const [busy, setBusy] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [permissionConfirm, setPermissionConfirm] = useState(false);
  const [commandSearch, setCommandSearch] = useState('');
  const fileInput = useRef<HTMLInputElement>(null);
  const editor = useRef<HTMLTextAreaElement>(null);
  const alive = useRef(true);
  useEffect(() => { alive.current = true; return () => { alive.current = false; }; }, []);
  async function refresh(signal?: AbortSignal) {
    if (!sessionId) return;
    const value = await request<Controls>(`/api/v1/agent/sessions/${encodeURIComponent(sessionId)}/controls`, { signal });
    if (alive.current && !signal?.aborted) setControls(value);
  }
  useEffect(() => {
    if (!sessionId || locked) return;
    const controller = new AbortController();
    setLoading(true);
    void refresh(controller.signal).catch(cause => { if (!controller.signal.aborted) setError(String(cause.message || cause)); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [sessionId, locked, revision]);
  const saved = controls?.values.modelSelection?.next;
  const selected = saved || { provider: 'tyloinfer', model: agent.state.models[0]?.id || '' };
  const model = agent.state.models.find(item => item.id === selected.model);
  const available = Boolean(model);
  const disabled = locked || !sessionId || !controls || busy || loading || uploading;
  const mtp = selected.provider === 'tyloinfer-mtp-off' ? 'off' : selected.provider === 'tyloinfer-mtp-on' ? 'on' : 'auto';
  const plan = controls?.values.plan?.pending ?? controls?.values.plan?.active ?? false;
  const permission = controls?.values.permissions?.currentValue;
  const permissionLabel = (value?: string) => ({ 'read-only': tr('只读', 'Read only'), 'workspace-write': tr('工作区权限', 'Workspace access'), 'danger-full-access': tr('完全访问', 'Full access'), auto: tr('自动审批', 'Auto review') })[value || ''] || controls?.permissions.find(item => item.value === value)?.name || tr('权限', 'Permissions');
  const effortLabel = (value: string) => ({ off: tr('关闭', 'Off'), minimal: tr('最低', 'Minimal'), low: tr('低', 'Low'), medium: tr('中', 'Medium'), high: model?.thinkingToggleOnly ? tr('开启', 'On') : tr('高', 'High'), xhigh: tr('极高', 'Extra high'), max: tr('最高', 'Max') })[value] || value;
  async function operation(action: () => Promise<void>) {
    setBusy(true); setError(''); setNotice('');
    try { await action(); }
    catch (cause) { if (alive.current) setError(cause instanceof Error ? cause.message : String(cause)); }
    finally { if (alive.current) setBusy(false); }
  }
  async function select(selection: AgentSelection) {
    await operation(async () => { await agentRpc('session/selectModel', { request: { sessionId, ...selection } }); await refresh(); });
  }
  async function command(line: string, parts: AgentPromptPart[] = []) {
    const result = await request<{ result: { text?: string } }>(`/api/v1/agent/sessions/${encodeURIComponent(sessionId!)}/commands`, {
      method: 'POST', body: JSON.stringify({ line, attachments: parts }) });
    if (line === '/export') {
      const response = await fetch(apiUrl(`/api/v1/agent/sessions/${encodeURIComponent(sessionId!)}/export`), { headers: authorizedHeaders() });
      if (!response.ok) throw await errorFromResponse(response);
      const url = URL.createObjectURL(await response.blob());
      const link = document.createElement('a'); link.href = url; link.download = 'conversation.zip'; link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    }
    if (alive.current) setNotice(result.result?.text || tr('已应用', 'Applied'));
    await refresh(); await agent.refresh();
  }
  async function submit() {
    if (disabled || (!draft.trim() && !attachments.length)) return;
    const line = draft.trim();
    const parts = attachments.map(item => item.part);
    if (!line.startsWith('/') && !available) return;
    await operation(async () => {
      if (line.startsWith('/')) await command(line, parts);
      else await onSend([...(line ? [{ type: 'text' as const, text: line }] : []), ...parts], selected, line || attachments[0].name);
      if (alive.current) { setDraft(''); setAttachments([]); }
    });
  }
  async function upload(files: File[]) {
    if (!sessionId || disabled) return;
    setUploading(true); setError('');
    try {
      for (const file of files) {
        if (file.size > 64 * 1024 ** 2) throw new Error(tr('单个文件不能超过 64 MiB。', 'Each file must be no larger than 64 MiB.'));
        let part: AgentPromptPart;
        if (['image/png', 'image/jpeg', 'image/webp', 'image/gif'].includes(file.type)) {
          if (!model?.input?.includes('image')) throw new Error(tr('当前模型不支持图像输入。', 'The selected model does not support image input.'));
          const data = await new Promise<string>((resolve, reject) => { const reader = new FileReader(); reader.onload = () => resolve(String(reader.result).split(',')[1]); reader.onerror = () => reject(new Error(tr('无法读取文件。', 'Could not read the file.'))); reader.readAsDataURL(file); });
          part = { type: 'image', name: file.name, mediaType: file.type, data };
        } else {
          const receipt = await request<{ receiptId: string }>(`/api/v1/agent/sessions/${encodeURIComponent(sessionId)}/files?name=${encodeURIComponent(file.name)}`, { method: 'POST', headers: { 'Content-Type': 'application/octet-stream' }, body: file });
          part = { type: 'file', receiptId: receipt.receiptId };
        }
        if (alive.current) setAttachments(current => [...current, { id: crypto.randomUUID(), name: file.name, part }]);
      }
    } catch (cause) { if (alive.current) setError(cause instanceof Error ? cause.message : String(cause)); }
    finally { if (alive.current) setUploading(false); }
  }
  const commands = controls?.commands.filter(item => `${item.name} ${item.description}`.toLowerCase().includes(commandSearch.toLowerCase())) || [];
  return <div className="agent-composer-wrap">
    {error && <p className="agent-composer-error" role="alert">{error}<button type="button" onClick={() => void operation(refresh)}>{tr('重试', 'Retry')}</button></p>}
    {notice && <p className="agent-composer-notice" role="status">{notice}</p>}
    <form className="agent-composer" onSubmit={event => { event.preventDefault(); void submit(); }} onDragOver={event => { event.preventDefault(); }} onDrop={event => { event.preventDefault(); void upload([...event.dataTransfer.files]); }}>
      {attachments.length > 0 && <div className="agent-attachment-rail">{attachments.map(item => <span key={item.id}><FileIcon size={14} /><span>{item.name}</span><button type="button" aria-label={`${tr('移除附件', 'Remove attachment')} ${item.name}`} onClick={() => setAttachments(current => current.filter(value => value.id !== item.id))}><XIcon size={12} /></button></span>)}</div>}
      <textarea ref={editor} aria-label={tr('发送给 Agent', 'Message the agent')} value={draft} onChange={event => setDraft(event.target.value)} placeholder={plan ? tr('描述你的计划…', 'Describe your plan…') : tr('描述任务，或输入 / 使用命令…', 'Describe a task, or type / for commands…')} rows={3}
        onKeyDown={event => { if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); void submit(); } }} />
      <div className="agent-composer-toolbar"><div className="agent-composer-leading">
        <ComposerMenu label={tr('添加与命令', 'Add and commands')} icon={<PlusIcon size={16} />} disabled={disabled}>
          <button type="button" role="menuitem" onClick={() => fileInput.current?.click()}><PaperclipIcon size={15} /><span>{tr('添加文件或图片', 'Attach files or images')}</span></button>
          <div className="agent-menu-divider" /><input className="agent-command-search" aria-label={tr('搜索命令', 'Search commands')} value={commandSearch} onChange={event => setCommandSearch(event.target.value)} placeholder={tr('搜索命令…', 'Search commands…')} />
          <div className="agent-command-list">{commands.map(item => <button type="button" role="menuitem" key={item.name} title={item.description} onClick={() => { setDraft(`/${item.name}${item.input ? ' ' : ''}`); setTimeout(() => editor.current?.focus(), 0); }}><span>/{item.name}</span><small>{item.description}</small></button>)}{!commands.length && <p>{tr('暂无匹配命令', 'No matching commands')}</p>}</div>
        </ComposerMenu>
        <input ref={fileInput} type="file" multiple hidden aria-label={tr('添加附件', 'Attach files')} onChange={event => { void upload([...event.target.files || []]); event.target.value = ''; }} />
        <ComposerMenu label={permissionLabel(permission)} icon={<PermissionShield value={permission} />} disabled={disabled || !controls?.permissions.length}>
          {controls?.permissions.map(item => <button type="button" role="menuitemradio" aria-checked={permission === item.value} key={item.value} title={item.description}
            onClick={() => { if (item.value === 'danger-full-access') setPermissionConfirm(true); else void operation(() => command(`/permission ${item.value}`)); }}><span className="agent-permission-option"><PermissionShield value={item.value} /><span>{permissionLabel(item.value)}</span></span>{permission === item.value && <CheckIcon size={13} />}</button>)}
        </ComposerMenu>
        <ComposerMenu label={plan ? tr('计划', 'Plan') : tr('执行', 'Agent')} icon={<ListChecksIcon size={14} />} disabled={disabled || !controls?.commands.some(item => item.name === 'plan')}>
          {[false, true].map(value => <button type="button" role="menuitemradio" aria-checked={plan === value} key={String(value)} onClick={() => void operation(() => command(value ? '/plan' : '/plan off'))}><span>{value ? tr('计划', 'Plan') : tr('执行', 'Agent')}</span>{plan === value && <CheckIcon size={13} />}</button>)}
        </ComposerMenu>
      </div><div className="agent-composer-trailing">
        <ComposerMenu label={`MTP · ${mtp === 'auto' ? tr('自动', 'Auto') : mtp === 'on' ? tr('开', 'On') : tr('关', 'Off')}`} icon={<LightningIcon size={13} />} disabled={disabled || !available} align="right">
          <small>{tr('当前会话', 'This conversation')}</small>{(['auto', 'on', 'off'] as const).map(value => <button type="button" role="menuitemradio" aria-checked={mtp === value} disabled={value === 'on' && !model?.mtpAvailable} key={value}
            onClick={() => void select({ ...selected, provider: value === 'auto' ? 'tyloinfer' : `tyloinfer-mtp-${value}` })}><span>{value === 'auto' ? tr('跟随服务', 'Follow service') : value === 'on' ? tr('开启', 'On') : tr('关闭', 'Off')}</span>{mtp === value && <CheckIcon size={13} />}</button>)}
          {!model?.mtpAvailable && <p>{tr('当前模型或服务未启用 MTP。', 'MTP is not available for this model or service.')}</p>}
        </ComposerMenu>
        <ComposerMenu label={available ? model!.name : tr('尚未加载模型', 'No model loaded')} detail={model?.reasoningEfforts ? selected.reasoningEffort ? effortLabel(selected.reasoningEffort) : tr('默认', 'Default') : undefined} disabled={disabled || !agent.state.models.length} align="right">
          <small>{tr('模型', 'Model')}</small>{agent.state.models.map(item => <button type="button" role="menuitemradio" aria-checked={item.id === selected.model} key={item.id} onClick={() => void select({ provider: selected.provider, model: item.id })}><span>{item.name}</span>{item.id === selected.model && <CheckIcon size={13} />}</button>)}
          {model?.reasoningEfforts && <><div className="agent-menu-divider" /><small><BrainIcon size={13} /> Thinking</small>
            {[undefined, ...Object.keys(model.reasoningEfforts)].map(effort => <button type="button" role="menuitemradio" aria-checked={selected.reasoningEffort === effort} key={effort || 'default'} onClick={() => void select({ provider: selected.provider, model: selected.model, ...effort ? { reasoningEffort: effort } : {} })}><span>{effort ? effortLabel(effort) : tr('模型默认', 'Model default')}</span>{selected.reasoningEffort === effort && <CheckIcon size={13} />}</button>)}</>}
        </ComposerMenu>
        {running && <button type="button" className="agent-composer-submit" aria-label={tr('停止', 'Stop')} onClick={() => void operation(onStop)}><StopIcon size={15} /></button>}
        <button type="submit" className="agent-composer-submit" aria-label={running ? tr('加入队列', 'Queue message') : tr('发送', 'Send')} disabled={disabled || (!draft.trim() && !attachments.length) || (!draft.trim().startsWith('/') && !available)}>{busy || loading || uploading ? <CircleNotchIcon size={16} /> : <ArrowUpIcon size={16} />}</button>
      </div></div>
    </form>
    <Dialog open={permissionConfirm} onOpenChange={setPermissionConfirm} title={tr('允许完全访问？', 'Allow full access?')} description={tr('Agent 将能在工作区外读写文件和运行命令，且不再逐项请求批准。', 'The agent can read and write outside the workspace and run commands without asking for approval.')} closeLabel={tr('关闭', 'Close')}>
      <footer className="agent-confirm-actions"><button type="button" onClick={() => setPermissionConfirm(false)}>{tr('取消', 'Cancel')}</button><button type="button" onClick={() => { setPermissionConfirm(false); void operation(() => command('/permission danger-full-access')); }}>{tr('允许完全访问', 'Allow full access')}</button></footer>
    </Dialog>
  </div>;
}
