import { useState } from 'react';
import { request } from '../../shared/api/client';
import { useSettings } from '../settings/SettingsProvider';
import { renderMarkdown } from '../chat/MessageMarkdown';
import type { AgentQuestion } from './AgentProvider';

export function AgentQuestionCard({ pending, onAnswered }: { pending: AgentQuestion; onAnswered(): Promise<void> }) {
  const { tr } = useSettings();
  const [answers, setAnswers] = useState<Record<string, { selected: string[]; custom: string }>>({});
  const [error, setError] = useState('');
  const [sending, setSending] = useState(false);
  async function submit() {
    setSending(true); setError('');
    try {
      await request('/api/v1/agent/answers', { method: 'POST', body: JSON.stringify({ event_id: pending.eventId,
        answers: pending.request.questions.map(question => ({ id: question.id, selected: answers[question.id]?.selected || [], custom: answers[question.id]?.custom || '' })) }) });
      await onAnswered();
    } catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); }
    finally { setSending(false); }
  }
  return <form className="agent-question" onSubmit={event => { event.preventDefault(); void submit(); }}>
    {pending.request.questions.map(question => <fieldset key={question.id}><legend>{question.question}</legend>
      {question.detail && <div className="agent-question-detail">{renderMarkdown(question.detail)}</div>}
      {question.options?.map(option => <label key={option.label} title={option.description}>
        <input type={question.multiSelect ? 'checkbox' : 'radio'} name={`${pending.eventId}-${question.id}`} checked={answers[question.id]?.selected.includes(option.label) || false}
          onChange={event => setAnswers(current => { const previous = current[question.id] || { selected: [], custom: '' }; return { ...current,
            [question.id]: { ...previous, selected: question.multiSelect ? event.target.checked ? [...previous.selected, option.label] : previous.selected.filter(value => value !== option.label) : [option.label] } }; })} />
        <span>{option.label}</span>
      </label>)}
      <textarea aria-label={`${question.question} · ${tr('补充回答', 'Additional answer')}`} placeholder={tr('或输入回答…', 'Or enter an answer…')}
        value={answers[question.id]?.custom || ''} onChange={event => setAnswers(current => ({ ...current, [question.id]: { selected: current[question.id]?.selected || [], custom: event.target.value } }))} />
    </fieldset>)}
    {error && <p role="alert">{error}</p>}
    <button type="submit" disabled={sending || pending.request.questions.some(question => !answers[question.id]?.selected.length && !answers[question.id]?.custom.trim())}>{tr('提交回答', 'Submit answers')}</button>
  </form>;
}
