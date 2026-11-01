import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import { request } from '../../shared/api/client';
import { AgentQuestionCard } from './AgentQuestionCard';

vi.mock('../../shared/api/client', () => ({ request: vi.fn().mockResolvedValue({ accepted: true }) }));
vi.mock('../settings/SettingsProvider', () => ({ useSettings: () => ({ tr: (_zh: string, en: string) => en }) }));

it('requires an answer and returns the selected labels to the owning DSH request', async () => {
  const refresh = vi.fn().mockResolvedValue(undefined);
  render(<AgentQuestionCard pending={{ eventId: 'question-1', request: { questions: [
    { id: 'choice', question: 'Continue?', options: [{ label: 'Yes' }, { label: 'No' }] },
  ] } }} onAnswered={refresh} />);
  const submit = screen.getByRole('button', { name: 'Submit answers' });
  expect(submit).toBeDisabled();
  fireEvent.click(screen.getByRole('radio', { name: 'Yes' }));
  fireEvent.click(submit);
  await waitFor(() => expect(refresh).toHaveBeenCalledOnce());
  expect(request).toHaveBeenCalledWith('/api/v1/agent/answers', { method: 'POST', body: JSON.stringify({
    event_id: 'question-1', answers: [{ id: 'choice', selected: ['Yes'], custom: '' }],
  }) });
});
