import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import { ApiError } from '../../shared/api/client';
import { modelsApi } from '../../shared/api/resources/models';
import { WorkspaceDirectoryDialog } from './WorkspaceDirectoryDialog';

vi.mock('../../shared/api/resources/models', () => ({ modelsApi: { modelDirectories: vi.fn() } }));
vi.mock('../settings/SettingsProvider', () => ({ useSettings: () => ({ tr: (_zh: string, en: string) => en }) }));
const root = { current_id: 'root', current_name: '/', current_path: '/', parent_id: null, model_file_count: 0,
  data: [{ id: 'projects', name: 'Projects', model_file_count: 0 }] };
beforeEach(() => vi.mocked(modelsApi.modelDirectories).mockReset());

it('browses into folders and selects their canonical path without requiring typed paths', async () => {
  vi.mocked(modelsApi.modelDirectories).mockImplementation(async id => id === 'projects'
    ? { ...root, current_id: 'projects', current_path: '/Projects', parent_id: 'root', data: [] } : root);
  const select = vi.fn().mockResolvedValue(undefined);
  render(<WorkspaceDirectoryDialog initialPath="/" onSelect={select} onClose={vi.fn()} trigger={{ current: null }} />);
  await screen.findByRole('button', { name: 'Projects' });
  fireEvent.click(screen.getByRole('button', { name: 'Projects' }));
  await waitFor(() => expect(screen.getByRole('textbox', { name: 'Current directory' })).toHaveValue('/Projects'));
  expect(screen.getByRole('button', { name: 'Up' })).toBeEnabled();
  fireEvent.click(screen.getByRole('button', { name: 'Select folder' }));
  await waitFor(() => expect(select).toHaveBeenCalledWith('/Projects'));
});

it('rejects an unavailable jump, restores the last folder and allows retry after an add failure', async () => {
  vi.mocked(modelsApi.modelDirectories).mockImplementation(async (_id, path) => {
    if (path === '/missing') throw new ApiError(404, { error: { code: 'model_directory_not_found', message: 'model directory is unavailable', retryable: false, details: {} } });
    return root;
  });
  const select = vi.fn().mockRejectedValueOnce(new Error('Agent unavailable')).mockResolvedValue(undefined);
  render(<WorkspaceDirectoryDialog initialPath="/" onSelect={select} onClose={vi.fn()} trigger={{ current: null }} />);
  const input = await screen.findByRole('textbox', { name: 'Current directory' });
  await screen.findByRole('button', { name: 'Projects' });
  fireEvent.change(input, { target: { value: '/missing' } });
  expect(screen.getByRole('button', { name: 'Select folder' })).toBeDisabled();
  fireEvent.click(screen.getByRole('button', { name: 'Go' }));
  expect(await screen.findByRole('alert')).toHaveTextContent('Cannot open this folder. Check its path and permissions.');
  expect(input).toHaveValue('/');
  expect(screen.getByRole('button', { name: 'Select folder' })).toBeDisabled();
  fireEvent.click(screen.getByRole('button', { name: 'Go' }));
  await waitFor(() => expect(screen.getByRole('button', { name: 'Select folder' })).toBeEnabled());
  fireEvent.click(screen.getByRole('button', { name: 'Select folder' }));
  expect(await screen.findByRole('alert')).toHaveTextContent('Agent unavailable');
  fireEvent.click(screen.getByRole('button', { name: 'Select folder' }));
  await waitFor(() => expect(select).toHaveBeenCalledTimes(2));
});
