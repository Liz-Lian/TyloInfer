import { fireEvent, render, screen } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import { ModelSettingsDialog } from './ModelSettingsDialog';
import { DEFAULT_LOAD_CONTEXT } from './loadContextDefaults';
import type { useModelCatalog } from './useModelCatalog';

vi.mock('../../app/RuntimeProvider', () => ({ useRuntime: () => ({ refreshRuntime: vi.fn() }) }));
vi.mock('../settings/SettingsProvider', () => ({ useSettings: () => ({ tr: (_zh: string, en: string) => en }) }));

it('shows native instance policies as read-only and never mounts unsupported context controls', () => {
  const fetchMock = vi.fn();
  vi.stubGlobal('fetch', fetchMock);
  const catalog = {
    runtime: { source: 'native-instances' }, settingsModel: 'Model', connectionRevision: 1,
    artifacts: [], instances: [{ id: 'instance-a', model: 'Model', state: 'ready', pinned: true, idle_ttl_seconds: null }],
    modelSettingsTriggerRef: { current: null }, setSettingsModel: vi.fn(),
  } as unknown as ReturnType<typeof useModelCatalog>;
  render(<ModelSettingsDialog catalog={catalog} />);
  expect(screen.getByText(/cannot change context or KV/)).toBeInTheDocument();
  fireEvent.click(screen.getByRole('tab', { name: 'Residency and device' }));
  expect(screen.getByText(/cannot update residency/)).toBeInTheDocument();
  expect(screen.getByRole('switch', { name: 'Pin in memory' })).toBeDisabled();
  expect(screen.getByRole('combobox', { name: 'Idle unload' })).toBeDisabled();
  expect(screen.getByRole('button', { name: 'Apply' })).toBeDisabled();
  expect(fetchMock).not.toHaveBeenCalled();
});

it('retains saved unsupported KV preferences and allows explicitly restoring Automatic', () => {
  const fetchMock = vi.fn();
  vi.stubGlobal('fetch', fetchMock);
  const change = vi.fn();
  const catalog = {
    runtime: { source: 'native-instances' }, settingsModel: 'Model', connectionRevision: 1,
    artifacts: [{ id: 'artifact-a', name: 'Model', complete: true, loadable: true }], instances: [],
    loadContext: { ...DEFAULT_LOAD_CONTEXT, automatic: false, ssd_kv: true }, setLoadContext: change,
    loadDeviceIds: [], busy: false, modelSettingsTriggerRef: { current: null }, setSettingsModel: vi.fn(),
    loadPinned: false, loadIdleTtl: null,
  } as unknown as ReturnType<typeof useModelCatalog>;
  render(<ModelSettingsDialog catalog={catalog} />);
  expect(screen.getAllByText(/Saved manual KV settings cannot be applied/)[0]).toBeVisible();
  fireEvent.click(screen.getByRole('checkbox', { name: 'Automatic' }));
  expect(change).toHaveBeenCalledWith(expect.objectContaining({ automatic: true, ssd_kv: true }));
  expect(fetchMock).not.toHaveBeenCalled();
});
