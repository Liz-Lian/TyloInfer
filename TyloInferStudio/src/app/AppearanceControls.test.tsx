import { fireEvent, render, screen } from '@testing-library/react';
import { beforeEach, expect, it } from 'vitest';
import { SettingsProvider } from '../features/settings/SettingsProvider';
import { DEFAULT_SETTINGS, SETTINGS_KEY } from '../features/settings/configuration';
import { AppearanceControls } from './AppearanceControls';

beforeEach(() => localStorage.clear());
it('applies theme and language immediately without changing inference preferences', () => {
  localStorage.setItem(SETTINGS_KEY, JSON.stringify({ ...DEFAULT_SETTINGS, language: 'en', temperature: .37 }));
  render(<SettingsProvider><AppearanceControls /></SettingsProvider>);
  const controls = screen.getByRole('group', { name: 'Appearance and language' });
  expect(controls.querySelectorAll(':scope > [role="group"]')).toHaveLength(2);
  expect(controls.querySelectorAll('.appearance-controls-divider')).toHaveLength(1);
  expect(controls.querySelectorAll('button')).toHaveLength(6);
  expect(screen.getByRole('button', { name: 'Automatic appearance' })).toHaveAttribute('aria-pressed', 'true');
  fireEvent.click(screen.getByRole('button', { name: 'Dark' }));
  expect(document.documentElement.dataset.theme).toBe('dark');
  fireEvent.click(screen.getByRole('button', { name: '简体中文' }));
  expect(screen.getByRole('button', { name: '深色' })).toHaveAttribute('aria-pressed', 'true');
  fireEvent.click(screen.getByRole('button', { name: '自动明暗' }));
  fireEvent.click(screen.getByRole('button', { name: '自动语言' }));
  const saved = JSON.parse(localStorage.getItem(SETTINGS_KEY)!);
  expect(saved).toMatchObject({ theme: 'system', language: 'system', temperature: .37 });
});
