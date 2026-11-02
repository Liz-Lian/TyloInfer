import { renderHook } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import type { RuntimeStatus, RuntimeCapabilities } from '../../../shared/api/types';
import { DEFAULT_SETTINGS } from '../../settings/configuration';
import { useChatInference } from './useChatInference';

const context = vi.hoisted(() => ({
  runtime: {} as RuntimeStatus,
  instances: [{ model: 'model', state: 'ready', mtp_supported: true, mtp_available: true }],
  selectedModel: 'model', models: [], capabilities: null as RuntimeCapabilities | null, realtime: null,
  artifactNames: { 'artifact-a': 'model' },
}));
const settings = { ...DEFAULT_SETTINGS };
vi.mock('../../../app/RuntimeProvider', () => ({ useRuntime: () => context }));
vi.mock('../../settings/SettingsProvider', () => ({ useSettings: () => ({ settings, updateSettings: vi.fn() }) }));

beforeEach(() => {
  Object.assign(settings, DEFAULT_SETTINGS);
  context.runtime = { mtp_service_enabled: true, sampling_defaults: { enable_mtp: true } };
  context.capabilities = null;
});

it('keeps model declarations separate from native upload and voice service availability', () => {
  context.runtime = { source: 'native-instances' };
  context.capabilities = { model: 'model', model_type: 'test', vision_available: false, mtp_available: false,
    duplex_available: false, model_capabilities: { architecture_family: 'test', source: 'registry', features: {
      text: true, image_input: true, video_input: true, audio_input: true, audio_output: true, full_duplex: true, mtp: false,
    } } };
  settings.inheritModelDefaults = false;
  settings.reasoningEffort = 'high';
  const { result } = renderHook(() => useChatInference('text'));
  expect(result.current.nativeTextOnly).toBe(true);
  expect(result.current.visionSupported).toBe(true);
  expect(result.current.visionAvailable).toBe(false);
  expect(result.current.attachmentAccept).toBe('');
  expect(result.current.realtimeAvailable).toBe(false);
  expect(result.current.sampling.reasoning_effort).toBe('high');
});

it('disables chat MTP when the service forbids it without overwriting the chat preference', () => {
  context.runtime.mtp_service_enabled = false;
  const view = renderHook(() => useChatInference('text'));
  expect(view.result.current.mtpServiceEnabled).toBe(false);
  expect(view.result.current.mtpAvailable).toBe(false);
  expect(view.result.current.sampling.enable_mtp).toBe(false);
  expect(view.result.current.effectiveSettings.enableMtp).toBe(true);
  context.runtime.mtp_service_enabled = true;
  view.rerender();
  expect(view.result.current.mtpAvailable).toBe(true);
  expect(view.result.current.sampling.enable_mtp).toBe(true);
});

it('still honors a disabled per-chat preference when the service permits MTP', () => {
  settings.inheritModelDefaults = false;
  settings.enableMtp = false;
  const view = renderHook(() => useChatInference('text'));
  expect(view.result.current.mtpAvailable).toBe(true);
  expect(view.result.current.sampling.enable_mtp).toBe(false);
});
