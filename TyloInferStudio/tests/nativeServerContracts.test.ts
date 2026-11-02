import { afterEach, expect, it, vi } from 'vitest';
import { setApiBaseUrl, setApiToken } from '../src/shared/api/client';
import { modelsApi } from '../src/shared/api/resources/models';
import { runtimeApi } from '../src/shared/api/resources/runtime';
import { jobsApi } from '../src/shared/api/resources/jobs';
import { projectNativeRuntime, type NativeRuntimeInstance } from '../src/shared/api/nativeRuntime';
import { modelLoadJobName, runtimeSelectionNames } from '../src/features/runtime/modelSelection';
import { useJobStore } from '../src/stores/jobStore';
import type { JobResource } from '../src/shared/api/types';

const profile = {
  architecture_family: 'qwen3', source: 'architecture-registry:qwen3',
  features: { text: true, image_input: true, video_input: false, audio_input: true,
    audio_output: true, full_duplex: true, mtp: true },
};
const instance: NativeRuntimeInstance = {
  instance_id: 'instance-a', model: 'Local Model', artifact_id: 'artifact-a', backend: 'cuda',
  state: 'loaded', context_size: 8192, max_requests: 4, active_requests: 2,
  capabilities: profile, policy: { pin: true, idle_ttl_seconds: null },
  metrics_age_ms: 12, metrics: { cuda_allocated_bytes: 64 }, session_metrics: { decode_tps: 3 },
};
const json = (value: unknown, status = 200) => new Response(JSON.stringify(value), { status });

afterEach(() => {
  useJobStore.getState().clearJobStreams();
  useJobStore.getState().setJobs([]);
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  setApiBaseUrl('');
  setApiToken('');
});

it('adapts real native instance fields and leaves missing telemetry unknown', async () => {
  const fetchMock = vi.fn().mockResolvedValue(json({ data: [instance] }));
  vi.stubGlobal('fetch', fetchMock);
  const instances = await runtimeApi.runtimeInstances();
  expect(instances[0]).toMatchObject({ id: 'instance-a', artifact_id: 'artifact-a', backend: 'cuda',
    state: 'busy', active_requests: 2, context_size: 8192, pinned: true, metrics: instance.metrics });
  expect(instances[0].devices).toEqual([]);
  expect(instances[0].active_sessions).toBeUndefined();
  expect(instances[0].queued_requests).toBeUndefined();
  expect(instances[0].memory).toBeUndefined();
  const view = projectNativeRuntime(instances, 'Local Model');
  expect(view.models).toEqual([{ id: 'Local Model' }]);
  expect(view.runtime).toMatchObject({ source: 'native-instances', runtime_state: 'busy', model: 'Local Model', max_context: 8192 });
  expect(view.runtime.gpu_devices).toBeUndefined();
  expect(view.runtime.total_requests).toBeUndefined();
  expect(view.capabilities?.model_capabilities).toEqual(profile);
  expect(view.capabilities?.vision_available).toBe(false);
  expect(view.capabilities?.duplex_available).toBe(false);
  expect(fetchMock.mock.calls.map(([url]) => url)).toEqual(['/api/v1/runtime/instances']);
  expect(projectNativeRuntime([], '').runtime).toEqual({ source: 'native-instances', runtime_state: 'idle' });
});

it('registers a directory and submits ID-based native loads with omitted automatic fields', async () => {
  const fetchMock = vi.fn()
    .mockResolvedValueOnce(json({ data: [{ id: 'artifact-a', name: 'Local Model' }] }))
    .mockImplementation(async () => json({ operation_id: 'job-a', status: 'accepted' }, 202));
  vi.stubGlobal('fetch', fetchMock);
  expect(await modelsApi.registerModelDirectory('directory-a')).toEqual([{ id: 'artifact-a', name: 'Local Model' }]);
  expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({ directory_id: 'directory-a' });
  await modelsApi.loadModel('artifact-a');
  expect(JSON.parse(fetchMock.mock.calls[1][1].body)).toEqual({ artifact_id: 'artifact-a', requirements: {}, policy: { pin: false } });
  await modelsApi.loadModel('artifact-a', 8192, 'auto', { idle_ttl_seconds: 300 });
  expect(JSON.parse(fetchMock.mock.calls[2][1].body)).toEqual({ artifact_id: 'artifact-a', requirements: { context_size: 8192 }, policy: { pin: false, idle_ttl_seconds: 300 } });
  await modelsApi.loadModel('artifact-a', null, 'auto', { pin: true, idle_ttl_seconds: 300 });
  expect(JSON.parse(fetchMock.mock.calls[3][1].body)).toEqual({ artifact_id: 'artifact-a', requirements: {}, policy: { pin: true } });
  await modelsApi.unloadModel('instance-a');
  expect(JSON.parse(fetchMock.mock.calls[4][1].body)).toEqual({ instance_id: 'instance-a', force: false });
});

it('omits the default automatic KV configuration and rejects explicit prefill controls', async () => {
  const fetchMock = vi.fn().mockImplementation(async () => json({ operation_id: 'job-a', status: 'accepted' }, 202));
  vi.stubGlobal('fetch', fetchMock);
  await modelsApi.loadModel('artifact-a', null, 'auto', { context_management: { automatic: true,
    ssd_kv: false, ram_preallocated: true, vram_preallocated: true,
    kv_quantization: { enabled: false, bits: 4, algorithm: 'turboquant' } } });
  expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({ artifact_id: 'artifact-a', requirements: {}, policy: { pin: false } });
  await expect(modelsApi.loadModel('artifact-a', null, 256)).rejects.toThrow('not supported');
  expect(fetchMock).toHaveBeenCalledOnce();
});

it.each([
  { device_ids: ['cuda:0'] },
  { gguf_companions: ['/models/table.gguf'] },
  { context_management: { automatic: false, ssd_kv: false, ram_preallocated: true, vram_preallocated: true,
    kv_quantization: { enabled: false, bits: 4 as const, algorithm: 'turboquant' as const } } },
])('rejects explicit unsupported load settings before sending a request: %j', async policy => {
  const fetchMock = vi.fn();
  vi.stubGlobal('fetch', fetchMock);
  await expect(modelsApi.loadModel('artifact-a', null, 'auto', policy)).rejects.toThrow('not supported');
  expect(fetchMock).not.toHaveBeenCalled();
});

it('resolves native load jobs by artifact ID without rewriting their payload', () => {
  const job: JobResource = { id: 'job-a', kind: 'model.load', status: 'running', payload: { artifact_id: 'artifact-a' },
    progress: 0, cancel_requested: false, created_at: '', updated_at: '' };
  expect(modelLoadJobName(job, { 'artifact-a': 'Local Model' })).toBe('Local Model');
  expect(runtimeSelectionNames([], [], [job], { 'artifact-a': 'Local Model' })).toEqual(['Local Model']);
  expect(job.payload).toEqual({ artifact_id: 'artifact-a' });
});

it('consumes native job SSE progress data and terminal failure without waiting for a later poll', async () => {
  const job: JobResource = { id: 'job-a', kind: 'model.load', status: 'running', progress: 0, cancel_requested: false,
    payload: { artifact_id: 'artifact-a' }, created_at: '2026-11-02T00:00:00Z', updated_at: '2026-11-02T00:00:00Z' };
  useJobStore.getState().setJobs([job]);
  const error = { code: 'model_load_failed', message: 'Insufficient memory', retryable: false, details: {} };
  const events = [
    { job_id: job.id, sequence: 1, type: 'progress', level: 'info', data: { phase: 'loading', progress: 0 }, created_at: job.updated_at },
    { job_id: job.id, sequence: 2, type: 'state', level: 'info', data: { status: 'failed', error, result: null }, created_at: job.updated_at },
  ];
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(events.map(event => `id: ${event.sequence}\ndata: ${JSON.stringify(event)}\n\n`).join(''), {
    headers: { 'Content-Type': 'text/event-stream' },
  })));
  const terminal = vi.fn();
  useJobStore.getState().streamJobEvents(job.id, { onTerminal: terminal });
  await vi.waitFor(() => expect(terminal).toHaveBeenCalledOnce());
  expect(useJobStore.getState().jobs[0]).toMatchObject({ status: 'failed', error, result: null, progress_data: { phase: 'loading' } });
  expect(useJobStore.getState().activeJobIds).toEqual([]);
});

it('uses the real job cancellation endpoint and preserves deferred cancellation state', async () => {
  const pending = { id: 'job-a', status: 'cancelling', cancel_requested: true };
  const fetchMock = vi.fn().mockResolvedValue(json(pending));
  vi.stubGlobal('fetch', fetchMock);
  await expect(jobsApi.cancelJob('job-a')).resolves.toEqual(pending);
  expect(fetchMock.mock.calls[0][0]).toBe('/api/v1/jobs/job-a/cancel');
  expect(fetchMock.mock.calls[0][1].method).toBe('POST');
});

it('propagates a native job stream error envelope instead of treating it as a job event', async () => {
  const error = { code: 'storage_error', message: 'Cannot read job events', retryable: false, details: {} };
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(`event: error\ndata: ${JSON.stringify({ error })}\n\n`, {
    headers: { 'Content-Type': 'text/event-stream' },
  })));
  await expect(jobsApi.streamJobEvents('job-a', vi.fn(), new AbortController().signal)).rejects.toThrow('Cannot read job events');
});
