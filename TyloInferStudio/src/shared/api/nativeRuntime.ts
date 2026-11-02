import type { ModelCapabilities, RuntimeCapabilities, RuntimeInstance, RuntimeModel, RuntimeStatus } from './types';

/** The native /runtime/instances wire contract; metrics remain backend-owned data. */
export interface NativeRuntimeInstance {
  instance_id: string;
  model: string;
  artifact_id: string;
  backend: string;
  state: 'loaded' | 'unloading' | 'failed';
  context_size: number;
  max_requests: number;
  active_requests: number;
  capabilities: ModelCapabilities;
  policy: { pin: boolean; idle_ttl_seconds: number | null };
  metrics_age_ms: number;
  metrics: Record<string, unknown>;
  session_metrics: Record<string, unknown>;
}

/** Adapt native instance identity and state without inventing device or memory telemetry. */
export function adaptNativeInstance(instance: NativeRuntimeInstance): RuntimeInstance {
  return {
    id: instance.instance_id,
    model: instance.model,
    artifact_id: instance.artifact_id,
    backend: instance.backend,
    state: instance.state === 'loaded' ? instance.active_requests > 0 ? 'busy' : 'ready' : instance.state,
    devices: [],
    active_requests: instance.active_requests,
    context_size: instance.context_size,
    max_requests: instance.max_requests,
    capabilities: instance.capabilities,
    pinned: instance.policy.pin,
    idle_ttl_seconds: instance.policy.idle_ttl_seconds,
    metrics_age_ms: instance.metrics_age_ms,
    metrics: instance.metrics,
    session_metrics: instance.session_metrics,
  };
}

/** Project Studio's basic runtime view from a single authoritative instance snapshot. */
export function projectNativeRuntime(instances: RuntimeInstance[], selectedModel: string) {
  const instance = instances.find(item => item.model === selectedModel && item.state !== 'failed')
    ?? instances.find(item => item.state === 'ready' || item.state === 'busy')
    ?? instances[0];
  const runtime: RuntimeStatus = {
    source: 'native-instances',
    runtime_state: instance?.state ?? 'idle',
    ...(instance ? {
      instance_id: instance.id,
      model: instance.model,
      max_context: instance.context_size ?? undefined,
      active_requests: instance.active_requests,
      model_capabilities: instance.capabilities,
      model_type: instance.capabilities?.architecture_family,
    } : {}),
  };
  const models: RuntimeModel[] = Array.from(new Set(instances
    .filter(item => item.state === 'ready' || item.state === 'busy').map(item => item.model)))
    .map(id => ({ id }));
  const capabilities: RuntimeCapabilities | null = instance?.capabilities ? {
    model: instance.model,
    model_type: instance.capabilities.architecture_family,
    model_capabilities: instance.capabilities,
    vision_available: false,
    mtp_available: false,
    duplex_available: false,
  } : null;
  return { runtime, models, capabilities };
}
