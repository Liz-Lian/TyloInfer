/** 汇总运行时模型、实例与加载任务，保持模型选择状态一致。 */
import { JobResource, RuntimeInstance, RuntimeModel } from '../../shared/api/types';

/** Return whether an adapted runtime state can accept inference requests. */
export function isRuntimeReady(state: string | null | undefined): boolean {
  return state === "ready" || state === "busy";
}

/** Merge advertised names and ready instances into inference choices. */
export function runtimeModelNames(
  advertised: RuntimeModel[],
  instances: RuntimeInstance[],
): string[] {
  return Array.from(new Set([
    ...advertised.map((item) => item.id),
    ...instances
      .filter((item) => isRuntimeReady(item.state))
      .map((item) => item.model),
  ].filter(Boolean)));
}

/** Resolve a native loading job to its display name while retaining its artifact ID. */
export function modelLoadJobName(job: JobResource, artifactNames: Record<string, string> = {}): string {
  if (typeof job.payload.artifact_id === 'string') return artifactNames[job.payload.artifact_id] ?? job.payload.artifact_id;
  return typeof job.payload.model === 'string' ? job.payload.model : '';
}

/** Merge ready instances and active loads into display-name selection choices. */
export function runtimeSelectionNames(
  advertised: RuntimeModel[],
  instances: RuntimeInstance[],
  jobs: JobResource[] = [],
  artifactNames: Record<string, string> = {},
): string[] {
  return Array.from(new Set([
    ...runtimeModelNames(advertised, instances),
    ...instances
      .filter((item) => item.state !== "failed" && item.state !== "unloading")
      .map((item) => item.model),
    ...jobs
      .filter((item) => item.kind === "model.load"
        && ["queued", "running", "cancelling"].includes(item.status))
      .map((item) => modelLoadJobName(item, artifactNames))
      .filter((item): item is string => typeof item === "string" && Boolean(item)),
  ]));
}
