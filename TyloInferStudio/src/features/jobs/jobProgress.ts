import type { JobResource } from '../../shared/api/types';

/** Identify native loads, which report phases rather than measured completion percentages. */
export function isNativePhaseLoad(job: JobResource): boolean {
  return job.kind === 'model.load' && typeof job.payload.artifact_id === 'string';
}

/** Describe native load progress and the deferred cancellation boundary. */
export function nativeLoadPhase(job: JobResource, tr: (zh: string, en: string) => string): string {
  if (job.status === 'cancelling') return tr('等待原生加载返回后取消', 'Cancelling after native loading returns');
  if (job.status === 'queued') return tr('排队中', 'Queued');
  const phases: Record<string, string> = {
    resolving: tr('解析模型', 'Resolving model'),
    loading: tr('加载权重', 'Loading weights'),
    initializing: tr('初始化', 'Initializing'),
  };
  return phases[job.progress_data?.phase ?? ''] ?? tr('准备中', 'Preparing');
}
