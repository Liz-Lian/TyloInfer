import type { JobResource } from '../../shared/api/types';

export const TERMINAL_JOB_STATUSES = new Set<JobResource["status"]>([
  "succeeded",
  "failed",
  "cancelled",
  "interrupted",
]);


export function isTerminalJob(job: JobResource): boolean {
  return TERMINAL_JOB_STATUSES.has(job.status);
}
