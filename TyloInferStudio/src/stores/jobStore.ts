



import { create } from 'zustand';
import { jobsApi } from '../shared/api/resources/jobs';
import type { JobResource, JobEventResource } from '../shared/api/types';

const ACTIVE_STATUSES: ReadonlySet<JobResource['status']> = new Set([
  'queued',
  'running',
  'cancelling',
]);

const TERMINAL_STATUSES: ReadonlySet<JobResource['status']> = new Set([
  'succeeded',
  'failed',
  'cancelled',
  'interrupted',
]);


function extractActiveJobIds(jobs: JobResource[]): string[] {
  return jobs.filter((job) => ACTIVE_STATUSES.has(job.status)).map((job) => job.id);
}


function haveSameElements(a: string[], b: string[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

export interface WatchActiveJobsOptions {

  onTerminal?: (jobId: string, status: JobResource['status']) => void;

  onError?: (jobId: string, cause: unknown) => void;

  onEvent?: (jobId: string) => void;

  signal?: AbortSignal;
}

export interface StreamJobEventsOptions {

  onTerminal?: (job: JobResource) => void;

  onError?: (cause: unknown) => void;

  onEvent?: () => void;

  signal?: AbortSignal;
}

export interface JobState {

  jobs: JobResource[];

  activeJobIds: string[];





  setJobs: (jobs: JobResource[]) => void;





  addJob: (job: JobResource) => void;






  updateJob: (id: string, patch: Partial<JobResource>) => void;







  streamJobEvents: (id: string, options?: StreamJobEventsOptions) => () => void;






  watchActiveJobs: (options?: WatchActiveJobsOptions) => () => void;



  clearJobStreams: () => void;
}


const activeStreams = new Map<string, AbortController>();
const eventSequences = new Map<string, number>();


function stopOwnedStream(id: string, controller: AbortController): void {
  if (activeStreams.get(id) !== controller) return;
  activeStreams.delete(id);
  controller.abort();
}


export const useJobStore = create<JobState>()((set, get) => ({
  jobs: [],
  activeJobIds: [],

  setJobs: (jobs) => {
    if (!jobs.length) eventSequences.clear();
    const current = new Map(get().jobs.map((job) => [job.id, job]));
    const nextJobs = jobs.map((job) => {
      const previous = current.get(job.id);
      return { ...(previous && Date.parse(previous.updated_at) > Date.parse(job.updated_at) ? previous : job) };
    });
    const activeJobIds = extractActiveJobIds(nextJobs);
    set({ jobs: nextJobs, activeJobIds });
  },

  addJob: (job) => {
    set((state) => {
      const jobs = [{ ...job }, ...state.jobs.filter((item) => item.id !== job.id)];
      const activeJobIds = extractActiveJobIds(jobs);
      return { jobs, activeJobIds };
    });
  },

  updateJob: (id, patch) => {
    set((state) => {
      let changed = false;
      const jobs = state.jobs.map((job) => {
        if (job.id !== id) return job;
        if (patch.updated_at && Date.parse(job.updated_at) > Date.parse(patch.updated_at)) return job;
        changed = true;
        return { ...job, ...patch };
      });
      if (!changed) return state;
      const nextActive = extractActiveJobIds(jobs);
      return {
        jobs,
        activeJobIds: haveSameElements(state.activeJobIds, nextActive)
          ? state.activeJobIds
          : nextActive,
      };
    });
  },

  streamJobEvents: (id, options) => {
    if (activeStreams.has(id) || options?.signal?.aborted) return () => {};
    const controller = new AbortController();
    activeStreams.set(id, controller);
    const stop = () => stopOwnedStream(id, controller);

    if (options?.signal) {
      options.signal.addEventListener('abort', stop, { once: true, signal: controller.signal });
      if (options.signal.aborted) {
        stop();
        return stop;
      }
    }

    void jobsApi
      .streamJobEvents(
        id,
        (event: JobEventResource) => {
          if (controller.signal.aborted || activeStreams.get(id) !== controller) return;
          if (event.sequence <= (eventSequences.get(id) ?? 0)) return;
          eventSequences.set(id, event.sequence);
          options?.onEvent?.();
          if (controller.signal.aborted || activeStreams.get(id) !== controller) return;
          const status =
            event.type === 'state' && typeof event.data.status === 'string'
              ? (event.data.status as JobResource['status'])
              : null;
          get().updateJob(id, {
            ...(status ? { status } : {}),
            ...(typeof event.progress === 'number' ? { progress: event.progress } : {}),
            ...(event.type === 'progress' ? { progress_data: event.data } : {}),
            updated_at: event.created_at,
          });
          if (status && TERMINAL_STATUSES.has(status)) {
            stop();
            eventSequences.delete(id);
            const finishedJob = get().jobs.find((j) => j.id === id);
            if (finishedJob) options?.onTerminal?.(finishedJob);
          }
        },
        controller.signal,
        eventSequences.get(id) ?? 0,
      )
      .then(() => {
        if (!controller.signal.aborted && activeStreams.get(id) === controller) {
          stop();
          options?.onError?.(new Error('Task event stream closed unexpectedly'));
        }
      })
      .catch((cause) => {
        if (!controller.signal.aborted && activeStreams.get(id) === controller) {
          stop();
          options?.onError?.(cause);
        }
      });

    return stop;
  },

  watchActiveJobs: (options) => {
    const currentActive = get().activeJobIds;
    for (const id of eventSequences.keys()) if (!currentActive.includes(id)) eventSequences.delete(id);
    const ownedStreams = new Map<string, AbortController>();

    for (const id of currentActive) {
      if (activeStreams.has(id)) continue;
      get().streamJobEvents(id, {
        signal: options?.signal,
        onTerminal: (job) => options?.onTerminal?.(job.id, job.status),
        onEvent: () => options?.onEvent?.(id),
        onError: (cause) => options?.onError?.(id, cause),
      });
      const controller = activeStreams.get(id);
      if (controller) ownedStreams.set(id, controller);
    }


    for (const [id, controller] of activeStreams.entries()) {
      if (!currentActive.includes(id)) {
        stopOwnedStream(id, controller);
      }
    }

    return () => {
      for (const [id, controller] of ownedStreams) stopOwnedStream(id, controller);
      ownedStreams.clear();
    };
  },

  clearJobStreams: () => {
    for (const [id, controller] of activeStreams) stopOwnedStream(id, controller);
  },
}));
