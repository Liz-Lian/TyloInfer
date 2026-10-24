
import type { RuntimeLogEntry, JobResource, JobEventResource, JobKindResource } from '../types';
import { request, apiUrl, errorFromResponse, authorizedHeaders } from '../client';
import { readEventStream } from '../eventStream';

export const jobsApi = {
  getJob(id: string): Promise<JobResource> {
    return request(`/api/v1/jobs/${id}`);
  },

  async jobs(limit = 100): Promise<JobResource[]> {
    return (await request<{ data: JobResource[] }>(`/api/v1/jobs?limit=${limit}`)).data;
  },


  async jobKinds(): Promise<JobKindResource[]> {
    return (await request<{ data: JobKindResource[] }>('/api/v1/jobs/kinds')).data;
  },


  createJob(kind: string, payload: Record<string, unknown>): Promise<JobResource> {
    return request('/api/v1/jobs', {
      method: 'POST',
      body: JSON.stringify({ kind, payload }),
    });
  },


  cancelJob(id: string): Promise<JobResource> {
    return request(`/api/v1/jobs/${id}/cancel`, { method: 'POST' });
  },


  retryJob(id: string): Promise<JobResource> {
    return request(`/api/v1/jobs/${id}/retry`, { method: 'POST' });
  },


  async deleteJob(id: string): Promise<void> {
    const response = await fetch(apiUrl(`/api/v1/jobs/${id}`), {
      method: 'DELETE',
      headers: authorizedHeaders(),
    });
    if (!response.ok) throw await errorFromResponse(response);
  },


  async clearCompletedJobs(): Promise<void> {
    const response = await fetch(apiUrl('/api/v1/jobs/completed'), {
      method: 'DELETE',
      headers: authorizedHeaders(),
    });
    if (!response.ok) throw await errorFromResponse(response);
  },


  async jobEvents(id: string, signal?: AbortSignal): Promise<RuntimeLogEntry[]> {
    const response = await request<{
      data: Array<{
        sequence: number;
        level: RuntimeLogEntry['level'];
        message?: string | null;
        data: Record<string, unknown>;
        created_at: string;
      }>;
    }>(`/api/v1/jobs/${id}/events?limit=1000&tail=true`, { signal });
    return response.data
      .filter((event) => event.message)
      .map((event) => ({
        sequence: event.sequence,
        level: event.level,
        message: event.message || '',
        fields: event.data,
        created_at: event.created_at,
      }));
  },


  async streamJobEvents(
    id: string,
    onEvent: (event: JobEventResource) => void,
    signal: AbortSignal,
    after = 0,
  ): Promise<void> {
    const response = await fetch(apiUrl(`/api/v1/jobs/${id}/events/stream${after > 0 ? `?after=${after}` : ''}`), {
      headers: authorizedHeaders({ Accept: 'text/event-stream' }),
      signal,
    });
    if (!response.ok) throw await errorFromResponse(response);
    await readEventStream(response, onEvent, signal);
  },
};
