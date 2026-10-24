import { request } from '../client';

export interface ReferenceFiles {
  path: string;
  parent: string | null;
  data: { name: string; path: string; directory: boolean; byte_size: number | null }[];
}

export interface ReferenceSource {
  path: string;
  architecture: string;
  tensors: number;
  parameters: number;
  format: 'hf';
  full_precision: boolean;
  source_precisions?: string[];
  reference_supported?: boolean;
}

export interface SourceLoadingPlan {
  backend: string;
  resident_required_bytes: number;
  available_bytes: number;
  resident_allowed: boolean;
  weights_bytes: number;
  workspace_bytes: number;
}

export const referenceApi = {
  workspace: () => request<{ import_directory: string }>('/api/v1/evaluations/reference/workspace'),
  files: (path: string, signal?: AbortSignal) => request<ReferenceFiles>(`/api/v1/evaluations/reference/files?path=${encodeURIComponent(path)}`, { signal }),
  source: (path: string) => request<ReferenceSource>('/api/v1/evaluations/reference/source', {
    method: 'POST', body: JSON.stringify({ path }),
  }),
  loadingPlan: (path: string, purpose: 'wt2', context_size: number) => request<SourceLoadingPlan>('/api/v1/evaluations/reference/loading-plan', {
    method: 'POST', body: JSON.stringify({ path, purpose, context_size }),
  }),
};
