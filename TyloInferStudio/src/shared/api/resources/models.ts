
import type {
  ModelArtifact,
  ModelDeletionPlan,
  ModelCacheProfile,
  CheckpointAnalysis,
  ModelDirectoryList,
  HubModelSummary,
  HubModelInfo,
  HubDownloadReuseRequest,
  HubDownloadReuseResponse,
  OfficialModelList,
  ArtifactLineage,
} from '../types';
import { request } from '../client';
import type { KvQuantizationSettings } from './runtime';

export interface ModelLoadContextSettings {
  automatic: boolean;
  ssd_kv: boolean;
  ram_preallocated: boolean;
  vram_preallocated: boolean;
  kv_quantization: KvQuantizationSettings;
}

export interface DownloadDirectoryStatus {
  directory: string | null;
  actual_directory: string;
  default_directory: string;
  disk_total_bytes?: number | null;
  disk_free_bytes?: number | null;
  disk_available_bytes?: number | null;
  download_reserved_bytes?: number;
  reclaimable_prefix_bytes?: number;
  evict_prefix_cache?: boolean;
}

export const modelsApi = {
  downloadReuse(payload: HubDownloadReuseRequest, signal?: AbortSignal): Promise<HubDownloadReuseResponse> {
    return request('/api/v1/hub/download-reuse', { method: 'POST', body: JSON.stringify(payload), signal });
  },
  downloadDirectory(): Promise<DownloadDirectoryStatus> {
    return request('/api/v1/hub/download-directory');
  },
  configureDownloadDirectory(directoryId: string | null): Promise<DownloadDirectoryStatus> {
    return request('/api/v1/hub/download-directory', { method: 'PUT', body: JSON.stringify({ directory_id: directoryId }) });
  },
  configureDownloadReclaim(evictPrefixCache: boolean): Promise<DownloadDirectoryStatus> {
    return request('/api/v1/hub/download-directory', { method: 'PUT', body: JSON.stringify({ evict_prefix_cache: evictPrefixCache }) });
  },
  probeDownloadSources(sources: { provider: HubModelSummary['provider']; repo_id: string; revision?: string | null }[], signal?: AbortSignal): Promise<{
    data: { provider: HubModelSummary['provider']; repo_id: string; bytes_per_second: number; route?: 'direct' | 'proxy' }[];
  }> {
    return request('/api/v1/hub/download-sources/probe', { method: 'POST', body: JSON.stringify({ sources }), signal });
  },

  async modelArtifacts(refresh = false): Promise<ModelArtifact[]> {
    return (
      await request<{ data: ModelArtifact[] }>(`/api/v1/models${refresh ? '?refresh=true' : ''}`)
    ).data;
  },


  modelDirectories(directoryId?: string | null, path?: string | null): Promise<ModelDirectoryList> {
    const query = new URLSearchParams();
    if (directoryId) query.set('directory_id', directoryId);
    else if (path?.trim()) query.set('path', path.trim());
    const suffix = query.size ? `?${query.toString()}` : '';
    return request(`/api/v1/models/directories${suffix}`);
  },

  modelArtifactDirectory(modelId: string): Promise<ModelDirectoryList> {
    return request(`/api/v1/models/${encodeURIComponent(modelId)}/directory`);
  },

  modelDeletionPlan(modelId: string): Promise<ModelDeletionPlan> {
    return request(`/api/v1/models/${encodeURIComponent(modelId)}/deletion`);
  },

  deleteModel(modelId: string, token: string): Promise<{ files: number; total_bytes: number }> {
    return request(`/api/v1/models/${encodeURIComponent(modelId)}/delete`, {
      method: 'POST', body: JSON.stringify({ token }),
    });
  },

  checkpointAnalysis(modelId: string, signal?: AbortSignal): Promise<CheckpointAnalysis> {
    return request(`/api/v1/models/${encodeURIComponent(modelId)}/analysis`, { signal });
  },

  modelCacheProfile(modelId: string, signal?: AbortSignal, includeRecurrent = false): Promise<ModelCacheProfile | null> {
    return request(`/api/v1/models/${encodeURIComponent(modelId)}/cache-profile${includeRecurrent ? '?include_recurrent=true' : ''}`, { signal });
  },

  openModelDirectory(directoryId: string): Promise<{ opened: boolean }> {
    return request('/api/v1/models/directories/open', {
      method: 'POST', body: JSON.stringify({ directory_id: directoryId }),
    });
  },


  async registerModelDirectory(directoryId: string): Promise<ModelArtifact[]> {
    return (
      await request<{ data: ModelArtifact[] }>('/api/v1/models/directories/register', {
        method: 'POST',
        body: JSON.stringify({ directory_id: directoryId }),
      })
    ).data;
  },


  loadModel(
    model: string,
    contextSize: number | null = null,
    prefillChunkSize: number | 'auto' = 'auto',
    policy: { pin?: boolean; idle_ttl_seconds?: number | null; device_ids?: string[]; context_management?: ModelLoadContextSettings } = {},
  ): Promise<{ operation_id: string; status: 'accepted' }> {
    return request('/api/v1/models/load', {
      method: 'POST',
      body: JSON.stringify({
        model,
        context_size: contextSize,
        prefill_chunk_size: prefillChunkSize,
        pin: policy.pin ?? false,
        ...(policy.context_management ? { context_management: {
          automatic: policy.context_management.automatic, ssd_kv: policy.context_management.ssd_kv,
          ram_preallocated: policy.context_management.ram_preallocated, vram_preallocated: policy.context_management.vram_preallocated,
          kv_quantization: policy.context_management.kv_quantization,
        } } : {}),
        ...(policy.device_ids?.length ? { device_ids: policy.device_ids } : {}),
        idle_ttl_seconds: policy.pin ? null : (policy.idle_ttl_seconds ?? null),
      }),
    });
  },


  unloadModel(
    instanceId: string,
    force = false,
  ): Promise<{ operation_id: string; status: 'accepted' }> {
    return request('/api/v1/models/unload', {
      method: 'POST',
      body: JSON.stringify({ instance_id: instanceId, force }),
    });
  },


  async searchHubModels(
    provider: HubModelSummary['provider'],
    query: string,
  ): Promise<HubModelSummary[]> {
    const params = new URLSearchParams({ provider, query, limit: '20' });
    return (await request<{ data: HubModelSummary[] }>(`/api/v1/hub/models?${params}`)).data;
  },


  officialHubModels(refresh = false, signal?: AbortSignal): Promise<OfficialModelList> {
    return request(`/api/v1/hub/official${refresh ? '?refresh=true' : ''}`, { signal });
  },


  resolveHubModel(
    reference: string,
    fallbackProvider: HubModelSummary['provider'],
  ): Promise<HubModelInfo> {
    return request('/api/v1/hub/resolve', {
      method: 'POST',
      body: JSON.stringify({
        reference,
        fallback_provider: fallbackProvider,
      }),
    });
  },


  hubModelInfo(
    provider: HubModelSummary['provider'],
    repoId: string,
    revision?: string,
  ): Promise<HubModelInfo> {
    const [owner, name] = repoId.split('/', 2);
    const suffix = revision ? `?revision=${encodeURIComponent(revision)}` : '';
    return request(
      `/api/v1/hub/models/${provider}/${encodeURIComponent(owner)}/${encodeURIComponent(name)}${suffix}`,
    );
  },


  removeWorkspaceArtifact(uri: string): Promise<Record<string, unknown>> {
    return request('/api/v1/artifacts/remove', {
      method: 'POST',
      body: JSON.stringify({ artifact_uri: uri }),
    });
  },


  async artifactLineage(limit = 200): Promise<ArtifactLineage[]> {
    return (await request<{ data: ArtifactLineage[] }>(`/api/v1/artifacts/lineage?limit=${limit}`))
      .data;
  },
};
