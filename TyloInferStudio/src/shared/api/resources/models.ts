/** 封装 models 领域资源请求，不保存组件状态。 */
import type {
  ModelArtifact,
  ModelDeletionPlan,
  ModelCacheProfile,
  CheckpointAnalysis,
  ModelDirectoryList,
  HubModelSummary,
  HubModelSearchPage,
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
  browseHubModels(provider: HubModelSummary['provider'], query: string, options: {
    formats: ('tq' | 'gguf' | 'mlx')[]; sort: 'relevance' | 'downloads' | 'likes' | 'updated';
    direction: 'asc' | 'desc'; cursor?: string | null; signal?: AbortSignal;
  }): Promise<HubModelSearchPage> {
    const params = new URLSearchParams({ provider, query, formats: options.formats.join(','),
      sort: options.sort, direction: options.direction, limit: '40', browse: 'true' });
    if (options.cursor) params.set('cursor', options.cursor);
    return request(`/api/v1/hub/models?${params}`, { signal: options.signal });
  },
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
  /** 列出模型资产；refresh 为真时要求服务重新扫描。 */
  async modelArtifacts(refresh = false, includeComponents = false): Promise<ModelArtifact[]> {
    return (
      await request<{ data: ModelArtifact[] }>(`/api/v1/models${refresh || includeComponents ? `?${new URLSearchParams({ ...(refresh ? { refresh: 'true' } : {}), ...(includeComponents ? { include_components: 'true' } : {}) })}` : ''}`)
    ).data;
  },

  /** 浏览服务端模型目录，优先使用目录标识，其次使用完整路径。 */
  modelDirectories(directoryId?: string | null, path?: string | null, signal?: AbortSignal): Promise<ModelDirectoryList> {
    const query = new URLSearchParams();
    if (directoryId) query.set('directory_id', directoryId);
    else if (path?.trim()) query.set('path', path.trim());
    const suffix = query.size ? `?${query.toString()}` : '';
    return request(`/api/v1/models/directories${suffix}`, { signal });
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

  /** 注册目录内的模型资产，返回新增或更新的模型记录。 */
  async registerModelDirectory(directoryId: string): Promise<ModelArtifact[]> {
    return (
      await request<{ data: ModelArtifact[] }>('/api/v1/models/directories/register', {
        method: 'POST',
        body: JSON.stringify({ directory_id: directoryId }),
      })
    ).data;
  },

  /** Submit a native artifact load, rejecting explicit options the server cannot honor. */
  loadModel(
    artifactId: string,
    contextSize: number | null = null,
    prefillChunkSize: number | 'auto' = 'auto',
    policy: { pin?: boolean; idle_ttl_seconds?: number | null; device_ids?: string[]; context_management?: ModelLoadContextSettings; gguf_companions?: string[] } = {},
  ): Promise<{ operation_id: string; status: 'accepted' }> {
    if (prefillChunkSize !== 'auto' || policy.device_ids?.length || policy.gguf_companions?.length
      || policy.context_management && !policy.context_management.automatic) {
      return Promise.reject(new Error('The native server supports context size, pin and idle timeout only. Explicit device, KV, companion and prefill settings are not supported.'));
    }
    return request('/api/v1/models/load', {
      method: 'POST',
      body: JSON.stringify({
        artifact_id: artifactId,
        requirements: contextSize == null ? {} : { context_size: contextSize },
        policy: {
          pin: policy.pin ?? false,
          ...(!policy.pin && policy.idle_ttl_seconds != null ? { idle_ttl_seconds: policy.idle_ttl_seconds } : {}),
        },
      }),
    });
  },

  /** 提交实例卸载任务；强制卸载需调用方先确认用户意图。 */
  unloadModel(
    instanceId: string,
    force = false,
  ): Promise<{ operation_id: string; status: 'accepted' }> {
    return request('/api/v1/models/unload', {
      method: 'POST',
      body: JSON.stringify({ instance_id: instanceId, force }),
    });
  },

  /** 在指定模型仓库搜索远程模型条目。 */
  async searchHubModels(
    provider: HubModelSummary['provider'],
    query: string,
    options: { format?: 'all' | 'tq' | 'gguf' | 'mlx'; sort?: 'downloads' | 'likes' | 'updated'; limit?: number; signal?: AbortSignal } = {},
  ): Promise<HubModelSummary[]> {
    const params = new URLSearchParams({ provider, query, limit: String(options.limit ?? 40), format: options.format ?? 'all', sort: options.sort ?? 'downloads' });
    return (await request<{ data: HubModelSummary[] }>(`/api/v1/hub/models?${params}`, { signal: options.signal })).data;
  },

  hubModelReadme(model: Pick<HubModelInfo, 'provider' | 'repo_id' | 'revision'>, signal?: AbortSignal): Promise<{ content: string | null }> {
    const [owner, name] = model.repo_id.split('/');
    return request(`/api/v1/hub/models/${model.provider}/${encodeURIComponent(owner)}/${encodeURIComponent(name)}/readme?${new URLSearchParams({ revision: model.revision })}`, { signal });
  },

  /** 读取内置官方目录以及服务端评估的设备适配建议。 */
  officialHubModels(refresh = false, signal?: AbortSignal): Promise<OfficialModelList> {
    return request(`/api/v1/hub/official${refresh ? '?refresh=true' : ''}`, { signal });
  },

  /** 解析 owner/repo 或受信任的 Hugging Face、ModelScope 仓库链接。 */
  resolveHubModel(
    reference: string,
    fallbackProvider: HubModelSummary['provider'],
    signal?: AbortSignal,
  ): Promise<HubModelInfo> {
    return request('/api/v1/hub/resolve', {
      method: 'POST',
      signal,
      body: JSON.stringify({
        reference,
        fallback_provider: fallbackProvider,
      }),
    });
  },

  /** 读取远程模型的版本、文件列表与容量信息。 */
  hubModelInfo(
    provider: HubModelSummary['provider'],
    repoId: string,
    revision?: string,
    signal?: AbortSignal,
  ): Promise<HubModelInfo> {
    const [owner, name] = repoId.split('/', 2);
    const suffix = revision ? `?revision=${encodeURIComponent(revision)}` : '';
    return request(
      `/api/v1/hub/models/${provider}/${encodeURIComponent(owner)}/${encodeURIComponent(name)}${suffix}`, { signal },
    );
  },

  /** 删除服务工作区内的指定产物，调用方应先确认。 */
  removeWorkspaceArtifact(uri: string): Promise<Record<string, unknown>> {
    return request('/api/v1/artifacts/remove', {
      method: 'POST',
      body: JSON.stringify({ artifact_uri: uri }),
    });
  },

  /** 获取任务产物及其来源关系，供量化和数据集选择使用。 */
  async artifactLineage(limit = 200): Promise<ArtifactLineage[]> {
    return (await request<{ data: ArtifactLineage[] }>(`/api/v1/artifacts/lineage?limit=${limit}`))
      .data;
  },
};
