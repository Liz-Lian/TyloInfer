
import type { GenerationPresetResource } from '../types';
import { request, apiUrl, errorFromResponse, authorizedHeaders } from '../client';

export const presetsApi = {

  async generationPresets(): Promise<GenerationPresetResource[]> {
    return (await request<{ data: GenerationPresetResource[] }>('/api/v1/presets')).data;
  },


  createGenerationPreset(
    body: Omit<GenerationPresetResource, 'id' | 'created_at' | 'updated_at'>,
  ): Promise<GenerationPresetResource> {
    return request('/api/v1/presets', { method: 'POST', body: JSON.stringify(body) });
  },


  updateGenerationPreset(
    id: string,
    body: Omit<GenerationPresetResource, 'id' | 'created_at' | 'updated_at'>,
  ): Promise<GenerationPresetResource> {
    return request(`/api/v1/presets/${id}`, {
      method: 'PUT',
      body: JSON.stringify(body),
    });
  },


  async deleteGenerationPreset(id: string): Promise<void> {
    const response = await fetch(apiUrl(`/api/v1/presets/${id}`), {
      method: 'DELETE',
      headers: authorizedHeaders(),
    });
    if (!response.ok) throw await errorFromResponse(response);
  },
};
