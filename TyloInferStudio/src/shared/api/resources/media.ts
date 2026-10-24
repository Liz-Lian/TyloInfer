
import type { MediaResource, DocumentResource } from '../types';
import { request, apiUrl, errorFromResponse, authorizedHeaders } from '../client';

export const mediaApi = {

  async uploadMedia(file: File, mimeType?: string): Promise<MediaResource> {
    const digest = await crypto.subtle.digest('SHA-256', await file.arrayBuffer());
    const sha256 = Array.from(new Uint8Array(digest), (value) =>
      value.toString(16).padStart(2, '0'),
    ).join('');
    const response = await fetch(apiUrl('/api/v1/media'), {
      method: 'POST',
      headers: authorizedHeaders({
        'Content-Type': mimeType || file.type || 'application/octet-stream',
        'X-Content-SHA256': sha256,
      }),
      body: file,
    });
    if (!response.ok) throw await errorFromResponse(response);
    return (await response.json()) as MediaResource;
  },


  mediaUrl(id: string): string {
    return apiUrl(`/api/v1/media/${id}`);
  },


  async fetchMedia(id: string, signal?: AbortSignal): Promise<Blob> {
    const response = await fetch(apiUrl(`/api/v1/media/${id}`), {
      headers: authorizedHeaders(),
      signal,
    });
    if (!response.ok) throw await errorFromResponse(response);
    return response.blob();
  },


  createDocument(mediaId: string, name: string): Promise<DocumentResource> {
    return request('/api/v1/documents', {
      method: 'POST',
      body: JSON.stringify({ media_id: mediaId, name }),
    });
  },
};
