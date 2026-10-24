
import type {
  SessionMode,
  MessageRole,
  ContentPart,
  Session,
  SessionArchive,
  Message,
  ResponseResource,
} from '../types';
import { request, apiUrl, errorFromResponse, authorizedHeaders } from '../client';

export const sessionsApi = {

  async listSessions(offset = 0): Promise<Session[]> {
    return (await request<{ data: Session[] }>('/api/v1/sessions?limit=200' + (offset ? `&offset=${offset}` : ''))).data;
  },


  createSession(
    model?: string,
    mode: SessionMode = 'text',
    title?: string,
    metadata?: Record<string, unknown>,
  ): Promise<Session> {
    return request('/api/v1/sessions', {
      method: 'POST',
      body: JSON.stringify({ model, mode, title, metadata: metadata ?? {} }),
    });
  },


  getSession(id: string, signal?: AbortSignal): Promise<Session> {
    return request(`/api/v1/sessions/${id}`, { signal });
  },


  updateSession(
    id: string,
    update: { title?: string | null; mode?: SessionMode; metadata?: Record<string, unknown> },
  ): Promise<Session> {
    return request(`/api/v1/sessions/${id}`, {
      method: 'PATCH',
      body: JSON.stringify(update),
    });
  },


  async listMessages(id: string, signal?: AbortSignal): Promise<Message[]> {
    return (await request<{ data: Message[] }>(`/api/v1/sessions/${id}/messages`, { signal })).data;
  },


  async listResponses(id: string, signal?: AbortSignal): Promise<ResponseResource[]> {
    return (
      await request<{ data: ResponseResource[] }>(`/api/v1/sessions/${id}/responses?limit=1000`, {
        signal,
      })
    ).data;
  },


  cancelResponse(id: string, signal?: AbortSignal): Promise<ResponseResource> {
    return request(`/api/v1/sessions/${id}/responses/cancel`, { method: 'POST', signal });
  },


  appendMessage(
    id: string,
    expectedRevision: number,
    role: MessageRole,
    parts: ContentPart[],
  ): Promise<{ session: Session; message: Message }> {
    return request(`/api/v1/sessions/${id}/messages`, {
      method: 'POST',
      body: JSON.stringify({ expected_revision: expectedRevision, role, parts }),
    });
  },


  forkSession(
    id: string,
    atMessageId: string | null,
    includeMessage = true,
    title?: string | null,
    model?: string,
  ): Promise<Session> {
    return request(`/api/v1/sessions/${id}/fork`, {
      method: 'POST',
      body: JSON.stringify({
        at_message_id: atMessageId,
        include_message: includeMessage,
        title,
        model,
      }),
    });
  },


  rewindSession(
    id: string,
    expectedRevision: number,
    atMessageId: string,
    includeMessage = true,
  ): Promise<Session> {
    return request(`/api/v1/sessions/${id}/rewind`, {
      method: 'POST',
      body: JSON.stringify({
        expected_revision: expectedRevision,
        at_message_id: atMessageId,
        include_message: includeMessage,
      }),
    });
  },


  async deleteSession(id: string): Promise<void> {
    const response = await fetch(apiUrl(`/api/v1/sessions/${id}`), {
      method: 'DELETE',
      headers: authorizedHeaders(),
    });
    if (!response.ok) throw await errorFromResponse(response);
  },


  exportSession(id: string): Promise<SessionArchive> {
    return request(`/api/v1/sessions/${id}/export`);
  },


  importSession(archive: SessionArchive): Promise<{
    session: Session;
    messages_imported: number;
    media_imported: number;
  }> {
    return request('/api/v1/sessions/import', {
      method: 'POST',
      body: JSON.stringify(archive),
    });
  },
};
