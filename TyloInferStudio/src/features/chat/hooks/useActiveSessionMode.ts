
import type { SessionMode } from '../../../shared/api/types';
import { useConversationSelector } from '../state/conversationStore';






export function useActiveSessionMode(): SessionMode {
  return useConversationSelector(
    (state) => state.sessions.find((session) => session.id === state.activeId)?.mode ?? 'text',
  );
}
