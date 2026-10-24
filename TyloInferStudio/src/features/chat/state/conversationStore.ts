
import { create } from 'zustand';
import type { Dispatch, SetStateAction } from 'react';
import type { Message, ResponseResource, Session } from '../../../shared/api/types';

type Responses = Record<string, ResponseResource>;


export interface ConversationState {
  sessions: Session[];
  activeId: string | null;
  messages: Message[];
  responses: Responses;
  historyLoadedId: string | null;
  epoch: number;

  reset: () => void;

  setSessions: Dispatch<SetStateAction<Session[]>>;

  setActiveId: Dispatch<SetStateAction<string | null>>;

  setMessages: Dispatch<SetStateAction<Message[]>>;

  setResponses: Dispatch<SetStateAction<Responses>>;

  loadSessions: (epoch: number, sessions: Session[]) => boolean;

  beginHistory: (id: string) => void;

  applyHistory: (
    epoch: number,
    id: string,
    messages: Message[],
    responses: ResponseResource[],
  ) => void;

  applySynchronized: (session: Session, messages: Message[], responses: ResponseResource[]) => void;
}


function indexResponses(responses: ResponseResource[]): Responses {
  return Object.fromEntries(
    responses
      .filter((response) => response.output_message_id)
      .map((response) => [response.output_message_id!, response]),
  );
}


export const useConversationStore = create<ConversationState>()((set) => ({
  sessions: [],
  activeId: null,
  messages: [],
  responses: {},
  historyLoadedId: null,
  epoch: 0,
  reset: () =>
    set((state) => ({
      sessions: [],
      activeId: null,
      messages: [],
      responses: {},
      historyLoadedId: null,
      epoch: state.epoch + 1,
    })),
  setSessions: (value) =>
    set((state) => ({
      sessions: typeof value === 'function' ? value(state.sessions) : value,
    })),
  setActiveId: (value) =>
    set((state) => {
      const activeId = typeof value === 'function' ? value(state.activeId) : value;
      return activeId === state.activeId
        ? state
        : {
            activeId,
            messages: [],
            responses: {},
            historyLoadedId: null,
          };
    }),
  setMessages: (value) =>
    set((state) => ({
      messages: typeof value === 'function' ? value(state.messages) : value,
    })),
  setResponses: (value) =>
    set((state) => ({
      responses: typeof value === 'function' ? value(state.responses) : value,
    })),
  loadSessions: (epoch, sessions) => {
    if (useConversationStore.getState().epoch !== epoch) return false;
    set({
      sessions,
      activeId: sessions[0]?.id ?? null,
      messages: [],
      responses: {},
      historyLoadedId: null,
    });
    return true;
  },
  beginHistory: (id) =>
    set((state) =>
      state.activeId === id ? { messages: [], responses: {}, historyLoadedId: null } : state,
    ),
  applyHistory: (epoch, id, messages, responses) =>
    set((state) =>
      state.epoch === epoch && state.activeId === id
        ? { messages, responses: indexResponses(responses), historyLoadedId: id }
        : state,
    ),
  applySynchronized: (session, messages, responses) =>
    set((state) => ({
      sessions: state.sessions.map((item) => (item.id === session.id ? session : item)),
      ...(state.activeId === session.id ? { messages, responses: indexResponses(responses) } : {}),
    })),
}));


export function useConversationSelector<T>(selector: (state: ConversationState) => T): T {
  return useConversationStore(selector);
}


export const conversationActions = {
  reset: useConversationStore.getState().reset,
  applySynchronized: useConversationStore.getState().applySynchronized,
  applySync: useConversationStore.getState().applySynchronized,
  setSessions: useConversationStore.getState().setSessions,
  setActiveId: useConversationStore.getState().setActiveId,
  setMessages: useConversationStore.getState().setMessages,
  setResponses: useConversationStore.getState().setResponses,
};
