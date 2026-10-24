
import { create } from 'zustand';

interface DraftState {
  drafts: Record<string, string>;

  setDraft: (sessionId: string, text: string) => void;
}


export const useDraftStore = create<DraftState>()((set) => ({
  drafts: {},
  setDraft: (sessionId, text) =>
    set((state) => {
      if ((state.drafts[sessionId] ?? '') === text) return state;
      const drafts = { ...state.drafts };
      if (text) drafts[sessionId] = text;
      else delete drafts[sessionId];
      return { drafts };
    }),
}));
