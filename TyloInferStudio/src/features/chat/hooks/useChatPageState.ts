
import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router';
import { useChat } from '../ChatProvider';
import { useConversationSelector } from '../state/conversationStore';
import { useChatAutoScroll } from './useChatAutoScroll';
import type { EditDraft } from '../SavedMessageList';


export function useChatPageState() {
  const chat = useChat();
  const navigate = useNavigate();
  const activeId = useConversationSelector((state) => state.activeId);
  const active = useConversationSelector(
    (state) => state.sessions.find((session) => session.id === state.activeId) ?? null,
  );
  const [editDraft, setEditDraft] = useState<EditDraft | null>(null);
  const [chatSessionsOpen, setChatSessionsOpen] = useState(
    () => window.matchMedia('(min-width: 681px)').matches,
  );
  const scroll = useChatAutoScroll(activeId, true);

  useEffect(() => setEditDraft(null), [activeId]);
  useEffect(() => {
    const viewport = window.matchMedia('(max-width: 680px)');
    const close = () => {
      if (viewport.matches) setChatSessionsOpen(false);
    };
    viewport.addEventListener('change', close);
    return () => viewport.removeEventListener('change', close);
  }, []);


  async function saveCurrentEdit(message: Parameters<typeof chat.messageActions.saveEdit>[0]) {
    const draft = editDraft;
    if (!draft || draft.messageId !== message.id) return;
    await chat.messageActions.saveEdit(message, draft.text, () => {
      setEditDraft((current) =>
        current?.messageId === draft.messageId && current.text === draft.text ? null : current,
      );
    });
  }


  function selectSession(id: string) {
    if (chat.busy) return;
    chat.conversation.selectSession(id);
    if (window.matchMedia('(max-width: 680px)').matches) setChatSessionsOpen(false);
  }


  async function createSession() {
    await chat.conversation.createSession(active?.mode);
    if (window.matchMedia('(max-width: 680px)').matches) setChatSessionsOpen(false);
  }


  function chooseModelDirectory() {
    navigate('/models');
  }

  function selectModel(value: string) {
    if (chat.inference.availableModelNames.includes(value)) chat.inference.setSelectedModel(value);
  }

  return {
    chat,
    activeId,
    active,
    editDraft,
    setEditDraft,
    saveCurrentEdit,
    chatSessionsOpen,
    setChatSessionsOpen,
    scroll,
    selectSession,
    createSession,
    chooseModelDirectory,
    selectModel,
  };
}

export type ChatPageState = ReturnType<typeof useChatPageState>;
