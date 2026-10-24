import { useMemo, useRef, useState } from 'react';
import type { EditDraft } from '../SavedMessageList';
import { sessionsApi } from '../../../shared/api/resources/sessions';
import type { ContentPart, Message, Session } from '../../../shared/api/types';
import { textParts, isMediaPart } from '../messageParts';
import type { useConversationSessions } from './useConversationSessions';
import { errorMessage } from '../../../app/formatters';

interface MessageActionOptions {
  conversation: ReturnType<typeof useConversationSessions>;
  blocked: boolean;

  generate: (
    session: Session,
    input: ContentPart[],
    optimistic?: boolean,
    role?: 'user' | 'tool',
  ) => Promise<void>;

  setBusy: (busy: boolean) => void;
}


export function useMessageActions({
  conversation,
  blocked,
  generate,
  setBusy,
}: MessageActionOptions) {
  const latest = useRef({ conversation, blocked, generate, setBusy });
  latest.current = { conversation, blocked, generate, setBusy };

  return useMemo(() => {
    const current = (sessionId: string) =>
      latest.current.conversation.activeIdRef.current === sessionId;


    async function saveEdit(
      message: Message,
      draftText: string,
      onCommitted: () => void,
    ): Promise<boolean> {
      const { conversation, blocked, generate, setBusy } = latest.current;
      const { active, messages, setMessages, setSessions, setResponses, setError } = conversation;
      if (!active || blocked) return false;
      const index = messages.findIndex((item) => item.id === message.id);
      const text = draftText.trim();
      if (index < 0 || (!text && !message.parts.some(isMediaPart))) return false;
      setBusy(true);
      try {
        const rewound = await sessionsApi.rewindSession(active.id, active.revision, message.id, false);
        if (!current(active.id)) return false;
        const parts: ContentPart[] = [
          ...(text ? [{ type: 'text' as const, text }] : []),
          ...message.parts.filter((part) => isMediaPart(part) || part.type === 'document'),
        ];
        const previous = messages.slice(0, index);
        setSessions((items) =>
          items.map((session) => (session.id === rewound.id ? rewound : session)),
        );
        setMessages([...previous, { ...message, parts }]);
        setResponses((items) =>
          Object.fromEntries(
            Object.entries(items).filter(([id]) => previous.some((item) => item.id === id)),
          ),
        );
        onCommitted();
        await generate(rewound, parts, false);
        return true;
      } catch (cause) {
        if (current(active.id)) setError(errorMessage(cause));
        return false;
      } finally {
        setBusy(false);
      }
    }


    async function regenerate(message: Message) {
      const { conversation, blocked, generate, setBusy } = latest.current;
      const { active, messages, setMessages, setSessions, setResponses, setError } = conversation;
      if (!active || blocked || message.role !== 'assistant') return;
      const index = messages.findIndex((item) => item.id === message.id);
      const user = messages
        .slice(0, index)
        .reverse()
        .find((item) => item.role === 'user');
      if (!user?.parts.length) return;
      setBusy(true);
      try {
        const rewound = await sessionsApi.rewindSession(active.id, active.revision, user.id, false);
        if (!current(active.id)) return;
        const previous = messages.slice(0, messages.indexOf(user) + 1);
        setSessions((items) =>
          items.map((session) => (session.id === rewound.id ? rewound : session)),
        );
        setMessages(previous);
        setResponses((items) =>
          Object.fromEntries(
            Object.entries(items).filter(([id]) => previous.some((item) => item.id === id)),
          ),
        );
        await generate(rewound, user.parts, false);
      } catch (cause) {
        if (current(active.id)) setError(errorMessage(cause));
      } finally {
        setBusy(false);
      }
    }


    async function copyMessage(message: Message) {
      const { setError } = latest.current.conversation;
      try {
        const parts = textParts(message);
        await navigator.clipboard.writeText(
          [parts.reasoning, parts.text].filter(Boolean).join('\n\n'),
        );
      } catch (cause) {
        setError(errorMessage(cause));
      }
    }

    return { saveEdit, regenerate, copyMessage };
  }, []);
}
