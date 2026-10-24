
import { useChatPageState } from './hooks/useChatPageState';
import { ChatSessionSidebar } from './components/ChatSessionSidebar';
import { ChatPageHeader } from './components/ChatPageHeader';
import { ChatMessageList } from './components/ChatMessageList';
import { ChatInputArea } from './components/ChatInputArea';


export function ChatPage() {
  const page = useChatPageState();
  return (
    <section className={'chat-view ' + (page.chatSessionsOpen ? 'sessions-open' : 'sessions-collapsed')}>
      <ChatSessionSidebar page={page} />
      <ChatPageHeader page={page} />
      <ChatMessageList page={page} />
      <ChatInputArea page={page} />
    </section>
  );
}
