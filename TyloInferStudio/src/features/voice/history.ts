


export interface VoiceMessage {
  id: string;
  sessionId: string;
  role: "user" | "assistant";
  text: string;
  audioId?: string;
  pending?: boolean;
  created_at: string;
}

export interface LiveVoiceOutput {
  sessionId: string;
  text: string;
}

export const VOICE_HISTORY_KEY = "tyloinfer.studio.voice-history.v1";


export function loadVoiceHistory(): VoiceMessage[] {
  try {
    const value = JSON.parse(localStorage.getItem(VOICE_HISTORY_KEY) || "[]");
    return Array.isArray(value) ? value.slice(-200) : [];
  } catch {
    return [];
  }
}
