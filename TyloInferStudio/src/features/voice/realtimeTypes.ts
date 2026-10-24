

export type VoiceState = "idle" | "connecting" | "listening" | "processing" | "error";


export interface RealtimeSessionConfig {
  sessionId: string;
  systemPrompt: string;
  temperature: number;
  topP: number;
  topK: number;
  repetitionPenalty: number;
}


export interface VoiceTurn {
  id: string;
  sessionId: string;
  text: string;
  audio: Blob | null;
}


export interface VoiceInputTurn {
  id: string;
  sessionId: string;
  audio?: Blob | null;
}


export interface BufferedVoiceTurn {
  id: string;
  sessionId: string;
  inputTurnId: string | null;
  text: string;
  audio: Float32Array[];
  audioRate: number;
}


export interface RealtimeCallbacks {

  onState(state: VoiceState): void;

  onLevel(level: number): void;

  onText(sessionId: string, text: string): void;

  onInputStart(turn: VoiceInputTurn): void;

  onInputEnd(turn: VoiceInputTurn): void;

  onTurn(turn: VoiceTurn): void;

  onError(message: string): void;
}
