
import { useSyncExternalStore } from 'react';

let level = 0;
const listeners = new Set<() => void>();

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}


export function getVoiceLevel(): number {
  return level;
}


export function setVoiceLevel(nextLevel: number): void {
  if (Object.is(level, nextLevel)) return;
  level = nextLevel;
  listeners.forEach((listener) => listener());
}


export function resetVoiceLevel(): void {
  setVoiceLevel(0);
}


export function useVoiceLevel(): number {
  return useSyncExternalStore(subscribe, getVoiceLevel, () => 0);
}
