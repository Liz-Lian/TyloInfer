
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type Dispatch,
  type ReactNode,
  type SetStateAction,
} from 'react';
import { loadSettings, SETTINGS_KEY, type GenerationSettings } from './configuration';

interface SettingsContextValue {
  settings: GenerationSettings;

  updateSettings: (patch: Partial<GenerationSettings>) => void;

  replaceSettings: Dispatch<SetStateAction<GenerationSettings>>;
  tr: (zh: string, en: string) => string;
  english: boolean;
  contextSize: number | null;

  setContextSize: Dispatch<SetStateAction<number | null>>;
}

const SettingsContext = createContext<SettingsContextValue | null>(null);


export function SettingsProvider({ children }: { children: ReactNode }) {
  const [settings, replaceSettings] = useState(loadSettings);
  const [contextSize, setContextSize] = useState<number | null>(null);
  const english =
    settings.language === 'en' ||
    (settings.language === 'system' && !navigator.language.toLowerCase().startsWith('zh'));
  const tr = useCallback((zh: string, en: string) => (english ? en : zh), [english]);
  const updateSettings = useCallback(
    (patch: Partial<GenerationSettings>) =>
      replaceSettings((current) => ({ ...current, ...patch })),
    [],
  );

  useEffect(() => {
    try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings)); } catch {}
    document.documentElement.dataset.theme = settings.theme;
  }, [settings]);

  const value = useMemo(
    () => ({ settings, replaceSettings, updateSettings, english, tr, contextSize, setContextSize }),
    [settings, updateSettings, english, tr, contextSize],
  );
  return <SettingsContext.Provider value={value}>{children}</SettingsContext.Provider>;
}


export function useSettings() {
  const context = useContext(SettingsContext);
  if (!context) throw new Error('SettingsProvider is required');
  return context;
}
