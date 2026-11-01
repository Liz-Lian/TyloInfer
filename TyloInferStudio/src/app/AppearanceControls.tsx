import { MoonIcon, SunIcon } from '@phosphor-icons/react';
import { useSettings } from '../features/settings/SettingsProvider';

export function AppearanceControls() {
  const { settings, updateSettings, tr } = useSettings();
  return <div className="appearance-controls" role="group" aria-label={tr('明暗与语言', 'Appearance and language')}>
    <div role="group" aria-label={tr('明暗', 'Appearance')}>
      {(['light', 'dark', 'system'] as const).map(value => <button key={value} type="button"
        aria-label={value === 'light' ? tr('浅色', 'Light') : value === 'dark' ? tr('深色', 'Dark') : tr('自动明暗', 'Automatic appearance')}
        aria-pressed={settings.theme === value} onClick={() => updateSettings({ theme: value })}>
        {value === 'light' ? <SunIcon size={15} /> : value === 'dark' ? <MoonIcon size={15} /> : 'Auto'}
      </button>)}
    </div>
    <span className="appearance-controls-divider" aria-hidden="true" />
    <div role="group" aria-label={tr('语言', 'Language')}>
      {(['en', 'zh-CN', 'system'] as const).map(value => <button key={value} type="button"
        aria-label={value === 'en' ? 'English' : value === 'zh-CN' ? '简体中文' : tr('自动语言', 'Automatic language')}
        aria-pressed={settings.language === value} onClick={() => updateSettings({ language: value })}>
        {value === 'en' ? 'EN' : value === 'zh-CN' ? 'CN' : 'Auto'}
      </button>)}
    </div>
  </div>;
}
