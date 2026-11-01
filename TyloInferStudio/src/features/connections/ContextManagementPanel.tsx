import { SectionLabel, TMPanel } from '../../app/display';
import { useSettings } from '../settings/SettingsProvider';
import { ModelContextSettings } from '../runtime/ModelContextSettings';

export function ContextManagementPanel() {
  const { tr } = useSettings();
  return <>
    <SectionLabel title={tr('上下文默认值', 'Context defaults')} />
    <TMPanel className="server-settings-panel server-context-panel">
      <div className="setting-list"><ModelContextSettings scope="global" /></div>
    </TMPanel>
  </>;
}
