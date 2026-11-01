import { useEffect, useRef } from 'react';
import { useLocation } from 'react-router';
import { useSettings } from '../settings/SettingsProvider';
import { Icon, ScreenHeader } from '../../app/display';
import { useModelCatalog } from './useModelCatalog';
import { ModelDirectoryDialog } from './ModelDirectoryDialog';
import { LoadedModels } from './LoadedModels';
import { ModelSettingsDialog } from './ModelSettingsDialog';
import { LocalCheckpoints } from './LocalCheckpoints';
import { formatBytes } from '../../app/formatters';
import '../../shared/styles/configuration-layout.css';

export function ModelsPage() {
  const catalog = useModelCatalog();
  const { tr } = useSettings();
  const { hash } = useLocation();
  const checkpoints = useRef<HTMLElement | null>(null);
  useEffect(() => {
    if (hash === '#local-checkpoints') checkpoints.current?.scrollIntoView({ block: 'start' });
  }, [hash]);
  const { artifacts, busy,
    openStudioPage, chooseModelDirectory, modelFolderPath } = catalog;
  const totalBytes = artifacts.reduce((sum, artifact) => sum + (artifact.owned_file_bytes ?? artifact.total_bytes), 0);
  const modelCount = artifacts.filter(artifact => !artifact.asset_kind || artifact.asset_kind === 'model').length;
  const loadedCount = new Set(catalog.instances?.filter(instance => instance.state === 'ready' || instance.state === 'busy').map(instance => instance.model)).size;
  return (
    <section className="dashboard-view">
      <ScreenHeader
        title={tr('模型', 'Models')}
        subtitle={tr('管理本地模型与运行实例。', 'Manage local models and runtime instances.')}
        trailing={
          <>
            <button onClick={() => openStudioPage('lab', 'models')} type="button">
              <Icon name="download" size={14} />{tr('模型下载', 'Model downloads')}
            </button>
            <button className="primary" disabled={busy}
              onClick={() => void chooseModelDirectory()} type="button">
              <Icon name="folder" size={14} />{tr('添加模型', 'Add model')}
            </button>
          </>
        }
      />
      <div className="model-folder-location">
        <Icon name="folder" />
        <span>{tr('模型文件夹', 'Model folder')}</span>
        <code title={modelFolderPath}>{modelFolderPath}</code>
      </div>
      <div className="model-library-summary">
        <span>{tr('已加载', 'Loaded')} <strong>{loadedCount}</strong></span>
        <span>{tr('本地模型', 'Local models')} <strong>{modelCount}</strong></span>
        <span>{tr('文件总大小', 'Total file size')} <strong>{totalBytes === 0 ? '0 B' : formatBytes(totalBytes).replace(/\b(KB|MB|GB|TB)\b/g, (unit) => `${unit[0]}iB`)}</strong></span>
      </div>
      <section id="local-checkpoints" ref={checkpoints}>
        <LocalCheckpoints catalog={catalog} />
      </section>
      {catalog.instances?.some(instance => !artifacts.some(artifact => artifact.name === instance.model)) && <LoadedModels catalog={catalog} unlistedOnly />}
      <ModelSettingsDialog key={`${catalog.connectionRevision}:${catalog.settingsModel ?? ''}`} catalog={catalog} />
      <ModelDirectoryDialog catalog={catalog} />
    </section>
  );
}
