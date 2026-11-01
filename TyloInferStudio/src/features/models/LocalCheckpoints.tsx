import { useEffect, useId, useState, type CSSProperties } from 'react';
import { useSettings } from '../settings/SettingsProvider';
import { ListToolbar } from '../../shared/ui/ListToolbar';
import { Icon, SectionLabel, TMPanel, EmptyPanel } from '../../app/display';
import type { useModelCatalog } from './useModelCatalog';
import { ModelVendorMark } from '../../app/ModelVendorMark';
import { ModelLoadProgress } from './ModelLoadProgress';
import { formatNumber } from '../../app/formatters';
import { ModelPressureBars } from './ModelMemoryPressure';
import { availableResidentMemory, hasSeparateVram } from '../runtime/memoryArchitecture';
import { ModelDeleteDialog } from './ModelDeleteDialog';
import { availableLoadMemory } from './loadMemoryEstimate';
import { groupCheckpoints } from './checkpointGroups';

export function LocalCheckpoints({ catalog }: { catalog: ReturnType<typeof useModelCatalog> }) {
  const { tr } = useSettings();
  const { runtime, artifacts, busy, instances, modelFilter,
    setModelFilter, filteredArtifacts, unloadInstance, loadArtifact, chooseModelDirectory, openModelFiles, prepareModelDeletion } = catalog;
  const [status, setStatus] = useState('all');
  const [architecture, setArchitecture] = useState('all');
  const [format, setFormat] = useState('all');
  const [sort, setSort] = useState('recent');
  const [companions, setCompanions] = useState<Record<string, string>>({});
  const headingId = useId();
  const isComponent = (item: typeof artifacts[number]) => !!item.asset_kind && item.asset_kind !== 'model';
  const components = artifacts.filter(isComponent).length;
  useEffect(() => { setCompanions({}); }, [catalog.connectionRevision]);
  const isLoaded = (name: string) => instances.some(item => item.model === name && item.state !== 'failed') || name === runtime?.model;
  const visibleArtifacts = filteredArtifacts.filter(item =>
    (architecture === 'all' || item.architecture === architecture) && (format === 'all' || item.format === format) &&
    (status === 'all' || status === 'loaded' && isLoaded(item.name) || status === 'loadable' && item.loadable ||
      status === 'incomplete' && !item.complete || status === 'components' && isComponent(item) || status === 'unavailable' && !item.loadable && !isComponent(item)))
    .sort((a, b) => sort === 'name' ? a.name.localeCompare(b.name) : sort === 'size'
      ? b.total_bytes - a.total_bytes : (Date.parse(b.modified_at) || 0) - (Date.parse(a.modified_at) || 0));
  const families = groupCheckpoints(visibleArtifacts);
  const filterCount = [status, architecture, format].filter(value => value !== 'all').length;
  const separate = hasSeparateVram(runtime);
  const available = separate ? runtime?.runtime_memory_headroom_bytes : availableResidentMemory(runtime);
  const loadMemory = availableLoadMemory(runtime, catalog.loadDeviceIds);
  const system = separate ? { physical_memory_bytes: runtime?.host_memory_total_bytes, memory_pools: [
    { kind: 'vram' as const, capacity_bytes: runtime?.device_memory_total_bytes },
    { kind: 'ram' as const, capacity_bytes: runtime?.host_memory_total_bytes },
  ] } : undefined;
  const gib = (bytes?: number | null) => bytes == null ? '—' : `${formatNumber(bytes / 2 ** 30, 1)} GiB`;
  return (
    <>
      <SectionLabel
        title={tr('本地检查点', 'Local checkpoints')}
        subtitle={tr(`${artifacts.length - components} 个本地模型${components ? ` · ${components} 个配套资产` : ''}`,
          `${artifacts.length - components} local models${components ? ` · ${components} companion assets` : ''}`)}
      />
      <ListToolbar label={tr('搜索本地检查点', 'Search local checkpoints')} placeholder={tr('按模型名称搜索', 'Search model names')}
        query={modelFilter} onQueryChange={setModelFilter} count={visibleArtifacts.length} total={artifacts.length}
        afterFilters={!separate && <small className="checkpoint-available-memory">{tr('剩余可用内存', 'Remaining available memory')} <span>{gib(available)}</span></small>}
        activeFilters={filterCount} onReset={() => { setModelFilter(''); setStatus('all'); setArchitecture('all'); setFormat('all'); setSort('recent'); }}>
        <label>{tr('模型状态', 'Model status')}<select aria-label={tr('模型状态', 'Model status')} value={status} onChange={event => setStatus(event.target.value)}>
          <option value="all">{tr('全部', 'All')}</option><option value="loaded">{tr('已加载', 'Loaded')}</option>
          <option value="loadable">{tr('可直接加载', 'Loadable')}</option><option value="incomplete">{tr('分片不全', 'Incomplete shards')}</option><option value="unavailable">{tr('不可直接加载', 'Not loadable')}</option>
          <option value="components">{tr('配套资产', 'Companion assets')}</option>
        </select></label>
        <label>{tr('架构', 'Architecture')}<select aria-label={tr('筛选架构', 'Filter architecture')} value={architecture} onChange={event => setArchitecture(event.target.value)}>
          <option value="all">{tr('全部', 'All')}</option>{[...new Set(artifacts.map(item => item.architecture).filter(Boolean))].sort().map(value => <option key={value} value={value}>{value}</option>)}
        </select></label>
        <label>{tr('格式', 'Format')}<select aria-label={tr('筛选格式', 'Filter format')} value={format} onChange={event => setFormat(event.target.value)}>
          <option value="all">{tr('全部', 'All')}</option><option value="tq">TQ</option><option value="hf">HuggingFace / MLX</option><option value="gguf">GGUF</option>
        </select></label>
        <label>{tr('排序', 'Sort')}<select aria-label={tr('检查点排序', 'Checkpoint sort')} value={sort} onChange={event => setSort(event.target.value)}>
          <option value="recent">{tr('最近修改', 'Recently modified')}</option><option value="name">{tr('模型名称', 'Model name')}</option><option value="size">{tr('文件大小', 'File size')}</option>
        </select></label>
      </ListToolbar>
      {visibleArtifacts.length > 0 ? (
        <div className="checkpoint-family-groups">
          {families.map((family, familyIndex) => <TMPanel className="model-catalog-panel model-library-panel checkpoint-family-panel" key={family.key}>
            <header className="checkpoint-family-heading">
              <h3>{family.name || tr('其它', 'Other')}</h3>
              <span>{(() => {
                const count = family.models.reduce((sum, group) => sum + group.artifacts.filter(isComponent).length, 0);
                return tr(`${family.count - count} 个版本${count ? ` · ${count} 个配套资产` : ''}`,
                  `${family.count - count} variants${count ? ` · ${count} companion assets` : ''}`);
              })()}</span>
            </header>
            <div className="model-list" style={{ '--checkpoint-group-rows': family.models.map(group => `auto repeat(${group.artifacts.length}, minmax(0, 1fr))`).join(' ') } as CSSProperties}>
              {family.models.map((group, modelIndex) => <section className="checkpoint-model-group" key={group.key}
                aria-labelledby={`${headingId}-${familyIndex}-${modelIndex}`}>
                <h4 className="checkpoint-model-heading" id={`${headingId}-${familyIndex}-${modelIndex}`}>{group.name}</h4>
            {group.artifacts.map((item) => {
              const component = isComponent(item);
              const componentLabel = item.asset_kind === 'engram' ? tr('可独立使用的Engram', 'Independently usable Engram')
                : tr('可独立使用的PLE', 'Independently usable PLE');
              const instance = instances.find(
                (candidate) => candidate.model === item.name && candidate.state !== 'failed',
              );
              const loaded = Boolean(instance) || item.name === runtime?.model;
              const streamsExperts = !loaded && item.loadable && (item.estimated_weight_bytes_by_role?.experts ?? 0) > 0
                && loadMemory != null && item.estimated_resident_weight_bytes != null && item.estimated_resident_weight_bytes > loadMemory;
              const unavailableLabel = item.complete && item.format === 'hf'
                ? tr('需先转换', 'Convert first') : item.missing_shards
                  ? tr('分片不全', 'Incomplete shards') : tr('不可用', 'Invalid');
              const streamingNotice = tr('剩余内存不足，超出部分专家会被放入 SSD 进行流式加载推理。',
                'Insufficient remaining memory. Experts that do not fit will be streamed from SSD during inference.');
              return (
                <div className={component ? 'model-row component-asset-row' : 'model-row'} key={item.id}>
                  <span className={component ? 'model-state' : loaded ? 'model-state active'
                    : item.loadable ? 'model-state' : 'model-state failed'} />
                  <div className="checkpoint-row-details">
                    <strong>{item.name}</strong>
                    {component && <small className="component-asset-label">{componentLabel}</small>}
                    <small>{gib(item.total_bytes)} · {item.missing_shards
                      ? tr(`${item.shard_count} 个分片，缺 ${item.missing_shards} 片`, `${item.shard_count} shards, ${item.missing_shards} missing`)
                      : tr(`${item.shard_count} 个分片`, `${item.shard_count} shards`)}</small>
                    {!component && <small>{tr('预计常驻内存', 'Estimated resident memory')} {gib(item.estimated_resident_weight_bytes)}
                      {(item.ssd_ple_bytes ?? 0) > 0 && ` · SSD PLE ${gib(item.ssd_ple_bytes)}`}</small>}
                    {instance && <small className="model-library-runtime">{instance.state === 'busy' ? tr('使用中', 'Busy')
                      : instance.state === 'loading' ? tr('加载中', 'Loading') : instance.state === 'unloading' ? tr('卸载中', 'Unloading') : tr('就绪', 'Ready')}
                      {instance.context_size != null && ` · ${formatNumber(instance.context_size)} ctx`}
                      {instance.pinned ? ` · ${tr('固定', 'Pinned')}` : instance.idle_ttl_seconds != null ? ` · TTL ${instance.idle_ttl_seconds}s` : ''}
                    </small>}
                    {!component && <ModelLoadProgress model={item.name} />}
                    {!loaded && item.gguf_companion_required && <details className="model-companion-picker">
                      <summary>{tr('Engram 配套文件', 'Engram companion')}</summary>
                      <input aria-label={tr(`Engram 路径：${item.name}`, `Engram path: ${item.name}`)}
                        disabled={busy} value={companions[item.id] ?? ''}
                        placeholder={tr('服务端的 GGUF/TQ/MFQ 表文件或 Engram 文件夹', 'Server-side GGUF/TQ/MFQ table file or Engram folder')}
                        onChange={event => setCompanions(previous => ({ ...previous, [item.id]: event.target.value }))} />
                    </details>}
                  </div>
                  {!component && <div className="local-checkpoint-resources">
                    <div className="local-checkpoint-notice">
                      {!loaded && !item.loadable && <em className="failed" id={`checkpoint-unavailable-${item.id}`} title={item.error || unavailableLabel}>{unavailableLabel}</em>}
                      {streamsExperts && <small className="model-streaming-notice" id={`expert-streaming-${item.id}`} title={streamingNotice}>{streamingNotice}</small>}
                    </div>
                    <div className="local-checkpoint-memory">
                    <ModelPressureBars weights={item.estimated_resident_weight_bytes} roles={item.estimated_weight_bytes_by_role}
                      system={system} available={available} combineMoe showDetails={false}
                      label={tr('预计占剩余可用内存', 'Estimated share of remaining available memory')}
                      tr={tr} />
                    </div>
                  </div>}
                  <div className="model-row-controls"><div className="model-row-actions"><ModelVendorMark name={item.name} architecture={item.architecture} /><button className="model-files-action" disabled={busy} onClick={() => void openModelFiles(item.id)} type="button"><Icon name="folder" size={13} />{tr('模型文件', 'Model files')}</button>
                  {!component && <>
                  <button className="model-settings-action" aria-label={tr(`配置模型：${item.name}`, `Configure model: ${item.name}`)}
                    title={tr('模型设置', 'Model settings')} disabled={busy || !item.loadable && !loaded}
                    onClick={() => catalog.openModelSettings(item.name)} type="button"><Icon name="settings" size={14} /></button>{instance ? (
                    <button className="model-load-action" disabled={busy || instance.state !== 'ready'}
                      onClick={() => void unloadInstance(instance.id)} type="button">
                      {tr('卸载', 'Unload')}
                    </button>
                  ) : loaded ? (
                    <em className="model-load-action">{tr('已加载', 'Loaded')}</em>
                  ) : (
                    <button className="model-load-action" disabled={busy || !item.loadable || (item.gguf_companion_required && !companions[item.id]?.trim())} title={!item.loadable ? item.error || unavailableLabel : undefined}
                      aria-describedby={!item.loadable ? `checkpoint-unavailable-${item.id}` : streamsExperts ? `expert-streaming-${item.id}` : undefined}
                      onClick={() => void (companions[item.id]?.trim()
                        ? loadArtifact(item.name, [companions[item.id].trim()]) : loadArtifact(item.name))} type="button">
                      {tr('加载', 'Load')}
                    </button>
                  )}<button className="model-delete-action" aria-label={tr(`删除模型：${item.name}`, `Delete model: ${item.name}`)}
                    disabled={busy || loaded} title={loaded ? tr('请先卸载模型再删除', 'Unload the model before deleting it') : tr('删除模型', 'Delete model')}
                    onClick={() => void prepareModelDeletion(item.id)} type="button"><Icon name="trash" size={14} /></button></>}
                    </div>
                  </div>
                </div>
              );
            })}
              </section>)}
            </div>
          </TMPanel>)}
        </div>
      ) : (
        <EmptyPanel
          icon="folder"
          title={tr(modelFilter || filterCount ? '没有匹配的本地模型' : '还没有本地模型',
            modelFilter || filterCount ? 'No local model matches' : 'No local models yet')}
          message={tr('添加一个模型文件夹即可开始。', 'Add a model folder to get started.')}
          action={
            <button className="primary screen-header-action" disabled={busy}
              onClick={() => void chooseModelDirectory()} type="button">
              <Icon name="folder" size={14} />{tr('选择模型文件夹', 'Choose model folder')}
            </button>
          }
        />
      )}
      <ModelDeleteDialog catalog={catalog} />
    </>
  );
}
