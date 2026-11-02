import type { ReactNode } from 'react';
import { useRuntime } from './RuntimeProvider';
import { serverCapabilities } from './serverCapabilities';
import { ScreenHeader, EmptyPanel } from './display';
import { useSettings } from '../features/settings/SettingsProvider';

const features = {
  resources: ['资源', 'Resources', '当前原生服务尚未提供资源监控与缓存管理，资源数值暂不可用。', 'The native server does not provide resource monitoring or cache management. Resource measurements are unavailable.'],
  analysis: ['模型分析', 'Model analysis', '当前原生服务尚未提供模型结构与张量分析。已注册模型仍可在模型页面加载。', 'The native server does not provide architecture or tensor analysis. Registered models can still be loaded from Models.'],
  evaluations: ['评测', 'Evaluations', '当前原生服务尚未提供数据集、质量评测与性能基准任务。', 'The native server does not provide datasets, quality evaluations or benchmark jobs.'],
};

/** Preserve future routes without mounting effects for unavailable server resources. */
export function ServerFeature({ feature, children }: { feature: keyof typeof features; children: ReactNode }) {
  const { runtime } = useRuntime();
  const { tr } = useSettings();
  if (serverCapabilities(runtime)[feature]) return children;
  const [zh, en, detailZh, detailEn] = features[feature];
  return <section className="dashboard-view">
    <ScreenHeader title={tr(zh, en)} subtitle={tr('当前服务暂不支持此功能', 'Unavailable on this server')} />
    <EmptyPanel icon="info" title={tr('暂不可用', 'Unavailable')} message={tr(detailZh, detailEn)} />
  </section>;
}
