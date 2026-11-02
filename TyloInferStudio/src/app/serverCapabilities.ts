import type { RuntimeStatus } from '../shared/api/types';

// 原生能力以实际注册路由为准；模型架构声明不代表服务已提供对应功能。
const nativeCapabilities = {
  agent: false,
  resources: false,
  analysis: false,
  evaluations: false,
  runtimeConfiguration: false,
  officialCatalog: false,
  hubSearch: false,
  hubReadme: false,
  downloadReuse: false,
  cacheReclaim: false,
  modelDeletion: false,
};

export type ServerCapabilities = typeof nativeCapabilities;

/** Restrict unsupported native resources, including while the connection is initializing. */
export function serverCapabilities(runtime: RuntimeStatus | null | undefined): ServerCapabilities {
  if (!runtime || runtime.source === 'native-instances') return nativeCapabilities;
  return Object.fromEntries(Object.keys(nativeCapabilities).map(key => [key, true])) as ServerCapabilities;
}
