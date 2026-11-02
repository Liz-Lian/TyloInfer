/**
 * TyloInfer 页面导航定义，统一维护桌面端与浏览器端共用的路由路径。
 */

export type ViewName = 'chat' | 'agent' | 'dashboard' | 'lab' | 'versions';
export type DashboardPage = 'overview' | 'models' | 'connections' | 'cache' | 'logs';
export type LabPage = 'models' | 'evaluations' | 'analysis';

export interface StudioLocation {
  view: ViewName;
  dashboardPage: DashboardPage;
  labPage: LabPage;
}

export const STUDIO_PATHS = {
  overview: '/',
  chat: '/chat',
  agent: '/agent',
  models: '/models',
  server: '/runtime',
  resources: '/resources',
  modelHub: '/model-hub',
  evaluations: '/evaluations',
  analysis: '/analysis',
  versions: '/versions',
  logs: '/logs',
} as const;

/** Choose an available primary conversation entry without redirecting explicit Agent links. */
export function primaryConversationPath(runtimeSource?: string): string {
  return runtimeSource === 'native-instances' ? STUDIO_PATHS.chat : STUDIO_PATHS.agent;
}

const DASHBOARD_PATHS: Record<DashboardPage, string> = {
  overview: STUDIO_PATHS.overview,
  models: STUDIO_PATHS.models,
  connections: STUDIO_PATHS.server,
  cache: STUDIO_PATHS.resources,
  logs: STUDIO_PATHS.logs,
};

const LAB_PATHS: Record<LabPage, string> = {
  models: STUDIO_PATHS.modelHub,
  evaluations: STUDIO_PATHS.evaluations,
  analysis: STUDIO_PATHS.analysis,
};

/** 规范化 Studio 路径，兼容浏览器或 Tauri 传入的尾部斜杠。 */
export function normalizeStudioPath(pathname: string): string {
  return pathname.length > 1 ? pathname.replace(/\/+$/, '') : pathname;
}

/** 判断路径是否为已注册的业务页面；未知路径由 404 路由接管。 */
export function isStudioPath(pathname: string): boolean {
  return Object.values(STUDIO_PATHS).some((path) => path === normalizeStudioPath(pathname));
}

/** 根据当前 URL 解析 Studio 应展示的业务页面。 */
export function resolveStudioLocation(pathname: string): StudioLocation {
  const normalizedPath = normalizeStudioPath(pathname);
  if (normalizedPath === STUDIO_PATHS.versions) {
    return { view: 'versions', dashboardPage: 'overview', labPage: 'models' };
  }
  if (normalizedPath === STUDIO_PATHS.agent) {
    return { view: 'agent', dashboardPage: 'overview', labPage: 'models' };
  }
  if (normalizedPath === STUDIO_PATHS.chat) {
    return { view: 'chat', dashboardPage: 'overview', labPage: 'models' };
  }

  const dashboardPage = (Object.entries(DASHBOARD_PATHS).find(([, path]) => path === normalizedPath)?.[0]
    ?? 'overview') as DashboardPage;
  const labPage = Object.entries(LAB_PATHS).find(([, path]) => path === normalizedPath)?.[0] as LabPage | undefined;

  return labPage
    ? { view: 'lab', dashboardPage: 'overview', labPage }
    : { view: 'dashboard', dashboardPage, labPage: 'models' };
}

/** 返回指定 Dashboard 页面对应的稳定路由。 */
export function dashboardPath(page: DashboardPage): string {
  return DASHBOARD_PATHS[page];
}

/** 返回指定模型工具页面对应的稳定路由。 */
export function labPath(page: LabPage): string {
  return LAB_PATHS[page];
}
