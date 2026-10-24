



export type ViewName = 'chat' | 'dashboard' | 'lab' | 'applications' | 'versions';
export type DashboardPage = 'overview' | 'models' | 'connections' | 'cache' | 'logs' | 'settings';
export type LabPage = 'models' | 'evaluations' | 'analysis';

export interface StudioLocation {
  view: ViewName;
  dashboardPage: DashboardPage;
  labPage: LabPage;
}

export const STUDIO_PATHS = {
  overview: '/',
  chat: '/chat',
  models: '/models',
  server: '/runtime',
  resources: '/resources',
  modelHub: '/model-hub',
  evaluations: '/evaluations',
  analysis: '/analysis',
  applications: '/applications',
  versions: '/versions',
  logs: '/logs',
  settings: '/settings',
} as const;

const DASHBOARD_PATHS: Record<DashboardPage, string> = {
  overview: STUDIO_PATHS.overview,
  models: STUDIO_PATHS.models,
  connections: STUDIO_PATHS.server,
  cache: STUDIO_PATHS.resources,
  logs: STUDIO_PATHS.logs,
  settings: STUDIO_PATHS.settings,
};

const LAB_PATHS: Record<LabPage, string> = {
  models: STUDIO_PATHS.modelHub,
  evaluations: STUDIO_PATHS.evaluations,
  analysis: STUDIO_PATHS.analysis,
};


export function normalizeStudioPath(pathname: string): string {
  return pathname.length > 1 ? pathname.replace(/\/+$/, '') : pathname;
}


export function isStudioPath(pathname: string): boolean {
  return Object.values(STUDIO_PATHS).some((path) => path === normalizeStudioPath(pathname));
}


export function resolveStudioLocation(pathname: string): StudioLocation {
  const normalizedPath = normalizeStudioPath(pathname);
  if (normalizedPath === STUDIO_PATHS.versions) {
    return { view: 'versions', dashboardPage: 'overview', labPage: 'models' };
  }
  if (normalizedPath === STUDIO_PATHS.applications) {
    return { view: 'applications', dashboardPage: 'overview', labPage: 'models' };
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


export function dashboardPath(page: DashboardPage): string {
  return DASHBOARD_PATHS[page];
}


export function labPath(page: LabPage): string {
  return LAB_PATHS[page];
}
