import { lazy, Suspense } from 'react';
import { Route, Routes } from 'react-router';
import { StudioUpdateProvider } from './features/settings/UpdateManager';
import { RuntimeProvider } from './app/RuntimeProvider';
import { StudioShell } from './app/StudioShell';
import { LabLayout } from './app/LabLayout';
import { LoadingPage } from './app/LoadingPage';
import { NotFoundPage } from './app/NotFoundPage';
import './shared/ui/primitives.css';

const OverviewPage = lazy(() =>
  import('./features/runtime/OverviewPage').then((module) => ({ default: module.OverviewPage })),
);
const ModelsPage = lazy(() =>
  import('./features/models/ModelsPage').then((module) => ({ default: module.ModelsPage })),
);
const ConnectionsPage = lazy(() =>
  import('./features/connections/ConnectionsPage').then((module) => ({
    default: module.ConnectionsPage,
  })),
);
const CachePage = lazy(() =>
  import('./features/runtime/CachePage').then((module) => ({ default: module.CachePage })),
);
const LogsPage = lazy(() =>
  import('./features/runtime/LogsPage').then((module) => ({ default: module.LogsPage })),
);
const ModelHubPage = lazy(() =>
  import('./features/models/ModelHubPage').then((module) => ({ default: module.ModelHubPage })),
);
const AnalysisPage = lazy(() =>
  import('./features/analysis/AnalysisPage').then((module) => ({ default: module.AnalysisPage })),
);
const VersionsPage = lazy(() =>
  import('./features/settings/VersionsPage').then((module) => ({ default: module.VersionsPage })),
);

export default function App() {
  return (
    <StudioUpdateProvider>
      <RuntimeProvider>
        <Suspense
          fallback={
            <main className="fatal-workspace">
              <LoadingPage />
            </main>
          }
        >
          <Routes>
            <Route element={<StudioShell />}>
              <Route index element={<OverviewPage />} />
              <Route path="models" element={<ModelsPage />} />
              <Route path="runtime" element={<ConnectionsPage />} />
              <Route path="resources" element={<CachePage />} />
              <Route path="logs" element={<LogsPage />} />
              <Route path="versions" element={<VersionsPage />} />
              <Route element={<LabLayout />}>
                <Route path="analysis" element={<AnalysisPage />} />
                <Route path="model-hub" element={<ModelHubPage />} />
              </Route>
              <Route path="*" element={<NotFoundPage />} />
            </Route>
          </Routes>
        </Suspense>
      </RuntimeProvider>
    </StudioUpdateProvider>
  );
}
