
import { Component, Suspense, useEffect, type ReactNode } from 'react';
import { useLocation, Outlet } from 'react-router';
import * as TooltipPrimitive from '@radix-ui/react-tooltip';
import { useRuntime } from './RuntimeProvider';
import { useSettings } from '../features/settings/SettingsProvider';
import { ToastContainer } from '../shared/ui/Toast';
import { FailurePage } from './FailurePage';
import { LoadingPage } from './LoadingPage';
import { LoginPage } from './LoginPage';
import { StudioSidebar } from './StudioSidebar';
import { RuntimeAlerts } from './RuntimeAlerts';
import { ServerConnectionPage } from './ServerConnectionPage';
import { resolveStudioLocation, isStudioPath } from '../navigation';


class PageErrorBoundary extends Component<{ children: ReactNode }, { detail: string | null }> {
  state = { detail: null as string | null };
  static getDerivedStateFromError(error: unknown) {
    return { detail: error instanceof Error ? error.message : String(error) };
  }
  render() {
    if (this.state.detail !== null)
      return (
        <FailurePage detail={this.state.detail} kind="render"
          onRetry={() => this.setState({ detail: null })} />
      );
    return this.props.children;
  }
}


export function StudioShell() {
  const location = useLocation();
  const { tr } = useSettings();
  const { connectionError, refreshError, jobStreamErrors, ready, reloadService, connectionRevision, authenticationRequired, studio } = useRuntime();
  const currentLocation = resolveStudioLocation(location.pathname);
  const view = isStudioPath(location.pathname) ? currentLocation.view : 'not-found';
  const pageAvailable = ready || view === 'not-found' ||
    location.pathname === '/runtime' || location.pathname === '/settings' || view === 'versions';
  const connectionProblem = ready ? connectionError : connectionError ?? refreshError;
  const hasAlerts = pageAvailable && Boolean(
    connectionProblem || (ready && (refreshError || Object.keys(jobStreamErrors).length)),
  );
  useEffect(() => {
    const title = location.pathname === '/'
      ? 'Overview' : location.pathname.slice(1).replaceAll('-', ' ');
    document.title = `${authenticationRequired ? tr('登录', 'Sign in') : title} · TyloInfer`;
  }, [location.pathname, authenticationRequired, tr]);

  if (authenticationRequired) return <LoginPage />;

  return (
    <TooltipPrimitive.Provider delayDuration={300}>
      <div className="app-shell">
        <ToastContainer />
        <a className="skip-link" href="#studio-main">
          {tr('跳到主要内容', 'Skip to main content')}
        </a>
        <StudioSidebar />
        <main className={`${view === 'chat' ? 'workspace chat-workspace' : 'workspace'}${hasAlerts ? ' has-runtime-alerts' : ''}`}
          id="studio-main" tabIndex={-1}>
          {hasAlerts && <RuntimeAlerts connectionProblem={connectionProblem} />}
          {pageAvailable ? (
            <PageErrorBoundary key={`${location.pathname}:${connectionRevision}`}>
              <Suspense fallback={<LoadingPage />}>
                <Outlet />
              </Suspense>
            </PageErrorBoundary>
          ) : studio?.connection === 'confirmation_required' || studio?.connection === 'incompatible' ? (
            <ServerConnectionPage />
          ) : connectionProblem ? (
            <FailurePage detail={connectionProblem} kind="connection"
              onRetry={() => void reloadService()} />
          ) : <LoadingPage />}
        </main>
      </div>
    </TooltipPrimitive.Provider>
  );
}
