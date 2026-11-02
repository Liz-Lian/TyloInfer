import { useEffect, useRef, useState } from 'react';
import { Icon, ScreenHeader, SectionLabel, SettingRow, TMPanel } from '../../app/display';
import { errorMessage } from '../../app/formatters';
import {
  configureStudio,
  isStudio,
  saveStudioCredential,
  studioCredential,
  type StudioConfig,
} from '../../studio';
import { useRuntime } from '../../app/RuntimeProvider';
import { runtimeModelNames } from '../runtime/modelSelection';
import { runtimeApi } from '../../shared/api/resources/runtime';
import { getApiBaseUrl, getApiToken, setApiToken, setBrowserServiceUrl } from '../../shared/api/client';
import { useSettings } from '../settings/SettingsProvider';
import { RemoteRoutingPanel } from './RemoteRoutingPanel';
import { MemorySettingsPanel } from './MemorySettingsPanel';
import { ContextManagementPanel } from './ContextManagementPanel';
import { PrefixCacheSettingsPanel } from './PrefixCacheSettingsPanel';
import { ModelAliasMapping } from './ModelAliasMapping';
import { InferencePolicyPanel } from './InferencePolicyPanel';
import { ApiAuthenticationPanel } from './ApiAuthenticationPanel';
import { toast } from '../../stores/toastStore';
import { ServerConnectionPage } from '../../app/ServerConnectionPage';
import { serverCapabilities } from '../../app/serverCapabilities';
import '../../shared/styles/configuration-layout.css';

function browserConfig(): StudioConfig {
  const address = getApiBaseUrl() || window.location.origin;
  const url = new URL(address);
  return {
    mode: ['127.0.0.1', 'localhost', '[::1]'].includes(url.hostname) ? 'local' : 'remote',
    remote_url: address, local_service_port: Number(url.port) || (url.protocol === 'https:' ? 443 : 80),
  };
}

/** Configure supported authentication and connections without probing unavailable native settings. */
export function ConnectionsPage() {
  const { tr } = useSettings();
  const {
    runtime,
    models,
    instances,
    selectedModel,
    studio,
    reloadService,
  } = useRuntime();
  const capabilities = serverCapabilities(runtime);
  const managedLocal = isStudio() && studio?.connection === 'managed' && studio.config.mode === 'local';
  const [draft, setDraft] = useState<StudioConfig>(() => studio?.config ?? browserConfig());
  const [listeningPort, setListeningPort] = useState<number | null>(null);
  const [anthropicPort, setAnthropicPort] = useState<number | null>(null);
  const [anthropicDraft, setAnthropicDraft] = useState('');
  const [configurable, setConfigurable] = useState(false);
  const [token, setToken] = useState('');
  const [credentialWritable, setCredentialWritable] = useState(false);
  const [busy, setBusy] = useState(false);
  const [applyState, setApplyState] = useState<{ error: boolean; text: string } | null>(null);
  const [section, setSection] = useState('general');
  const [visited, setVisited] = useState(['general']);
  const applying = useRef(false);
  const applied = useRef(draft);
  const mounted = useRef(false);
  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; };
  }, []);
  useEffect(() => {
    if (studio) { setDraft(studio.config); applied.current = studio.config; }
    if (!capabilities.runtimeConfiguration) {
      setListeningPort(browserConfig().local_service_port);
      setConfigurable(managedLocal);
      return;
    }
    let disposed = false;
    void runtimeApi.runtimeListener().then((listener) => {
      if (!disposed) {
        if (!studio) {
          setDraft((current) => ({ ...current, local_service_port: listener.port }));
          applied.current = { ...applied.current, local_service_port: listener.port };
        }
        setListeningPort(listener.port);
        setConfigurable(listener.configurable);
        setAnthropicPort(listener.anthropic_port ?? null);
        setAnthropicDraft(listener.anthropic_port == null ? '' : String(listener.anthropic_port));
      }
    }).catch(() => {});
    return () => { disposed = true; };
  }, [studio, capabilities.runtimeConfiguration, managedLocal]);
  useEffect(() => {
    let disposed = false;
    void studioCredential()
      .then((value) => {
        if (!disposed) setToken(value ?? '');
      })
      .catch((cause) => {
        if (!disposed) {
          toast.error(errorMessage(cause, tr));
        }
      });
    return () => {
      disposed = true;
    };
  }, []);
  const active = Boolean(studio?.reachable ?? runtime);
  const modelNames = runtimeModelNames(models, instances);


  /** Save a connection or navigate to its web UI; never reconfigure a native listener in place. */
  async function apply(draft: StudioConfig) {
    if (applying.current) return;
    if (JSON.stringify(draft) === JSON.stringify(applied.current) && !credentialWritable && !applyState?.error) return;
    const endpoint = getApiBaseUrl();
    const restartsManagedService = !capabilities.runtimeConfiguration && managedLocal && draft.mode === 'local'
      && studio.config.local_service_port !== draft.local_service_port;
    let credential = getApiToken();
    let boundPort = listeningPort;
    const current = () => getApiBaseUrl() === endpoint && getApiToken() === credential;
    applying.current = true;
    setBusy(true);
    setApplyState({ error: false, text: tr('正在应用…', 'Applying…') });
    try {
      if (draft.mode === 'local' && (!Number.isInteger(draft.local_service_port) || draft.local_service_port < 1 || draft.local_service_port > 65535)) {
        throw new Error(tr('端口必须为 1–65535 的整数', 'Port must be an integer between 1 and 65535'));
      }
      if (draft.mode === 'local' && draft.local_service_port === anthropicPort && draft.local_service_port !== listeningPort) {
        throw new Error(tr('OpenAI 和 Anthropic 不能使用同一个端口，请为它们设置不同的端口。', 'OpenAI and Anthropic cannot share a port. Choose a different port for each API.'));
      }
      if (isStudio()) {
        if (capabilities.runtimeConfiguration && studio?.connection === 'managed' && studio.config.mode === 'local' && draft.mode === 'local'
            && studio.config.local_service_port !== draft.local_service_port) {
          boundPort = (await runtimeApi.configureRuntimeListener(draft.local_service_port)).port;
          if (!current()) return;
        }
        await configureStudio(draft);
        if (!current()) return;
        if (credentialWritable) {
          await saveStudioCredential(token);
          if (!current()) return;
        }
      } else {
        let address = draft.remote_url.trim().replace(/\/+$/, '').replace(/\/v1$/, '');
        if (draft.mode === 'local') {
          if (capabilities.runtimeConfiguration && browserConfig().mode === 'local') boundPort = (await runtimeApi.configureRuntimeListener(draft.local_service_port)).port;
          if (!current()) return;
          address = `http://127.0.0.1:${draft.local_service_port}`;
        } else {
          let parsed: URL;
          try { parsed = new URL(address); }
          catch { throw new Error(tr('请输入完整的 HTTP 或 HTTPS 服务地址，例如 http://127.0.0.1:8090。', 'Enter a full HTTP or HTTPS service URL, such as http://127.0.0.1:8090.')); }
          if (!['http:', 'https:'].includes(parsed.protocol) || parsed.username || parsed.password) {
            throw new Error(tr('请输入不含凭据的 HTTP 或 HTTPS 服务地址', 'Enter an HTTP or HTTPS service URL without credentials'));
          }
        }
        if (!capabilities.runtimeConfiguration) {
          const page = new URL(address);
          if (page.pathname !== '/') {
            throw new Error(tr('网页连接仅支持服务根地址（可带 /v1），当前不支持路径前缀。', 'Web connections require a server root URL (optionally ending in /v1). Path prefixes are not supported.'));
          }
          if (page.origin !== window.location.origin) {
            page.pathname = window.location.pathname;
            window.location.assign(page.toString());
          } else {
            applied.current = draft;
            setApplyState(null);
          }
          return;
        }
        setBrowserServiceUrl(address);
        if (credentialWritable) { setApiToken(token); credential = token.trim(); }
        if (draft.mode === 'local' && window.location.port === String(listeningPort)
            && listeningPort !== draft.local_service_port) {
          const page = new URL(window.location.href);
          page.port = String(draft.local_service_port);
          window.location.assign(page.toString());
          return;
        }
      }
      if (mounted.current && draft.mode === 'local') setListeningPort(draft.local_service_port);
      const reconnected = await reloadService();
      applied.current = draft;
      if (mounted.current && reconnected) {
        setCredentialWritable(false);
        setApplyState({ error: false, text: restartsManagedService
          ? tr('服务已按新端口重启，请在模型页面确认加载状态。', 'The service restarted on the new port. Check model loading status in Models.')
          : tr('已应用，无需重载模型', 'Applied without reloading models') });
      } else if (mounted.current) {
        setApplyState({ error: true, text: tr('设置已应用，但连接失败，请检查地址和凭据', 'Settings applied, but reconnection failed. Check the address and credentials.') });
      }
    } catch (cause) {
      if (mounted.current && current()) {
        setDraft({ ...applied.current, local_service_port: boundPort ?? applied.current.local_service_port });
        setListeningPort(boundPort);
        setApplyState({ error: true, text: errorMessage(cause, tr) });
      }
    } finally {
      applying.current = false;
      if (mounted.current) setBusy(false);
    }
  }

  async function applyAnthropic() {
    const port = Number(anthropicDraft);
    if (applying.current || port === anthropicPort) return;
    if (!Number.isInteger(port) || port < 1 || port > 65535) {
      setAnthropicDraft(anthropicPort == null ? '' : String(anthropicPort));
      setApplyState({ error: true, text: tr('端口必须为 1–65535 的整数', 'Port must be an integer between 1 and 65535') });
      return;
    }
    if (port === listeningPort) {
      setAnthropicDraft(anthropicPort == null ? '' : String(anthropicPort));
      setApplyState({ error: true, text: tr('OpenAI 和 Anthropic 不能使用同一个端口，请为它们设置不同的端口。', 'OpenAI and Anthropic cannot share a port. Choose a different port for each API.') });
      return;
    }
    const endpoint = getApiBaseUrl();
    const credential = getApiToken();
    applying.current = true;
    setBusy(true);
    setApplyState({ error: false, text: tr('正在应用…', 'Applying…') });
    try {
      const listener = await runtimeApi.configureRuntimeListener(port, 'anthropic');
      if (mounted.current && endpoint === getApiBaseUrl() && credential === getApiToken()) {
        setAnthropicPort(listener.anthropic_port ?? port);
        setAnthropicDraft(String(listener.anthropic_port ?? port));
        setApplyState({ error: false, text: tr('已应用，无需重载模型', 'Applied without reloading models') });
      }
    } catch (cause) {
      if (mounted.current && endpoint === getApiBaseUrl() && credential === getApiToken()) {
        setAnthropicDraft(anthropicPort == null ? '' : String(anthropicPort));
        setApplyState({ error: true, text: errorMessage(cause, tr) });
      }
    } finally {
      applying.current = false;
      if (mounted.current) setBusy(false);
    }
  }


  return (
    <section className="dashboard-view service-view">
      <ScreenHeader
        title={tr('服务', 'Service')}
        subtitle={tr(
          '运行服务、连接与模型默认值。',
          'Runtime service, connections, and model defaults.',
        )}
      />
      <div className="configuration-tabs" role="tablist" aria-label={tr('服务设置分类', 'Service settings sections')}>
        {[
          ['general', tr('常用', 'General')], ['cache', tr('缓存', 'Cache')], ['advanced', tr('高级', 'Advanced')],
        ].map(([value, label], index, sections) => <button key={value} type="button" role="tab"
          id={`service-tab-${value}`} aria-selected={section === value} aria-controls={`service-section-${value}`}
          tabIndex={section === value ? 0 : -1}
          onClick={() => { setSection(value); setVisited(current => current.includes(value) ? current : [...current, value]); }}
          onKeyDown={event => {
            const next = event.key === 'ArrowRight' ? (index + 1) % sections.length
              : event.key === 'ArrowLeft' ? (index + sections.length - 1) % sections.length
                : event.key === 'Home' ? 0 : event.key === 'End' ? sections.length - 1 : null;
            if (next == null) return;
            event.preventDefault();
            const target = sections[next][0];
            setSection(target); setVisited(current => current.includes(target) ? current : [...current, target]);
            document.getElementById(`service-tab-${target}`)?.focus();
          }}>{label}</button>)}
      </div>
      <div className="server-page" role="tabpanel" id="service-section-general" aria-labelledby="service-tab-general" hidden={section !== 'general'}>
        {(studio?.connection === 'confirmation_required' || studio?.connection === 'incompatible') && <ServerConnectionPage />}
        {active && (
          <div className="server-active-notice">
            <Icon name="info" size={15} />
            <span>
              {tr(
                capabilities.runtimeConfiguration ? '服务器正在运行；网络设置失焦或按回车自动应用，无需重载模型。' : managedLocal ? '端口变更由桌面启动器应用，会重启此 App 管理的服务。模型别名与鉴权可直接修改。' : '原生服务的监听端口由启动参数决定。当前可修改模型别名与 API 鉴权。',
                capabilities.runtimeConfiguration ? 'The server is active. Network settings apply on blur or Enter without reloading models.' : managedLocal ? 'Changing the port restarts the service managed by this app. Model aliases and authentication can be changed directly.' : 'Native listener ports are set at startup. Model aliases and API authentication can be changed here.',
              )}
            </span>
          </div>
        )}
        <SectionLabel title={tr('API参数', 'API parameters')} />
        <TMPanel className="server-settings-panel server-api-panel">
          <div className="setting-list">
            {studio?.reachable && <SettingRow title={tr('服务归属', 'Server ownership')}
              detail={studio.connection === 'managed'
                ? tr('由 App 管理；退出 App 时停止服务。', 'Managed by this app; stopped when the app quits.')
                : tr('独立服务；退出 App 不会停止它。', 'Independent server; quitting the app leaves it running.')}
              trailing={<span>{studio.server && `TyloInfer Server ${studio.server.version} · PID ${studio.server.pid}`}</span>} />}
            <SettingRow
              title={tr('模型 ID', 'Model ID')}
              detail={tr(
                '由 /v1/models 公布，并用于对话补全请求。',
                'Advertised by /v1/models and accepted by chat completions.',
              )}
              trailing={
                <div className="server-row-actions server-model-control">
                  <ModelAliasMapping models={modelNames} selectedModel={selectedModel} />
                </div>
              }
            />
            <SettingRow
              title={tr('服务连接', 'Service connection')}
              detail={tr(
                '本地模式仅监听 127.0.0.1；远程模式连接另一台 TyloInfer Server。',
                'Local mode stays on 127.0.0.1; remote mode connects to another TyloInfer Server.',
              )}
              trailing={
                <select
                  aria-label={tr('服务连接', 'Service connection')}
                  disabled={busy || !draft}
                  onChange={(event) => {
                    const next = { ...draft, mode: event.target.value as StudioConfig['mode'] };
                    setDraft(next);
                    void apply(next);
                  }}
                  value={draft?.mode ?? 'local'}
                >
                  <option value="local">
                    {tr('仅本机 · 127.0.0.1', 'Local only · 127.0.0.1')}
                  </option>
                  <option value="remote">{tr('远程 TyloInfer Server', 'Remote TyloInfer Server')}</option>
                </select>
              }
            />
            {draft?.mode === 'remote' ? (
              <>
                <SettingRow
                  title={tr('远程端点', 'Remote endpoint')}
                  detail={tr(
                    !isStudio() && !capabilities.runtimeConfiguration ? '打开远程服务根地址，不支持路径前缀；如需鉴权，请在新页面登录。' : '远程 TyloInfer Server 的 OpenAI 兼容基础 URL。',
                    !isStudio() && !capabilities.runtimeConfiguration ? 'Open the remote server root URL; path prefixes are unsupported. Sign in on the destination page if required.' : 'OpenAI-compatible base URL for the remote TyloInfer Server.',
                  )}
                  trailing={
                    <input
                      aria-label={tr('远程端点', 'Remote endpoint')}
                      className="server-wide-input"
                      disabled={busy}
                      onBlur={() => void apply(draft)}
                      onKeyDown={(event) => { if (event.key === 'Enter') { event.preventDefault(); void apply(draft); } }}
                      onChange={(event) =>
                        setDraft(
                          (current) => current && { ...current, remote_url: event.target.value },
                        )
                      }
                      placeholder="https://host:port"
                      type="url"
                      value={draft.remote_url}
                    />
                  }
                />
                {(isStudio() || capabilities.runtimeConfiguration) && <SettingRow
                  title={tr('API 密钥', 'API key')}
                  detail={tr(
                    isStudio() ? '凭据只保存在系统凭据库中。' : '凭据仅保留在当前页面内存中。',
                    isStudio() ? 'The credential is stored only in the system credential vault.' : 'The credential stays only in this page’s memory.',
                  )}
                  trailing={
                    <input
                      aria-label={tr('API 密钥', 'API key')}
                      autoComplete="off"
                      className="server-wide-input"
                      disabled={busy}
                      onBlur={() => void apply(draft)}
                      onKeyDown={(event) => { if (event.key === 'Enter') { event.preventDefault(); void apply(draft); } }}
                      onChange={(event) => {
                        setToken(event.target.value);
                        setCredentialWritable(true);
                      }}
                      placeholder={tr('可选', 'Optional')}
                      type="password"
                      value={token}
                    />
                  }
                />}
              </>
            ) : (
              <>
              <SettingRow
                title={tr('OpenAI 端口', 'OpenAI port')}
                detail={listeningPort == null ? tr('OpenAI 兼容 HTTP 服务端口。', 'OpenAI-compatible HTTP server port.')
                  : `http://127.0.0.1:${listeningPort}/v1`}
                trailing={
                  <input
                    aria-label={tr('OpenAI 端口', 'OpenAI port')}
                    className="server-number-input"
                    disabled={busy || !configurable}
                    max={65535}
                    min={1}
                    onBlur={() => void apply(draft)}
                    onKeyDown={(event) => { if (event.key === 'Enter') { event.preventDefault(); void apply(draft); } }}
                    onChange={(event) =>
                      setDraft(
                        (current) =>
                          current && { ...current, local_service_port: Number(event.target.value) },
                      )
                    }
                    type="number"
                    value={draft?.local_service_port ?? 8090}
                  />
                }
              />
              <SettingRow
                title={tr('Anthropic 端口', 'Anthropic port')}
                detail={anthropicPort == null
                  ? capabilities.runtimeConfiguration ? tr('当前服务尚未启用 Anthropic API。', 'The current service has not enabled the Anthropic API.') : tr('与 OpenAI 共用服务端口 /v1/messages；不能单独配置。', 'Uses the same server port at /v1/messages; a separate port cannot be configured.')
                  : `http://127.0.0.1:${anthropicPort}/v1/messages`}
                trailing={<input
                  aria-label={tr('Anthropic 端口', 'Anthropic port')}
                  className="server-number-input"
                  disabled={busy || !configurable || anthropicPort == null}
                  max={65535} min={1} type="number" value={anthropicDraft}
                  onChange={(event) => setAnthropicDraft(event.target.value)}
                  onBlur={() => void applyAnthropic()}
                  onKeyDown={(event) => { if (event.key === 'Enter') { event.preventDefault(); void applyAnthropic(); } }}
                />}
              />
              </>
            )}
            <ApiAuthenticationPanel />
          </div>
          {applyState && <p className="server-apply-state" role={applyState.error ? 'alert' : 'status'}>{applyState.text}</p>}
        </TMPanel>
        {capabilities.runtimeConfiguration ? <><MemorySettingsPanel /><InferencePolicyPanel /><ContextManagementPanel /></> :
          <p role="status">{tr('内存、推理和全局上下文设置暂不可用。可在模型设置中配置下次加载的上下文大小与驻留策略。', 'Memory, inference and global context settings are unavailable. Configure context size and residency for the next load in Models.')}</p>}
      </div>
      <div className="server-page" role="tabpanel" id="service-section-cache" aria-labelledby="service-tab-cache" hidden={section !== 'cache'}>
        {capabilities.runtimeConfiguration ? visited.includes('cache') && <PrefixCacheSettingsPanel /> : <p role="status">{tr('当前原生服务暂不支持缓存设置。', 'Cache settings are unavailable on the native server.')}</p>}
      </div>
      <div className="server-page" role="tabpanel" id="service-section-advanced" aria-labelledby="service-tab-advanced" hidden={section !== 'advanced'}>
        {capabilities.runtimeConfiguration ? visited.includes('advanced') && <RemoteRoutingPanel /> : <p role="status">{tr('当前原生服务暂不支持集群节点与远程路由。', 'Cluster nodes and remote routing are unavailable on the native server.')}</p>}
      </div>
    </section>
  );
}
