import { useEffect, useMemo, useRef, useState, type FormEvent, type ReactNode } from 'react';
import { ArrowDownIcon, ArrowSquareOutIcon, ArrowUpIcon, CubeIcon, DownloadSimpleIcon, HeartIcon, MagnifyingGlassIcon, XIcon } from '@phosphor-icons/react';
import DOMPurify from 'dompurify';
import { marked } from 'marked';
import { modelsApi } from '../../shared/api/resources/models';
import { ApiError } from '../../shared/api/client';
import type { HubModelInfo, HubModelSummary, HubModelVariant } from '../../shared/api/types';
import { openStudioExternal } from '../../shared/platform/studio';
import { ModelVendorMark } from '../../app/ModelVendorMark';
import { parseHubReference } from './hubReference';
import { downloadSize as size } from './repositoryFileGroups';
import './community-model-browser.css';

type Translate = (zh: string, en: string) => string;
type Format = 'tq' | 'gguf' | 'mlx';
type Sort = 'relevance' | 'downloads' | 'likes' | 'updated';
const allFormats: Format[] = ['tq', 'gguf', 'mlx'];
const count = (value: number) => new Intl.NumberFormat(undefined, { notation: 'compact' }).format(value);

function ModelCard({ model, tr }: { model: HubModelInfo; tr: Translate }) {
  const [content, setContent] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [failed, setFailed] = useState(false);
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setFailed(false);
    void modelsApi.hubModelReadme(model, controller.signal).then(result => {
      if (!controller.signal.aborted) setContent(result.content);
    }).catch(() => {
      if (!controller.signal.aborted) setFailed(true);
    }).finally(() => {
      if (!controller.signal.aborted) setLoading(false);
    });
    return () => controller.abort();
  }, [model.provider, model.repo_id, model.revision, attempt]);
  const html = useMemo(() => {
    if (!content) return '';
    const markdown = content.replace(/^---\r?\n[\s\S]*?\r?\n---(?:\r?\n|$)/, '');
    const clean = DOMPurify.sanitize(marked.parse(markdown, { async: false }), {
      FORBID_TAGS: ['style', 'iframe', 'form', 'input', 'button', 'video', 'audio'],
      FORBID_ATTR: ['style', 'id', 'name', 'srcset'],
    });
    const document = new DOMParser().parseFromString(clean, 'text/html');
    const repository = model.provider === 'huggingface' ? `https://huggingface.co/${model.repo_id}` : `https://modelscope.cn/models/${model.repo_id}`;
    for (const element of document.querySelectorAll('a[href], img[src]')) {
      const attribute = element.tagName === 'IMG' ? 'src' : 'href';
      const base = `${repository}/${attribute === 'src' ? 'resolve' : 'blob'}/${encodeURIComponent(model.revision)}/`;
      try {
        const url = new URL(element.getAttribute(attribute)!, base);
        if (!['http:', 'https:'].includes(url.protocol) || attribute === 'src' && url.protocol !== 'https:') { element.removeAttribute(attribute); continue; }
        element.setAttribute(attribute, url.href);
        if (attribute === 'src') {
          element.setAttribute('loading', 'lazy');
          element.setAttribute('referrerpolicy', 'no-referrer');
          element.removeAttribute('srcset');
        }
      } catch { element.removeAttribute(attribute); }
    }
    return document.body.innerHTML;
  }, [content, model.provider, model.repo_id, model.revision]);
  const markup = useMemo(() => ({ __html: html }), [html]);
  if (loading) return <div className="community-placeholder" role="status">{tr('正在读取模型卡…', 'Loading model card…')}</div>;
  if (failed) return <div className="community-placeholder" role="alert"><p>{tr('暂时无法读取模型卡，请重试或返回概览下载模型。', 'The model card is unavailable. Retry or return to Overview to download.')}</p><button type="button" onClick={() => setAttempt(value => value + 1)}>{tr('重试', 'Retry')}</button></div>;
  if (!html) return <div className="community-placeholder">{tr('此仓库未提供模型卡。', 'This repository has no model card.')}</div>;
  return <div className="community-model-card" onClick={event => {
    const link = (event.target as HTMLElement).closest('a[href]');
    if (!link) return;
    event.preventDefault();
    void openStudioExternal(link.getAttribute('href')!).catch(() => {});
  }} dangerouslySetInnerHTML={markup} />;
}

export function CommunityModelBrowser({ model, onSelect, variants, details, files, renderDownload, tr, backend }: {
  model: HubModelInfo | null;
  onSelect(model: HubModelInfo | null): void;
  variants: HubModelVariant[];
  details: ReactNode;
  files: ReactNode;
  renderDownload(variant: HubModelVariant): ReactNode;
  tr: Translate;
  backend?: string;
}) {
  const [provider, setProvider] = useState<HubModelSummary['provider']>('huggingface');
  const [query, setQuery] = useState('');
  const [formats, setFormats] = useState<Format[]>(() => backend === 'cuda' || backend === 'hip' ? ['tq', 'gguf'] : allFormats);
  const [sort, setSort] = useState<Sort>('relevance');
  const [direction, setDirection] = useState<'asc' | 'desc'>('desc');
  const [nextCursor, setNextCursor] = useState<string | null>(null);
  const [loadingMore, setLoadingMore] = useState(false);
  const [results, setResults] = useState<HubModelSummary[]>([]);
  const [searching, setSearching] = useState(false);
  const [inspecting, setInspecting] = useState(false);
  const [selected, setSelected] = useState('');
  const [variantId, setVariantId] = useState('');
  const [view, setView] = useState<'overview' | 'card' | 'files'>('overview');
  const [openedCard, setOpenedCard] = useState('');
  const [searchError, setSearchError] = useState<'network' | 'expired' | null>(null);
  const [detailError, setDetailError] = useState(false);
  const searchController = useRef<AbortController | null>(null);
  const detailController = useRef<AbortController | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const formatChosen = useRef(false);
  const callbacks = useRef({ onSelect });
  callbacks.current = { onSelect };

  useEffect(() => {
    if (!formatChosen.current) setFormats(backend === 'cuda' || backend === 'hip' ? ['tq', 'gguf'] : allFormats);
  }, [backend]);

  async function inspect(item: HubModelSummary) {
    detailController.current?.abort();
    const controller = new AbortController();
    detailController.current = controller;
    setSelected(`${item.provider}:${item.repo_id}`);
    setInspecting(true);
    setDetailError(false);
    setView('overview');
    setVariantId('');
    callbacks.current.onSelect(null);
    try {
      const info = await modelsApi.hubModelInfo(item.provider, item.repo_id, undefined, controller.signal);
      if (!controller.signal.aborted) callbacks.current.onSelect(info);
    } catch {
      if (!controller.signal.aborted) setDetailError(true);
    } finally {
      if (!controller.signal.aborted) setInspecting(false);
    }
  }

  async function search(append = false) {
    if (append && (!nextCursor || loadingMore)) return;
    if (timer.current) clearTimeout(timer.current);
    searchController.current?.abort();
    if (!append) detailController.current?.abort();
    const controller = new AbortController();
    searchController.current = controller;
    setSearchError(null);
    if (append) setLoadingMore(true);
    else {
      setSearching(true);
      setLoadingMore(false);
      setNextCursor(null);
      setDetailError(false);
      setInspecting(false);
      setSelected('');
      setResults([]);
      callbacks.current.onSelect(null);
    }
    try {
      const reference = query.trim();
      if (parseHubReference(reference, provider) || /^https?:\/\//i.test(reference)) {
        const info = await modelsApi.resolveHubModel(reference, provider, controller.signal);
        if (controller.signal.aborted) return;
        setResults([info]);
        setSelected(`${info.provider}:${info.repo_id}`);
        setView('overview');
        setVariantId('');
        callbacks.current.onSelect(info);
      } else {
        if (!formats.length) return;
        const found = await modelsApi.browseHubModels(provider, reference, { formats, sort, direction,
          cursor: append ? nextCursor : undefined, signal: controller.signal });
        if (controller.signal.aborted) return;
        setResults(current => {
          const existing = append ? current : [];
          const seen = new Set(existing.map(item => `${item.provider}:${item.repo_id}`));
          return [...existing, ...found.data.filter(item => !seen.has(`${item.provider}:${item.repo_id}`))];
        });
        setNextCursor(found.next_cursor ?? null);
        if (!append && found.data[0]) void inspect(found.data[0]);
      }
    } catch (cause) {
      if (!controller.signal.aborted) setSearchError(cause instanceof ApiError && cause.code === 'search_cursor_expired' ? 'expired' : 'network');
    } finally {
      if (!controller.signal.aborted) { setSearching(false); setLoadingMore(false); }
    }
  }

  useEffect(() => {
    searchController.current?.abort();
    detailController.current?.abort();
    timer.current = setTimeout(() => void search(), query.trim() ? 350 : 0);
    return () => {
      if (timer.current) clearTimeout(timer.current);
      searchController.current?.abort();
      detailController.current?.abort();
    };
  }, [provider, query, formats.join(','), sort, direction]);

  function submit(event: FormEvent) { event.preventDefault(); void search(); }
  const variant = variants.find(item => item.id === variantId) ?? variants[0];
  const modelKey = model ? `${model.provider}:${model.repo_id}:${model.revision}` : '';
  const selectedItem = results.find(item => `${item.provider}:${item.repo_id}` === selected);
  return <div className="community-browser">
    <form className="community-search-toolbar" onSubmit={submit}>
      <label className="community-search-field"><MagnifyingGlassIcon size={18} />
        <input aria-label={tr('搜索模型', 'Search models')} placeholder={tr('模型名称、owner/repo 或仓库链接', 'Model name, owner/repo, or repository URL')}
          value={query} onChange={event => setQuery(event.target.value)} />
        {query && <button type="button" aria-label={tr('清空搜索', 'Clear search')} onClick={() => setQuery('')}><XIcon size={15} /></button>}
      </label>
      <select aria-label={tr('模型来源', 'Model source')} value={provider} onChange={event => {
        setProvider(event.target.value as typeof provider);
        if (event.target.value === 'modelscope') setDirection('desc');
      }}><option value="huggingface">Hugging Face</option><option value="modelscope">ModelScope</option></select>
      <button className="community-search-submit" type="submit">{tr('查找', 'Search')}</button>
    </form>
    <div className="community-browser-body">
      <section className="community-library" aria-label={tr('模型搜索结果', 'Model search results')}>
        <div className="community-library-toolbar">
          <div className="community-format-filters" role="group" aria-label={tr('权重格式', 'Weight format')}>
            <button type="button" aria-pressed={formats.length === allFormats.length} onClick={() => { formatChosen.current = true; setFormats(allFormats); }}>{tr('全部', 'All')}</button>
            {allFormats.map(item => <label key={item}><input type="checkbox" aria-label={item.toUpperCase()} checked={formats.includes(item)}
              onChange={event => { formatChosen.current = true; setFormats(current => allFormats.filter(value => value === item ? event.target.checked : current.includes(value))); }} />{item.toUpperCase()}</label>)}
          </div>
          <div className="community-result-order"><span role="status">{searching ? tr('正在查找…', 'Searching…') : !query.trim() && sort === 'relevance' ? tr('热门推荐', 'Recommended') : `${results.length} ${tr('个结果', 'results')}`}</span>
            <select aria-label={tr('模型排序', 'Sort models')} value={sort} onChange={event => setSort(event.target.value as Sort)}>
              <option value="relevance">{tr('最佳匹配', 'Best match')}</option><option value="downloads">{tr('下载量', 'Downloads')}</option><option value="likes">{tr('收藏量', 'Likes')}</option><option value="updated">{tr('更新时间', 'Last updated')}</option>
            </select>
            <button className="community-sort-direction" type="button" disabled={provider === 'modelscope'}
              aria-label={direction === 'desc' ? tr('降序', 'Descending') : tr('升序', 'Ascending')}
              title={provider === 'modelscope' ? tr('ModelScope 仅提供降序搜索。', 'ModelScope only provides descending search.') : direction === 'desc' ? tr('切换为升序', 'Switch to ascending') : tr('切换为降序', 'Switch to descending')}
              onClick={() => setDirection(current => current === 'desc' ? 'asc' : 'desc')}>{direction === 'desc' ? <ArrowDownIcon size={15} /> : <ArrowUpIcon size={15} />}</button>
          </div>
        </div>
        <div className="community-result-list" aria-busy={searching}>
          {results.map(item => <button key={`${item.provider}:${item.repo_id}`} type="button" className="community-result" aria-pressed={selected === `${item.provider}:${item.repo_id}`}
            onClick={() => void inspect(item)}>
            <div className="community-result-heading"><strong title={item.repo_id}>{item.repo_id.split('/').pop()}</strong><ModelVendorMark name={item.repo_id} size={21} /></div>
            <span className="community-result-author">{item.author || item.repo_id.split('/')[0]}</span>
            {item.description && <p>{item.description}</p>}
            <div className="community-result-meta"><span><DownloadSimpleIcon size={12} />{count(item.downloads)}</span><span><HeartIcon size={12} />{count(item.likes)}</span>
              <span className="community-result-formats">{item.formats?.map(value => <i key={value}>{value.toUpperCase()}</i>)}</span>
            </div>
          </button>)}
          {searchError ? <div className="community-placeholder" role="alert"><p>{searchError === 'expired' ? tr('搜索已过期，请重新搜索。', 'This search expired. Start a new search.') : tr('暂时无法连接模型来源，请稍后重试或切换来源。', 'Cannot reach the model source. Retry or choose another source.')}</p><button type="button" onClick={() => void search()}>{searchError === 'expired' ? tr('重新搜索', 'Search again') : tr('重试', 'Retry')}</button></div>
            : !results.length && <div className="community-placeholder"><MagnifyingGlassIcon size={28} /><p>{!formats.length ? tr('请至少选择一种权重格式。', 'Select at least one weight format.') : searching ? tr('正在读取模型列表…', 'Loading models…') : tr('没有找到模型，试试其它关键词或格式。', 'No models found. Try another keyword or format.')}</p></div>}
          {!searching && nextCursor && !searchError && <button className="community-load-more" type="button" disabled={loadingMore} onClick={() => void search(true)}>{loadingMore ? tr('加载中…', 'Loading…') : tr('显示更多', 'Show more')}</button>}
        </div>
      </section>
      <section className="community-inspector" aria-label={tr('模型详情', 'Model details')} aria-busy={inspecting}>
        {model ? <>
          <header className="community-inspector-header"><div><small>{model.author || model.repo_id.split('/')[0]}</small><h3>{model.repo_id.split('/').pop()}</h3></div>
            <ModelVendorMark name={model.repo_id} architecture={model.architectures} size={30} />
            <button type="button" className="community-external" aria-label={tr('打开模型卡', 'Open model card')} onClick={() => void openStudioExternal(model.source_url || `${model.provider === 'huggingface' ? 'https://huggingface.co/' : 'https://modelscope.cn/models/'}${model.repo_id}`).catch(() => {})}><ArrowSquareOutIcon size={17} /></button>
          </header>
          <div className="community-inspector-tabs" role="tablist" aria-label={tr('模型详情栏目', 'Model detail sections')}>
            {(['overview', 'card', 'files'] as const).map(item => <button type="button" role="tab" key={item} aria-selected={view === item} onClick={() => {
              setView(item);
              if (item === 'card') setOpenedCard(modelKey);
            }}>{item === 'overview' ? tr('概览', 'Overview') : item === 'card' ? tr('模型卡', 'Model card') : tr('文件', 'Files')}</button>)}
            {model.gated && <span className="gated-badge">{tr('需要授权', 'Gated')}</span>}
          </div>
          <div className="community-inspector-scroll">
            {view === 'overview' ? <>
              {model.description && <p className="community-description">{model.description}</p>}
              {details}
              <section className="community-download-options"><div className="community-section-heading"><h4>{tr('下载选项', 'Download options')}</h4><span>{variants.length}</span></div>
                {variants.length ? <div role="radiogroup" aria-label={tr('精度版本', 'Model variant')}>{variants.map(item => <label className="community-variant-option" key={item.id}>
                  <input type="radio" name="community-variant" checked={variant?.id === item.id} onChange={() => setVariantId(item.id)} />
                  <div><strong title={item.label}>{item.precision || item.label}</strong><small title={item.label}>{item.label}{item.files.length > 1 ? ` · ${item.files.length} ${tr('个文件', 'files')}` : ''}</small></div>
                  <span className="community-variant-format">{item.format.toUpperCase()}</span><span className="community-variant-size">{size(item.byte_size)}</span>
                </label>)}</div> : <p className="community-description">{tr('暂无可识别的精度版本，可在“文件”中选择下载。', 'No recognized variants. Browse Files to choose downloads.')}</p>}
              </section>
            </> : view === 'files' ? files : null}
            {openedCard === modelKey && <div className="community-model-card-panel" hidden={view !== 'card'}><ModelCard key={modelKey} model={model} tr={tr} /></div>}
          </div>
          {view === 'overview' && variant && <footer className="community-download-footer">{renderDownload(variant)}</footer>}
        </> : <div className="community-placeholder community-inspector-empty" role={detailError ? 'alert' : 'status'}><CubeIcon size={36} />
          <strong>{inspecting ? tr('正在读取模型…', 'Loading model…') : detailError ? tr('暂时无法读取此模型', 'Model details are unavailable') : tr('选择一个模型', 'Choose a model')}</strong>
          <p>{inspecting ? selectedItem?.repo_id : detailError ? tr('可以重试，也可以继续浏览其它模型。', 'Retry or continue browsing other models.') : tr('查看模型详情，选择精度并下载。', 'Explore model details and choose a download.')}</p>
          {detailError && selectedItem && <button type="button" onClick={() => void inspect(selectedItem)}>{tr('重试', 'Retry')}</button>}
        </div>}
      </section>
    </div>
  </div>;
}
