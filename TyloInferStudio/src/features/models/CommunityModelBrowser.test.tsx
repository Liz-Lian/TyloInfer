import { useState } from 'react';
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import { modelsApi } from '../../shared/api/resources/models';
import type { HubModelInfo, HubModelSearchPage } from '../../shared/api/types';
import { CommunityModelBrowser } from './CommunityModelBrowser';

vi.mock('../../shared/api/resources/models', () => ({ modelsApi: {
  browseHubModels: vi.fn(), hubModelInfo: vi.fn(), resolveHubModel: vi.fn(), hubModelReadme: vi.fn(),
} }));

function info(name: string): HubModelInfo {
  return { provider: 'huggingface', repo_id: `community/${name}`, author: 'community', revision: 'main',
    downloads: 1234, likes: 56, total_bytes: 9 * 2 ** 30, files: [], tags: [], formats: ['gguf'],
    architectures: ['qwen'], modalities: ['text'], gated: false,
    variants: [3, 4].map(bits => ({ id: `q${bits}`, label: `Q${bits}_K_M`, precision: `Q${bits}_K_M`, format: 'gguf',
      files: [`model-q${bits}.gguf`], byte_size: bits * 2 ** 30,
      configuration: { status: 'unknown', recommendation: 'unknown', reasons: [] } })),
  };
}

const download = vi.fn();
function Harness({ backend }: { backend?: string } = {}) {
  const [model, setModel] = useState<HubModelInfo | null>(null);
  return <CommunityModelBrowser model={model} onSelect={setModel} variants={model?.variants ?? []} backend={backend}
    details={<p>Repository details: {model?.repo_id}</p>} files={<p>Repository files: {model?.repo_id}</p>}
    renderDownload={variant => <button type="button" onClick={() => download(variant)}>{`Download ${variant.label}`}</button>}
    tr={(_, en) => en} />;
}

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(modelsApi.browseHubModels).mockResolvedValue({ data: [info('Qwen-GGUF'), info('Gemma-GGUF')] });
  vi.mocked(modelsApi.hubModelInfo).mockImplementation(async (_, repo) => info(repo.split('/').pop()!));
  vi.mocked(modelsApi.resolveHubModel).mockResolvedValue(info('Direct-GGUF'));
  vi.mocked(modelsApi.hubModelReadme).mockResolvedValue({ content: '# Model documentation' });
});

it('browses popular models immediately and downloads only the chosen variant', async () => {
  render(<Harness />);
  await screen.findByRole('button', { name: 'Download Q3_K_M' });
  expect(modelsApi.browseHubModels).toHaveBeenCalledWith('huggingface', '', expect.objectContaining({ formats: ['tq', 'gguf', 'mlx'], sort: 'relevance', direction: 'desc' }));
  fireEvent.click(screen.getByRole('radio', { name: /Q4_K_M/ }));
  fireEvent.click(screen.getByRole('button', { name: 'Download Q4_K_M' }));
  expect(download).toHaveBeenCalledExactlyOnceWith(expect.objectContaining({ id: 'q4', files: ['model-q4.gguf'] }));
  fireEvent.click(screen.getByRole('tab', { name: 'Files' }));
  expect(screen.getByText('Repository files: community/Qwen-GGUF')).toBeVisible();
  expect(screen.queryByRole('button', { name: 'Download Q4_K_M' })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('tab', { name: 'Overview' }));
  expect(screen.getByRole('button', { name: 'Download Q4_K_M' })).toBeVisible();
});

it('sends format, sort and source changes to the server', async () => {
  render(<Harness />);
  await screen.findByRole('button', { name: 'Download Q3_K_M' });
  fireEvent.click(screen.getByRole('checkbox', { name: 'MLX' }));
  await waitFor(() => expect(modelsApi.browseHubModels).toHaveBeenLastCalledWith('huggingface', '', expect.objectContaining({ formats: ['tq', 'gguf'] })));
  fireEvent.change(screen.getByRole('combobox', { name: 'Sort models' }), { target: { value: 'updated' } });
  await waitFor(() => expect(modelsApi.browseHubModels).toHaveBeenLastCalledWith('huggingface', '', expect.objectContaining({ formats: ['tq', 'gguf'], sort: 'updated' })));
  fireEvent.change(screen.getByRole('combobox', { name: 'Model source' }), { target: { value: 'modelscope' } });
  await waitFor(() => expect(modelsApi.browseHubModels).toHaveBeenLastCalledWith('modelscope', '', expect.objectContaining({ formats: ['tq', 'gguf'], sort: 'updated' })));
});

it('accepts a direct repository URL without waiting for the search debounce', async () => {
  render(<Harness />);
  await screen.findByRole('button', { name: 'Download Q3_K_M' });
  fireEvent.change(screen.getByRole('textbox', { name: 'Search models' }), { target: { value: 'https://huggingface.co/community/Direct-GGUF' } });
  fireEvent.click(screen.getByRole('button', { name: 'Search' }));
  await screen.findByRole('heading', { name: 'Direct-GGUF' });
  expect(modelsApi.resolveHubModel).toHaveBeenCalledWith('https://huggingface.co/community/Direct-GGUF', 'huggingface', expect.any(AbortSignal));
});

it('ignores a stale model detail response after another result is selected', async () => {
  let resolveFirst!: (model: HubModelInfo) => void;
  vi.mocked(modelsApi.hubModelInfo).mockImplementationOnce(() => new Promise(resolve => { resolveFirst = resolve; }));
  render(<Harness />);
  const results = await screen.findByRole('region', { name: 'Model search results' });
  await waitFor(() => expect(modelsApi.hubModelInfo).toHaveBeenCalledOnce());
  const signal = vi.mocked(modelsApi.hubModelInfo).mock.calls[0][3];
  fireEvent.click(within(results).getByRole('button', { name: /Gemma-GGUF/ }));
  await screen.findByRole('heading', { name: 'Gemma-GGUF' });
  await act(async () => resolveFirst(info('Qwen-GGUF')));
  expect(signal?.aborted).toBe(true);
  expect(screen.getByText('Repository details: community/Gemma-GGUF')).toBeVisible();
  expect(screen.queryByText('Repository details: community/Qwen-GGUF')).not.toBeInTheDocument();
});

it('ignores a stale search response after a new search is submitted', async () => {
  let resolveFirst!: (models: HubModelSearchPage) => void;
  vi.mocked(modelsApi.browseHubModels).mockImplementationOnce(() => new Promise(resolve => { resolveFirst = resolve; }))
    .mockResolvedValueOnce({ data: [info('New-GGUF')] });
  render(<Harness />);
  await waitFor(() => expect(modelsApi.browseHubModels).toHaveBeenCalledOnce());
  fireEvent.change(screen.getByRole('textbox', { name: 'Search models' }), { target: { value: 'new' } });
  fireEvent.click(screen.getByRole('button', { name: 'Search' }));
  await screen.findByRole('heading', { name: 'New-GGUF' });
  await act(async () => resolveFirst({ data: [info('Stale-GGUF')] }));
  expect(screen.queryByText('Stale-GGUF')).not.toBeInTheDocument();
  expect(screen.getByRole('heading', { name: 'New-GGUF' })).toBeVisible();
});

it('renders the repository model card safely and resolves relative links and images', async () => {
  vi.mocked(modelsApi.hubModelReadme).mockResolvedValue({ content: '---\nprivate: metadata\n---\n# Model documentation\n\n[Guide](guide.md)\n\n![Chart](assets/chart.png)\n\n<script>dangerous()</script><iframe src="https://invalid.test"></iframe><a href="javascript:alert(1)">Unsafe link</a>' });
  const { container } = render(<Harness />);
  await screen.findByRole('button', { name: 'Download Q3_K_M' });
  fireEvent.click(screen.getByRole('tab', { name: 'Model card' }));
  await screen.findByRole('heading', { name: 'Model documentation' });
  expect(container.querySelector('.community-model-card script, .community-model-card iframe')).toBeNull();
  expect(screen.queryByText('private: metadata')).not.toBeInTheDocument();
  expect(screen.getByRole('link', { name: 'Guide' })).toHaveAttribute('href', 'https://huggingface.co/community/Qwen-GGUF/blob/main/guide.md');
  expect(screen.getByRole('img', { name: 'Chart' })).toHaveAttribute('src', 'https://huggingface.co/community/Qwen-GGUF/resolve/main/assets/chart.png');
  expect(screen.getByText('Unsafe link')).not.toHaveAttribute('href');
  expect(screen.queryByRole('button', { name: 'Download Q3_K_M' })).not.toBeInTheDocument();
});

it('preserves loaded image elements across parent status refreshes', async () => {
  vi.mocked(modelsApi.hubModelReadme).mockResolvedValue({ content: '# Model documentation\n\n![Chart](assets/chart.png)' });
  const view = render(<Harness />);
  await screen.findByRole('button', { name: 'Download Q3_K_M' });
  fireEvent.click(screen.getByRole('tab', { name: 'Model card' }));
  const image = await screen.findByRole('img', { name: 'Chart' });
  for (let index = 0; index < 3; index += 1) {
    view.rerender(<Harness />);
    expect(screen.getByRole('img', { name: 'Chart' })).toBe(image);
  }
  expect(modelsApi.hubModelReadme).toHaveBeenCalledOnce();
});

it('retains the loaded model card and images when switching detail tabs', async () => {
  vi.mocked(modelsApi.hubModelReadme).mockResolvedValue({ content: '# Model documentation\n\n![Chart](assets/chart.png)' });
  render(<Harness />);
  await screen.findByRole('button', { name: 'Download Q3_K_M' });
  expect(modelsApi.hubModelReadme).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('radio', { name: /Q4_K_M/ }));
  fireEvent.click(screen.getByRole('tab', { name: 'Model card' }));
  const image = await screen.findByRole('img', { name: 'Chart' });
  expect(screen.queryByRole('button', { name: /Download Q/ })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('tab', { name: 'Files' }));
  expect(image).not.toBeVisible();
  expect(screen.queryByRole('button', { name: /Download Q/ })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('tab', { name: 'Overview' }));
  expect(screen.getByRole('button', { name: 'Download Q4_K_M' })).toBeVisible();
  fireEvent.click(screen.getByRole('tab', { name: 'Model card' }));
  expect(screen.getByRole('img', { name: 'Chart' })).toBe(image);
  expect(image).toBeVisible();
  expect(modelsApi.hubModelReadme).toHaveBeenCalledOnce();
});

it('keeps downloads available when a model card fails and supports retry', async () => {
  vi.mocked(modelsApi.hubModelReadme).mockRejectedValueOnce(new Error('offline'));
  render(<Harness />);
  await screen.findByRole('button', { name: 'Download Q3_K_M' });
  fireEvent.click(screen.getByRole('tab', { name: 'Model card' }));
  expect(await screen.findByRole('alert')).toHaveTextContent('return to Overview to download');
  expect(screen.queryByRole('button', { name: 'Download Q3_K_M' })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Retry' }));
  await screen.findByRole('heading', { name: 'Model documentation' });
  fireEvent.click(screen.getByRole('tab', { name: 'Overview' }));
  expect(screen.getByRole('button', { name: 'Download Q3_K_M' })).toBeEnabled();
});

it('shows a recoverable search error instead of raw backend exceptions', async () => {
  vi.mocked(modelsApi.browseHubModels).mockRejectedValueOnce(new Error('http_500: Internal Server Error'));
  render(<Harness />);
  expect(await screen.findByRole('alert')).toHaveTextContent('Cannot reach the model source');
  expect(screen.queryByText(/http_500/)).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Retry' }));
  await screen.findByRole('button', { name: 'Download Q3_K_M' });
});

it('continues past one hundred results without replacing the selected model or precision', async () => {
  vi.mocked(modelsApi.browseHubModels)
    .mockResolvedValueOnce({ data: Array.from({ length: 40 }, (_, index) => info(`model-${index}`)), next_cursor: 'page-two' })
    .mockResolvedValueOnce({ data: Array.from({ length: 40 }, (_, index) => info(`model-${index + 40}`)), next_cursor: 'page-three' })
    .mockResolvedValueOnce({ data: Array.from({ length: 40 }, (_, index) => info(`model-${index + 80}`)), next_cursor: null });
  render(<Harness />);
  await screen.findByRole('heading', { name: 'model-0' });
  fireEvent.click(screen.getByRole('radio', { name: /Q4_K_M/ }));
  fireEvent.click(screen.getByRole('button', { name: 'Show more' }));
  await screen.findByText('model-79');
  expect(screen.getByRole('button', { name: 'Download Q4_K_M' })).toBeVisible();
  fireEvent.click(screen.getByRole('button', { name: 'Show more' }));
  await screen.findByText('model-119');
  expect(modelsApi.browseHubModels).toHaveBeenLastCalledWith('huggingface', '', expect.objectContaining({ cursor: 'page-three' }));
  expect(modelsApi.hubModelInfo).toHaveBeenCalledOnce();
  expect(screen.getByRole('heading', { name: 'model-0' })).toBeVisible();
  expect(screen.queryByRole('button', { name: 'Show more' })).not.toBeInTheDocument();
});

it('defaults to supported formats on CUDA without preventing explicit cross-device downloads', async () => {
  render(<Harness backend="cuda" />);
  await screen.findByRole('button', { name: 'Download Q3_K_M' });
  expect(screen.getByRole('checkbox', { name: 'MLX' })).not.toBeChecked();
  expect(modelsApi.browseHubModels).toHaveBeenLastCalledWith('huggingface', '', expect.objectContaining({ formats: ['tq', 'gguf'] }));
  fireEvent.click(screen.getByRole('checkbox', { name: 'MLX' }));
  await waitFor(() => expect(modelsApi.browseHubModels).toHaveBeenLastCalledWith('huggingface', '', expect.objectContaining({ formats: ['tq', 'gguf', 'mlx'] })));
});

it('supports ascending HF sorting and resets it for a source that cannot honor it', async () => {
  render(<Harness />);
  await screen.findByRole('button', { name: 'Download Q3_K_M' });
  fireEvent.click(screen.getByRole('button', { name: 'Descending' }));
  await waitFor(() => expect(modelsApi.browseHubModels).toHaveBeenLastCalledWith('huggingface', '', expect.objectContaining({ direction: 'asc' })));
  fireEvent.change(screen.getByRole('combobox', { name: 'Model source' }), { target: { value: 'modelscope' } });
  await waitFor(() => expect(modelsApi.browseHubModels).toHaveBeenLastCalledWith('modelscope', '', expect.objectContaining({ direction: 'desc' })));
  expect(screen.getByRole('button', { name: 'Descending' })).toBeDisabled();
});

it('shows a validation hint and makes no search request when all formats are unchecked', async () => {
  render(<Harness />);
  await screen.findByRole('button', { name: 'Download Q3_K_M' });
  vi.mocked(modelsApi.browseHubModels).mockClear();
  for (const format of ['TQ', 'GGUF', 'MLX']) fireEvent.click(screen.getByRole('checkbox', { name: format }));
  expect(await screen.findByText('Select at least one weight format.')).toBeVisible();
  await waitFor(() => expect(screen.queryByRole('button', { name: 'Download Q3_K_M' })).not.toBeInTheDocument());
  expect(modelsApi.browseHubModels).not.toHaveBeenCalled();
});

it.each([
  [1, '1 B'], [512, '512 B'], [1024, '1 KiB'], [1536, '1.5 KiB'],
  [2 ** 20, '1 MiB'], [32 * 2 ** 20, '32 MiB'], [512 * 2 ** 20, '512 MiB'],
  [1023 * 2 ** 20, '1,023 MiB'], [2 ** 30, '1 GiB'], [2 ** 40, '1 TiB'],
  [0, '—'], [-1, '—'], [Number.NaN, '—'], [Number.POSITIVE_INFINITY, '—'],
])('formats a %s-byte download without rounding small files to zero GiB', async (bytes, expected) => {
  const data = info('Qwen-GGUF');
  data.variants = [{ ...data.variants[0], byte_size: bytes }];
  vi.mocked(modelsApi.hubModelInfo).mockResolvedValueOnce(data);
  const view = render(<Harness />);
  await screen.findByRole('button', { name: 'Download Q3_K_M' });
  expect(view.container.querySelector('.community-variant-size')).toHaveTextContent(expected);
});
