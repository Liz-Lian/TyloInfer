import { expect, test } from '@playwright/test';
import { mockStudioServer } from './mockServer';

test('browses repositories, selects one quantization and reads model cards without leaving the downloader', async ({ page }, testInfo) => {
  await mockStudioServer(page);
  await page.route('**/api/v1/jobs/kinds', route => route.fulfill({ json: { data: [{ kind: 'download.huggingface', payload_schema: {} }] } }));
  const model = {
    provider: 'huggingface', repo_id: 'community/Qwen3-GGUF', author: 'community', revision: 'main',
    downloads: 12345, likes: 123, total_bytes: 7 * 2 ** 30, tags: ['gguf'], formats: ['gguf'],
    architectures: ['qwen3'], modalities: ['text'], gated: false,
    description: 'A community quantization with multiple download options.',
    files: [{ name: 'model-q3.gguf', byte_size: 3 * 2 ** 30 },
      { name: 'model-q4-00001-of-00002.gguf', byte_size: 2 * 2 ** 30 },
      { name: 'model-q4-00002-of-00002.gguf', byte_size: 2 * 2 ** 30 }],
    variants: [3, 4].map(bits => ({ id: `q${bits}`, label: `Q${bits}_K_M`, precision: `Q${bits}_K_M`, format: 'gguf',
      files: bits === 4 ? ['model-q4-00001-of-00002.gguf', 'model-q4-00002-of-00002.gguf'] : ['model-q3.gguf'], byte_size: bits * 2 ** 30,
      configuration: { status: 'recommended', recommendation: 'three_stars', required_memory_bytes: bits * 2 ** 30, available_memory_bytes: 16 * 2 ** 30, reasons: [] } })),
  };
  const searches: URL[] = [];
  await page.route('**/api/v1/hub/models?*', route => {
    searches.push(new URL(route.request().url()));
    return route.fulfill({ json: { data: [model], next_cursor: null } });
  });
  await page.route('**/api/v1/hub/models/huggingface/community/Qwen3-GGUF', route => route.fulfill({ json: model }));
  let cardReads = 0;
  let imageReads = 0;
  await page.route('**/api/v1/hub/models/huggingface/community/Qwen3-GGUF/readme?*', route => {
    cardReads += 1;
    return route.fulfill({ json: { content: `# Community model card\n\nDocumentation fetched from the model repository.\n\n![Precision chart](chart.svg)\n\n[Model guide](guide.md)\n\n\`\`\`\n${'wide-model-command '.repeat(80)}\n\`\`\`` } });
  });
  await page.route('https://huggingface.co/community/Qwen3-GGUF/resolve/main/chart.svg', route => {
    imageReads += 1;
    return route.fulfill({ contentType: 'image/svg+xml', headers: { 'Cache-Control': 'no-store' }, body: '<svg xmlns="http://www.w3.org/2000/svg" width="4096" height="2048"><rect width="4096" height="2048" fill="gray"/></svg>' });
  });
  const downloads: Record<string, unknown>[] = [];
  await page.route(/\/api\/v1\/jobs$/, route => {
    if (route.request().method() !== 'POST') return route.fallback();
    downloads.push(route.request().postDataJSON());
    return route.fulfill({ json: { id: 'new-download', kind: 'download.huggingface', status: 'queued', progress: 0, cancel_requested: false, payload: {}, created_at: '', updated_at: '' } });
  });
  await page.goto('/model-hub');
  await page.getByRole('tab', { name: 'Model browser', exact: true }).click();
  const inspector = page.getByRole('region', { name: 'Model details' });
  await expect(inspector.getByRole('heading', { name: 'Qwen3-GGUF' })).toBeVisible();
  await page.getByRole('checkbox', { name: 'MLX', exact: true }).uncheck();
  await expect.poll(() => searches.at(-1)?.searchParams.get('formats')).toBe('tq,gguf');
  await inspector.getByRole('radio', { name: /Q4_K_M/ }).check();
  await inspector.getByRole('tab', { name: 'Model card', exact: true }).click();
  await expect(inspector.getByRole('heading', { name: 'Community model card' })).toBeVisible();
  await expect(inspector.getByRole('link', { name: 'Model guide' })).toHaveAttribute('href', 'https://huggingface.co/community/Qwen3-GGUF/blob/main/guide.md');
  await expect(inspector.locator('.community-download-footer')).toHaveCount(0);
  const image = inspector.getByRole('img', { name: 'Precision chart' });
  await expect.poll(() => image.evaluate(node => (node as HTMLImageElement).naturalWidth)).toBe(4096);
  const dimensions = await image.evaluate(node => {
    const image = node.getBoundingClientRect();
    const panel = node.closest('.community-inspector-scroll')!;
    const style = getComputedStyle(panel);
    return { width: image.width, height: image.height,
      available: panel.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight),
      overflow: panel.scrollWidth - panel.clientWidth };
  });
  expect(dimensions.width).toBeLessThanOrEqual(dimensions.available + 1);
  expect(dimensions.overflow).toBeLessThanOrEqual(1);
  expect(dimensions.width / dimensions.height).toBeCloseTo(2, 2);
  const loadedImage = await image.elementHandle();
  await inspector.getByRole('tab', { name: 'Files', exact: true }).click();
  await expect(inspector.locator('.community-download-footer')).toHaveCount(0);
  await inspector.getByRole('tab', { name: 'Model card', exact: true }).click();
  expect(await loadedImage!.evaluate(node => node === document.querySelector('.community-model-card img'))).toBe(true);
  await expect(image).toBeVisible();
  expect(cardReads).toBe(1);
  expect(imageReads).toBe(1);
  await inspector.getByRole('tab', { name: 'Overview', exact: true }).click();
  await inspector.getByRole('button', { name: 'Download', exact: true }).click();
  await expect.poll(() => downloads.length).toBe(1);
  expect(downloads[0]).toMatchObject({ kind: 'download.huggingface', payload: {
    repo_id: 'community/Qwen3-GGUF', revision: 'main', include: expect.arrayContaining(['model-q4-00001-of-00002.gguf', 'model-q4-00002-of-00002.gguf', 'tokenizer*']),
  } });
  expect((downloads[0].payload as { include: string[] }).include).not.toContain('model-q3.gguf');
  await expect(page.locator('.download-circle-count')).toHaveText('1');
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  const body = page.locator('.community-browser-body');
  const columns = await body.evaluate(element => getComputedStyle(element).gridTemplateColumns.split(' ').length);
  expect(columns).toBe(testInfo.project.name === 'mobile' ? 1 : 2);
});

test('downloads a complete model folder as one task without sibling weights', async ({ page }) => {
  await mockStudioServer(page);
  await page.route('**/api/v1/jobs/kinds', route => route.fulfill({ json: { data: [{ kind: 'download.huggingface', payload_schema: {} }] } }));
  const names = ['4bit/model-00001-of-00002.safetensors', '4bit/model-00002-of-00002.safetensors', '4bit/config.json', '4bit/tokenizer.json'];
  const model = {
    provider: 'huggingface', repo_id: 'community/Qwen-MLX', author: 'community', revision: 'main',
    downloads: 1, likes: 1, total_bytes: 5120, tags: ['mlx'], formats: ['mlx'],
    architectures: ['qwen3'], modalities: ['text'], gated: false,
    files: [...names, '8bit/model.safetensors'].map(name => ({ name, byte_size: 1024 })),
    variants: [{ id: 'hf:4bit', label: '4bit', format: 'hf', files: names.slice(0, 2), byte_size: 2048,
      configuration: { status: 'unknown', recommendation: 'unknown', reasons: [] } }],
  };
  await page.route('**/api/v1/hub/models?*', route => route.fulfill({ json: { data: [model], next_cursor: null } }));
  await page.route('**/api/v1/hub/models/huggingface/community/Qwen-MLX', route => route.fulfill({ json: model }));
  const downloads: Record<string, unknown>[] = [];
  await page.route(/\/api\/v1\/jobs$/, route => {
    if (route.request().method() !== 'POST') return route.fallback();
    downloads.push(route.request().postDataJSON());
    return route.fulfill({ json: { id: 'folder-download', kind: 'download.huggingface', status: 'queued', progress: 0, cancel_requested: false, payload: {}, created_at: '', updated_at: '' } });
  });
  await page.goto('/model-hub');
  await page.getByRole('tab', { name: 'Model browser', exact: true }).click();
  const inspector = page.getByRole('region', { name: 'Model details' });
  await inspector.getByRole('tab', { name: 'Files', exact: true }).click();
  await inspector.getByRole('button', { name: 'Download folder 4bit', exact: true }).click();
  await expect.poll(() => downloads.length).toBe(1);
  expect(downloads[0]).toMatchObject({ kind: 'download.huggingface', payload: {
    repo_id: 'community/Qwen-MLX', revision: 'main', include: names, expected_bytes: 4096,
  } });
  await expect(page.locator('.download-circle-count')).toHaveText('1');
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
});
