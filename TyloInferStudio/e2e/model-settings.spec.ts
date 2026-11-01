import { expect, test } from '@playwright/test';
import { mockStudioServer } from './mockServer';
import { openModelSettings } from './modelSettings';

const GiB = 2 ** 30;
const models = ['Model A', 'Model B'].map((name, index) => ({ id: `asset-${index}`, name,
  architecture: 'qwen35', format: 'tq', shard_count: 1, total_bytes: 4 * GiB, context_capacity: 131072,
  estimated_resident_weight_bytes: 4 * GiB, tensor_count: 1, record_count: 1, dtypes: [],
  complete: true, loadable: true, modified_at: '2026-11-01T00:00:00Z' }));

for (const language of ['en', 'zh-CN'] as const) {
  test(`remaining memory appears once beside Filter and follows live updates (${language})`, async ({ page }, testInfo) => {
    await mockStudioServer(page, { language });
    let headroom: number | null = 85.3 * GiB;
    await page.route('**/api/v1/runtime/status*', route => route.fulfill({ json: {
      runtime_state: 'idle', memory_architecture: 'unified', runtime_memory_headroom_bytes: headroom,
      active_requests: 0, sampling_defaults: {},
    } }));
    await page.route(/\/api\/v1\/models(?:\?.*)?$/, route => route.fulfill({ json: { data: models } }));
    await page.goto('/models');
    const memory = page.locator('.checkpoint-available-memory');
    await expect(memory).toHaveCount(1);
    await expect(memory).toContainText('85.3 GiB');
    await expect(page.locator('.model-row .checkpoint-available-memory')).toHaveCount(0);
    await expect(page.locator('.model-library-panel .model-row')).toHaveCount(2);
    const widths = testInfo.project.name === 'desktop' ? [1440, 1024] : [390];
    for (const width of widths) {
      await page.setViewportSize({ width, height: 1000 });
      const filterBounds = (await page.locator('.list-filters > summary').boundingBox())!;
      const memoryBounds = (await memory.boundingBox())!;
      expect(memoryBounds.x).toBeGreaterThanOrEqual(filterBounds.x + filterBounds.width);
      expect(Math.abs(memoryBounds.y + memoryBounds.height / 2 - filterBounds.y - filterBounds.height / 2)).toBeLessThan(2);
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    }
    headroom = 42.6 * GiB;
    await expect(memory).toContainText('42.6 GiB');
    headroom = 0;
    await expect(memory).toContainText('0 GiB');
    headroom = null;
    await expect(memory).toContainText('—');
  });

  test(`model summary uses three compact rounded cards (${language})`, async ({ page }) => {
    await mockStudioServer(page, { language });
    await page.goto('/models');
    const cards = page.locator('.model-library-summary > span');
    await expect(cards).toHaveCount(3);
    for (const card of await cards.all()) {
      await expect(card).toHaveCSS('border-radius', '10px');
      await expect(card).toHaveCSS('border-top-width', '1px');
      await expect(card.locator('strong')).toBeVisible();
    }
    const bounds = await cards.evaluateAll(elements => elements.map(element => {
      const box = element.getBoundingClientRect();
      return { y: box.y, height: box.height, fits: element.scrollWidth <= element.clientWidth };
    }));
    expect(new Set(bounds.map(box => box.y)).size).toBe(1);
    expect(bounds.every(box => box.height >= 64 && box.height <= 90 && box.fits)).toBe(true);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  });

  test(`local models group by architecture and model identity, preserving filtering (${language})`, async ({ page }) => {
    await mockStudioServer(page, { language });
    const items = [
      { ...models[0], id: 'flash-s2', name: 'Qwen3.8-Flash-Next-EW-S2-L', architecture: 'qwen4_exp_text' },
      { ...models[0], id: 'flash-iq3', name: 'Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS', architecture: 'qwen4_exp_text', format: 'gguf' },
      { ...models[0], id: 'dense', name: 'Qwen3.8-27B-V1_M', architecture: 'qwen3_5_text' },
      { ...models[0], id: 'deepseek', name: 'DeepSeek-V4.1-Flash-1017-Q3', architecture: 'deepseek_v41_text' },
    ];
    await page.route(/\/api\/v1\/models(?:\?.*)?$/, route => route.fulfill({ json: { data: items } }));
    await page.goto('/models');
    await expect(page.locator('.checkpoint-family-heading h3')).toHaveText(['Qwen', 'DeepSeek']);
    await expect(page.locator('.checkpoint-model-heading')).toHaveText(['Qwen3.8-Flash-Next', 'Qwen3.8-27B', 'DeepSeek-V4.1-Flash-1017']);
    await expect(page.getByRole('region', { name: 'Qwen3.8-Flash-Next', exact: true }).locator('.model-row')).toHaveCount(2);
    await expect(page.getByRole('region', { name: 'Qwen3.8-27B', exact: true }).locator('.model-row')).toHaveCount(1);
    const search = page.getByRole('searchbox', { name: language === 'en' ? 'Search local checkpoints' : '搜索本地检查点' });
    await search.fill('GSQ');
    await expect(page.locator('.checkpoint-family-heading h3')).toHaveText(['Qwen']);
    await expect(page.locator('.checkpoint-model-heading')).toHaveText(['Qwen3.8-Flash-Next']);
    await expect(page.locator('.model-library-panel .model-row')).toHaveCount(1);
    await search.fill('');
    await page.locator('.list-filters > summary').click();
    await page.getByRole('combobox', { name: language === 'en' ? 'Filter format' : '筛选格式' }).selectOption('gguf');
    await expect(page.locator('.checkpoint-family-heading h3')).toHaveText(['Qwen']);
    await expect(page.locator('.checkpoint-model-heading')).toHaveText(['Qwen3.8-Flash-Next']);
    await page.getByRole('button', { name: language === 'en' ? 'Reset' : '重置', exact: true }).click();
    await expect(page.locator('.checkpoint-family-panel')).toHaveCount(2);
    await expect(page.locator('.model-library-panel .model-row')).toHaveCount(4);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  });
}

test('model library comes first and load settings remain scoped to the chosen model', async ({ page }) => {
  await mockStudioServer(page);
  await page.route(/\/api\/v1\/models(?:\?.*)?$/, route => route.fulfill({ json: { data: models } }));
  await page.route('**/api/v1/runtime/instances', route => route.fulfill({ json: { data: [] } }));
  await page.route('**/api/v1/runtime/status*', route => route.fulfill({ json: { runtime_state: 'idle' } }));
  await page.route('**/api/v1/models/*/cache-profile*', route => route.fulfill({ json: {
    max_context: 131072, fixed_bytes: 0, fixed_components: [], components: [
      { group: 'GQA', name: 'raw_kv', bytes_per_row: 1024, tokens_per_row: 1, allocation: 'exact', minimum_rows: 0 },
    ],
  } }));
  const writes: Record<string, unknown>[] = [];
  await page.route('**/api/v1/models/load', route => {
    writes.push(route.request().postDataJSON());
    return route.fulfill({ json: { operation_id: 'load', status: 'accepted' } });
  });
  await page.goto('/models');
  await expect(page.locator('.model-library-panel .model-row')).toHaveCount(2);
  await expect(page.locator('.model-workbench-summary, .model-load-policy, .model-context-settings')).toHaveCount(0);
  expect((await page.locator('#local-checkpoints').boundingBox())!.y).toBeLessThan(400);
  const trigger = page.getByRole('button', { name: 'Configure model: Model A', exact: true });
  await trigger.click();
  let dialog = page.getByRole('dialog', { name: 'Model A', exact: true });
  await expect(dialog.getByRole('spinbutton', { name: 'Load context' })).toHaveValue('');
  await expect(dialog.getByRole('combobox', { name: 'Model to estimate' })).toHaveCount(0);
  await expect(dialog.getByRole('status', { name: 'Estimated total memory' })).toContainText('4.1 GiB');
  await dialog.getByRole('spinbutton', { name: 'Load context' }).fill('65536');
  await dialog.getByRole('tab', { name: 'Residency and device' }).click();
  await dialog.getByRole('switch', { name: 'Pin in memory' }).check();
  await dialog.getByRole('tab', { name: 'Context and KV' }).click();
  await expect(dialog.getByRole('spinbutton', { name: 'Load context' })).toHaveValue('65536');
  await page.keyboard.press('Escape');
  await expect(trigger).toBeFocused();
  dialog = await openModelSettings(page, 'Model B');
  await expect(dialog.getByRole('spinbutton', { name: 'Load context' })).toHaveValue('');
  await dialog.getByRole('tab', { name: 'Residency and device' }).click();
  await expect(dialog.getByRole('switch', { name: 'Pin in memory' })).not.toBeChecked();
  await page.keyboard.press('Escape');
  await page.locator('.model-row').filter({ hasText: 'Model B' }).getByRole('button', { name: 'Load', exact: true }).click();
  await expect.poll(() => writes.length).toBe(1);
  expect(writes[0]).toMatchObject({ model: 'Model B', context_size: null, pin: false });
  expect(writes[0]).not.toHaveProperty('context_management');
  await page.locator('.model-row').filter({ hasText: 'Model A' }).getByRole('button', { name: 'Load', exact: true }).click();
  await expect.poll(() => writes.length).toBe(2);
  expect(writes[1]).toMatchObject({ model: 'Model A', context_size: 65536 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});

test('running model residency applies only to that instance and never submits a context reload', async ({ page }) => {
  await mockStudioServer(page);
  await page.route(/\/api\/v1\/models(?:\?.*)?$/, route => route.fulfill({ json: { data: models } }));
  const instances = models.map((model, index) => ({ id: `running-${index}`, model: model.name, state: 'ready', devices: ['metal'],
    active_sessions: 0, queued_requests: 0, pinned: false, idle_ttl_seconds: null as number | null,
    context_size: 32768, context_capacity: 131072 }));
  await page.route('**/api/v1/runtime/instances', route => route.fulfill({ json: { data: instances } }));
  const writes: Record<string, unknown>[] = [];
  await page.route('**/api/v1/runtime/instances/running-1', route => {
    const payload = route.request().postDataJSON();
    writes.push(payload); Object.assign(instances[1], payload);
    return route.fulfill({ json: instances[1] });
  });
  const reloads: string[] = [];
  page.on('request', request => { if (request.method() === 'POST' && request.url().endsWith('/runtime/context')) reloads.push(request.url()); });
  const dialog = await openModelSettings(page, 'Model B');
  await expect(dialog.getByRole('spinbutton', { name: 'Global context cap' })).toHaveCount(0);
  await dialog.getByRole('tab', { name: 'Residency and device' }).click();
  await expect(dialog.getByRole('button', { name: 'Apply', exact: true })).toBeDisabled();
  await dialog.getByRole('combobox', { name: 'Idle unload' }).selectOption('300');
  await dialog.getByRole('button', { name: 'Apply', exact: true }).click();
  await expect.poll(() => writes).toEqual([{ pinned: false, idle_ttl_seconds: 300 }]);
  await expect(dialog.getByRole('status')).toHaveText('Applied');
  expect(reloads).toEqual([]);
  expect(instances[0].idle_ttl_seconds).toBeNull();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});
