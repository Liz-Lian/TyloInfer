import { expect, test } from '@playwright/test';
import { mockStudioServer } from './mockServer';

const GiB = 2 ** 30;
const notice = 'Insufficient remaining memory. Experts that do not fit will be streamed from SSD during inference.';

function artifact(id: string, name: string, experts: number) {
  return { id, name, architecture: experts ? 'qwen4_exp' : 'qwen35', format: 'mfq', shard_count: 1,
    total_bytes: 100 * GiB, estimated_resident_weight_bytes: 80 * GiB, ssd_ple_bytes: 20 * GiB,
    estimated_weight_bytes_by_role: { dense: (80 - experts) * GiB, experts: experts * GiB },
    tensor_count: 1, record_count: 1, dtypes: ['F16'], complete: true, loadable: true, modified_at: '2026-10-19T00:00:00Z' };
}

for (const memory_architecture of ['unified', 'discrete'] as const) {
  test(`low-memory MoE on ${memory_architecture} memory explains SSD streaming and still loads`, async ({ page }) => {
    await mockStudioServer(page);
    await page.route('**/api/v1/runtime/status*', route => route.fulfill({ json: {
      runtime_state: 'idle', memory_architecture, active_requests: 0, sampling_defaults: {},
      host_memory_available_bytes: 16 * GiB, runtime_memory_headroom_bytes: 16 * GiB,
      ...(memory_architecture === 'discrete' ? { gpu_devices: [{ id: '0', name: 'GPU', memory_available_bytes: 8 * GiB }] } : {}),
    } }));
    await page.route(/\/api\/v1\/models(?:\?.*)?$/, route => route.fulfill({ json: { data: [
      artifact('moe', 'Low Memory MoE', 76), artifact('dense', 'Low Memory Dense', 0),
    ] } }));
    const writes: Record<string, unknown>[] = [];
    let release = () => {};
    const held = new Promise<void>(resolve => { release = resolve; });
    await page.route('**/api/v1/models/load', async route => {
      writes.push(route.request().postDataJSON());
      await held;
      await route.fulfill({ json: { operation_id: 'load', status: 'accepted' } });
    });
    await page.goto('/models');

    const moe = page.locator('.model-row', { hasText: 'Low Memory MoE' });
    const load = moe.getByRole('button', { name: 'Load', exact: true });
    await expect(moe.locator('.model-streaming-notice')).toHaveText(notice);
    await expect(load).toBeEnabled();
    await expect(load).toHaveAccessibleDescription(notice);
    const dense = page.locator('.model-row', { hasText: 'Low Memory Dense' });
    await expect(dense.locator('.model-streaming-notice')).toHaveCount(0);
    await expect(dense.getByRole('button', { name: 'Load', exact: true })).toBeEnabled();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);

    await load.click();
    await expect.poll(() => writes.length).toBe(1);
    expect(writes[0]).toMatchObject({ model: 'Low Memory MoE', context_size: null, prefill_chunk_size: 'auto', pin: false });
    expect(writes[0]).not.toHaveProperty('context_management');
    expect(writes[0]).not.toHaveProperty('device_ids');
    await expect(load).toBeDisabled();
    release();
    await expect(load).toBeEnabled();
    expect(writes).toHaveLength(1);
  });
}

test('checkpoint warnings stay left of pressure bars with equal row heights and stable action columns', async ({ page }, testInfo) => {
  await mockStudioServer(page);
  await page.route('**/api/v1/runtime/status*', route => route.fulfill({ json: {
    runtime_state: 'idle', memory_architecture: 'unified', runtime_memory_headroom_bytes: 16 * GiB,
    active_requests: 0, sampling_defaults: {},
  } }));
  await page.route(/\/api\/v1\/models(?:\?.*)?$/, route => route.fulfill({ json: { data: [
    { ...artifact('fit', 'Fitting model', 0), estimated_resident_weight_bytes: 4 * GiB },
    artifact('streamed', 'Streamed model', 76),
    { ...artifact('incomplete', 'Incomplete model', 76), shard_count: 8, missing_shards: 2, complete: false, loadable: false, estimated_resident_weight_bytes: null },
    { ...artifact('convert', 'Conversion model', 0), format: 'hf', loadable: false },
  ] } }));
  await page.goto('/models');
  const rows = page.locator('.model-library-panel .model-row');
  await expect(rows).toHaveCount(4);
  const unavailableLoad = page.locator('.model-row', { hasText: 'Incomplete model' }).getByRole('button', { name: 'Load', exact: true });
  await expect(unavailableLoad).toBeVisible();
  await expect(unavailableLoad).toBeDisabled();
  await expect(unavailableLoad).toHaveAccessibleDescription('Incomplete shards');
  const widths = testInfo.project.name === 'desktop' ? [1440, 1024] : [390];
  for (const width of widths) {
    await page.setViewportSize({ width, height: 1000 });
    const layout = await rows.evaluateAll(elements => elements.map(element => {
      const row = element.getBoundingClientRect();
      const pressure = element.querySelector('.local-checkpoint-memory')!.getBoundingClientRect();
      const resources = element.querySelector('.local-checkpoint-resources')!.getBoundingClientRect();
      const notice = element.querySelector('.local-checkpoint-notice')!.getBoundingClientRect();
      const files = element.querySelector('.model-files-action')!.getBoundingClientRect();
      return { height: row.height, pressureLeft: pressure.left, pressureRight: pressure.right, noticeRight: notice.right,
        noticeCenter: notice.y + notice.height / 2, resourcesCenter: resources.y + resources.height / 2,
        filesLeft: files.left, rowRight: row.right,
        actionWarnings: element.querySelector('.model-row-controls')!.textContent!.includes('Incomplete shards') || element.querySelector('.model-row-controls')!.textContent!.includes('Insufficient remaining memory.') };
    }));
    expect(Math.max(...layout.map(row => row.height)) - Math.min(...layout.map(row => row.height)), JSON.stringify({ width, layout })).toBeLessThanOrEqual(1);
    const message = page.locator('.model-streaming-notice');
    await expect(message).toHaveCSS('white-space', 'normal');
    await expect(message).not.toHaveCSS('text-overflow', 'ellipsis');
    expect(await message.evaluate(element => element.scrollHeight <= element.clientHeight && element.scrollWidth <= element.clientWidth)).toBe(true);
    expect(Math.max(...layout.map(row => row.pressureLeft)) - Math.min(...layout.map(row => row.pressureLeft))).toBeLessThanOrEqual(1);
    expect(Math.max(...layout.map(row => row.filesLeft)) - Math.min(...layout.map(row => row.filesLeft))).toBeLessThanOrEqual(1);
    for (const row of layout) {
      expect(row.noticeRight).toBeLessThan(row.pressureLeft);
      expect(row.pressureRight).toBeLessThanOrEqual(row.rowRight + 1);
      expect(Math.abs(row.noticeCenter - row.resourcesCenter)).toBeLessThanOrEqual(1);
      expect(row.actionWarnings).toBe(false);
    }
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  }
});
