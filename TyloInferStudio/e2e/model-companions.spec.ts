import { expect, test } from '@playwright/test';
import { mockStudioServer } from './mockServer';

for (const language of ['en', 'zh-CN'] as const) {
  test(`independent tables are visible without model-loading UI (${language})`, async ({ page }) => {
    await mockStudioServer(page, { language });
    await page.route('**/api/v1/runtime/instances', route => route.fulfill({ json: { data: [] } }));
    const base = { architecture: 'qwen4_exp_text', format: 'tq', shard_count: 2, total_bytes: 2 ** 30,
      tensor_count: 2, record_count: 2, dtypes: [], complete: true, loadable: false, modified_at: '' };
    const items = [
      { ...base, id: 'model', name: 'Qwen3.8-Flash-Next-S2-L', asset_kind: 'model', loadable: true,
        total_bytes: 2 ** 31, owned_file_bytes: 2 ** 30 },
      { ...base, id: 'ple', name: 'Qwen3.8-Flash-Next-PLE', asset_kind: 'ple' },
      { ...base, id: 'engram', name: 'Qwen3.8-Flash-Next-Engram', asset_kind: 'engram' },
    ];
    let optedIn = false;
    await page.route(/\/api\/v1\/models(?:\?.*)?$/, route => {
      optedIn = new URL(route.request().url()).searchParams.get('include_components') === 'true';
      return route.fulfill({ json: { data: items } });
    });
    await page.goto('/models');
    await expect(page.locator('.component-asset-row')).toHaveCount(2);
    expect(optedIn).toBe(true);
    await expect(page.locator('.checkpoint-model-heading')).toHaveText(['Qwen3.8-Flash-Next']);
    await expect(page.locator('.model-library-summary strong')).toHaveText(['0', '1', '3 GiB']);
    for (const kind of ['PLE', 'Engram']) {
      const row = page.locator('.component-asset-row').filter({ hasText: language === 'en'
        ? `Independently usable ${kind}` : `可独立使用的${kind}` });
      await expect(row).toBeVisible();
      await expect(row.locator('.local-checkpoint-resources, .model-load-action, .model-settings-action')).toHaveCount(0);
      await expect(row.getByRole('button')).toHaveCount(1);
      const bounds = await row.boundingBox();
      expect(bounds!.x).toBeGreaterThanOrEqual(0);
      expect(bounds!.x + bounds!.width).toBeLessThanOrEqual(page.viewportSize()!.width);
    }
    await expect(page.locator('.model-load-action')).toHaveCount(1);
    const directoryRequests: string[] = [];
    await page.route('**/api/v1/models/engram/directory', route => {
      directoryRequests.push(route.request().url());
      return route.fulfill({ json: { current_id: 'folder', current_name: 'Engram', current_path: '/models/Engram',
        data: [], files: [{ name: 'table.tq', byte_size: 2 ** 30 }], model_file_count: 0 } });
    });
    await page.locator('.component-asset-row').filter({ hasText: 'Engram' }).getByRole('button').click();
    await expect(page.getByRole('dialog')).toBeVisible();
    await expect(page.getByRole('dialog')).toContainText('table.tq');
    expect(directoryRequests).toHaveLength(1);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  });
}
