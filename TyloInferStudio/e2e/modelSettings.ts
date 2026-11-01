import { expect, type Page } from '@playwright/test';

export async function openModelSettings(page: Page, name: string, language = 'en') {
  await page.goto('/models');
  const library = page.getByRole('button', { name: language === 'zh-CN' ? `配置模型：${name}` : `Configure model: ${name}`, exact: true });
  const running = page.locator('.loaded-model-panel .model-row').filter({ hasText: name }).getByRole('button', { name: language === 'zh-CN' ? '配置' : 'Configure', exact: true });
  await expect(library.or(running)).toBeVisible();
  await library.or(running).click();
  return page.getByRole('dialog', { name, exact: true });
}
