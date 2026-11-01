import { expect, test } from '@playwright/test';
import { mockStudioServer } from './mockServer';

for (const language of ['en', 'zh-CN'] as const) {
  test(`overview has one divided appearance control and Tools starts expanded (${language})`, async ({ page }, testInfo) => {
    await mockStudioServer(page, { language, preserveSettings: true });
    const errors: string[] = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto('/');
    const controls = page.locator('#dashboard-overview > .screen-header .appearance-controls');
    await expect(controls).toHaveCount(1);
    await expect(controls.getByRole('button')).toHaveCount(6);
    await expect(controls.locator(':scope > [role="group"]')).toHaveCount(2);
    await expect(controls.locator('.appearance-controls-divider')).toHaveCSS('width', '1px');
    await expect(controls.locator('.appearance-controls-divider')).toHaveCSS('height', '18px');
    await expect(controls).toHaveCSS('border-top-width', '1px');
    for (const group of await controls.locator(':scope > [role="group"]').all()) {
      await expect(group).toHaveCSS('border-top-width', '0px');
    }
    for (const button of await controls.getByRole('button').all()) {
      await expect(button).toHaveCSS('border-top-width', '0px');
      await expect(button).toHaveCSS('height', '26px');
    }
    const header = (await page.locator('#dashboard-overview > .screen-header').boundingBox())!;
    const bounds = (await controls.boundingBox())!;
    expect(Math.abs(bounds.x + bounds.width - header.x - header.width)).toBeLessThan(2);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await expect(page.locator('#studio-sidebar .appearance-controls')).toHaveCount(0);
    await controls.getByRole('button', { name: language === 'en' ? 'Dark' : '深色', exact: true }).click();
    await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
    await controls.getByRole('button', { name: '简体中文', exact: true }).click();
    await expect(page.getByRole('heading', { name: '概览', exact: true })).toBeVisible();
    await controls.getByRole('button', { name: 'English', exact: true }).click();
    await expect(page.getByRole('heading', { name: 'Overview', exact: true })).toBeVisible();
    await controls.getByRole('button', { name: 'Automatic appearance', exact: true }).click();
    await controls.getByRole('button', { name: 'Automatic language', exact: true }).click();
    await expect(page.locator('html')).toHaveAttribute('data-theme', 'system');
    expect(await page.evaluate(() => JSON.parse(localStorage.getItem('tyloinfer.studio.generation.v1')!))).toMatchObject({ theme: 'system', language: 'system' });
    await page.reload();
    await expect(controls.getByRole('button', { name: 'Automatic appearance', exact: true })).toHaveAttribute('aria-pressed', 'true');
    const sidebar = page.locator('#studio-sidebar');
    if (testInfo.project.name === 'mobile') await page.getByRole('button', { name: 'Open sidebar', exact: true }).click();
    const tools = sidebar.getByRole('button', { name: 'Tools', exact: true });
    await expect(tools).toHaveAttribute('aria-expanded', 'true');
    await expect(sidebar.getByRole('button', { name: 'Downloads', exact: true })).toBeVisible();
    await tools.click();
    await expect(tools).toHaveAttribute('aria-expanded', 'false');
    await expect(sidebar.getByRole('button', { name: 'Downloads', exact: true })).toHaveCount(0);
    await sidebar.getByRole('button', { name: 'Models', exact: true }).click();
    await expect(page.locator('.appearance-controls')).toHaveCount(0);
    expect(errors).toEqual([]);
  });
}
