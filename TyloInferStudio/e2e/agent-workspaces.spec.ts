import { expect, test } from '@playwright/test';
import { mockStudioServer } from './mockServer';
import type { AgentState } from '../src/features/agent/AgentProvider';

test('workspace toolbar searches names and content and applies persistent DSH view options', async ({ page }, testInfo) => {
  await mockStudioServer(page);
  const state: AgentState = { running: true, workspaces: [
    { workspaceId: 'root', title: 'Project', path: '/project', sessionIds: ['old', 'recent', 'archived', 'blank'] },
    { workspaceId: 'app', title: 'App', path: '/project/app', sessionIds: ['nested'] },
    { workspaceId: 'other', title: 'Other', path: '/other', sessionIds: ['review'] }],
    sessions: [['old', 'Old design', 1], ['recent', 'Recent design', 10], ['archived', 'Archived plan', 20], ['nested', 'Nested notes', 5], ['review', 'Review build', 15], ['loose', 'Loose notes', 2]]
      .map(([id, title, updatedAt]) => ({ sessionId: String(id), running: false, blank: false, updatedAt: Number(updatedAt), projections: { values: { title: String(title) } } })),
    archivedSessionIds: ['archived'], models: [], approvals: [] };
  state.sessions.push({ sessionId: 'blank', running: false, blank: true });
  await page.route('**/api/v1/agent/state', route => route.fulfill({ json: state }));
  await page.route('**/api/v1/agent/start', route => route.fulfill({ json: state }));
  await page.route('**/api/v1/agent/sessions/*/events', route => route.fulfill({ contentType: 'application/x-ndjson', body: JSON.stringify({ type: 'snapshot', cursor: 0, records: [] }) + '\n' }));
  const mutations: string[] = [];
  await page.route('**/api/v1/agent/rpc', route => {
    const body = route.request().postDataJSON(); mutations.push(body.method);
    if (body.method === 'workspace/unarchiveSession') state.archivedSessionIds = [];
    return route.fulfill({ json: { archivedSessionIds: [] } });
  });
  let connects = 0;
  await page.route('**/api/v1/agent/workspaces/*/session', route => { connects++; return route.fulfill({ json: { sessionId: 'blank' } }); });
  let releaseOld!: () => void;
  const oldResponse = new Promise<void>(resolve => { releaseOld = resolve; });
  let oldStarted = false;
  let rejectSearch = false;
  await page.route(/\/api\/v1\/agent\/search\?/, async route => {
    const query = new URL(route.request().url()).searchParams.get('query');
    if (query === 'stale') { oldStarted = true; await oldResponse; return route.fulfill({ json: { items: [{ sessionId: 'old', snippet: 'obsolete response' }], hasMore: false } }).catch(() => {}); }
    if (rejectSearch) { rejectSearch = false; return route.fulfill({ status: 503, json: { error: { code: 'agent_search_failed', message: 'Search unavailable', retryable: true, details: {} } } }); }
    return route.fulfill({ json: { items: query === 'phrase' || query === 'overflow' ? [{ sessionId: 'review', snippet: 'Unique phrase in conversation history' }] : [], hasMore: query === 'overflow' } });
  });
  await page.goto('/agent');
  if (testInfo.project.name === 'mobile') await page.getByRole('button', { name: 'Open sidebar', exact: true }).click();
  const sidebar = page.locator('#studio-sidebar');
  const toolbar = sidebar.locator('.agent-workspace-toolbar');
  expect(await toolbar.getByRole('button').evaluateAll(buttons => buttons.map(button => button.getAttribute('aria-label')))).toEqual(['Search conversations', 'View options', 'Add workspace']);
  await expect(sidebar.getByRole('button', { name: 'Recent design', exact: true })).toBeVisible();
  const root = sidebar.locator('.agent-workspace-node').filter({ has: page.getByRole('button', { name: 'Expand workspace Project', exact: true }) });
  expect(await root.locator('.agent-workspace-sessions > .agent-workspace-session-row > button:first-child > span > span').allTextContents()).toEqual(['Recent design', 'Old design', 'New conversation']);
  await expect(sidebar.getByRole('button', { name: 'Archived plan', exact: true })).toHaveCount(0);
  async function option(name: string) {
    await toolbar.getByRole('button', { name: 'View options', exact: true }).click();
    const menu = sidebar.getByRole('menu', { name: 'View options', exact: true });
    const bounds = await menu.boundingBox();
    expect(bounds!.x).toBeGreaterThanOrEqual(0);
    expect(bounds!.x + bounds!.width).toBeLessThanOrEqual(page.viewportSize()!.width);
    await menu.getByRole('menuitemradio', { name, exact: true }).click();
  }
  await option('Workspace order');
  expect(await root.locator('.agent-workspace-sessions > .agent-workspace-session-row > button:first-child > span > span').allTextContents()).toEqual(['Old design', 'Recent design', 'New conversation']);
  await option('Workspace tree');
  await expect(root.locator('.agent-workspace-node').getByRole('button', { name: 'App', exact: true })).toBeVisible();
  await sidebar.getByRole('button', { name: 'Expand workspace Project', exact: true }).click();
  await expect(sidebar.getByRole('button', { name: 'Nested notes', exact: true })).toHaveCount(0);
  await toolbar.getByRole('button', { name: 'Search conversations', exact: true }).click();
  const input = sidebar.getByRole('searchbox', { name: 'Search conversations', exact: true });
  await expect(input).toBeFocused();
  await input.fill('design');
  await expect(sidebar.locator('.agent-workspace-search-results').getByRole('button', { name: 'Old design', exact: false })).toBeVisible();
  await expect(sidebar.locator('.agent-workspace-search-results .agent-workspace-session-row')).toHaveCount(2);
  await input.fill('stale'); await expect.poll(() => oldStarted).toBe(true);
  await input.fill('phrase');
  await expect(sidebar.getByText('Unique phrase in conversation history', { exact: true })).toBeVisible();
  releaseOld();
  await expect(sidebar.getByText('obsolete response', { exact: true })).toHaveCount(0);
  await expect(sidebar.locator('.agent-workspace-search-results .agent-workspace-session-row')).toHaveCount(1);
  await input.fill('overflow');
  await expect(sidebar.getByText('Showing the first 20 results. Narrow your search.', { exact: true })).toBeVisible();
  rejectSearch = true; await input.fill('design');
  await expect(sidebar.getByRole('alert')).toContainText('Could not search conversation history.');
  await expect(sidebar.locator('.agent-workspace-search-results .agent-workspace-session-row')).toHaveCount(2);
  await sidebar.getByRole('button', { name: 'Retry', exact: true }).click();
  await expect(sidebar.getByRole('alert')).toHaveCount(0);
  await sidebar.getByRole('button', { name: 'Clear search', exact: true }).click();
  await expect(input).toHaveValue('');
  await input.press('Escape');
  await expect(input).toHaveCount(0);
  await expect(toolbar.getByRole('button', { name: 'Search conversations', exact: true })).toBeFocused();
  await option('In one list');
  await expect(sidebar.locator('.agent-workspace-heading')).toHaveCount(0);
  await expect(sidebar.getByRole('button', { name: 'Nested notes App', exact: true })).toBeVisible();
  await option('All conversations');
  await expect(sidebar.getByRole('button', { name: 'Archived plan Project', exact: true })).toBeDisabled();
  await option('Archived only');
  await expect(sidebar.locator('.agent-workspace-session-row')).toHaveCount(1);
  await sidebar.getByRole('button', { name: 'Unarchive conversation · Archived plan', exact: true }).click();
  await expect(sidebar.locator('.agent-workspace-session-row')).toHaveCount(0);
  expect(mutations).toEqual(['workspace/unarchiveSession']);
  await option('All conversations');
  await page.reload();
  if (testInfo.project.name === 'mobile') await page.getByRole('button', { name: 'Open sidebar', exact: true }).click();
  await expect(sidebar.locator('.agent-workspace-heading')).toHaveCount(0);
  await expect(sidebar.getByRole('button', { name: 'Archived plan Project', exact: true })).toBeEnabled();
  expect(connects).toBe(0);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});
