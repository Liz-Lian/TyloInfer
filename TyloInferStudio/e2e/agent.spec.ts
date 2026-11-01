import { expect, test } from '@playwright/test';
import { mockStudioServer } from './mockServer';
import type { AgentState } from '../src/features/agent/AgentProvider';

test('agent navigation, workspaces and appearance replace applications and settings', async ({ page }, testInfo) => {
  await mockStudioServer(page);
  const state: AgentState = { running: true, workspaces: [{ workspaceId: 'workspace-1', title: 'Project', path: '/workspace/project', sessionIds: ['session-1'] }],
    sessions: [{ sessionId: 'session-1', running: false, blank: false, projections: { values: { title: 'Review changes' } } }],
    models: [{ id: 'test-model', name: 'Test model' }], approvals: [] };
  await page.route('**/api/v1/agent/state', route => route.fulfill({ json: state }));
  await page.route('**/api/v1/agent/start', route => route.fulfill({ json: state }));
  await page.route('**/api/v1/agent/workspaces/workspace-1/session', route => {
    if (!state.sessions.some(item => item.sessionId === 'blank-1')) {
      state.sessions.push({ sessionId: 'blank-1', running: false, blank: true });
      state.workspaces[0].sessionIds.push('blank-1');
    }
    return route.fulfill({ json: { sessionId: 'blank-1' } });
  });
  const calls: { method: string; args: Record<string, unknown> }[] = [];
  await page.route('**/api/v1/agent/rpc', route => {
    const body = route.request().postDataJSON();
    calls.push(body);
    return route.fulfill({ json: body.method === 'session/create' ? { sessionId: 'session-2' } : {} });
  });
  await page.route('**/api/v1/agent/sessions/*/events', route => route.fulfill({ contentType: 'application/x-ndjson', body:
    JSON.stringify({ type: 'snapshot', cursor: 2, records: [{ type: 'event', event: { seq: 1, type: 'assistant/message', data: { message: { content: [{ type: 'text', text: 'Workspace restored.' }] } } } }] }) + '\n' }));
  const failures: string[] = [];
  page.on('pageerror', error => failures.push(error.message));
  await page.goto('/agent');
  await expect(page.getByRole('heading', { name: 'Untitled conversation', exact: true })).toBeVisible();
  const title = page.locator('.agent-page-header > div').first();
  await expect(title).toHaveText('Untitled conversation');
  await expect(title.locator('svg, small')).toHaveCount(0);
  if (testInfo.project.name === 'mobile') await page.getByRole('button', { name: 'Open sidebar', exact: true }).click();
  const sidebar = page.locator('#studio-sidebar');
  const newConversation = sidebar.getByRole('button', { name: 'New conversation', exact: true });
  await expect(newConversation).toHaveClass('agent-new-conversation');
  await expect(newConversation).toHaveCSS('border-top-width', '1px');
  await expect(newConversation).toHaveCSS('height', '38px');
  const contentOffset = await newConversation.evaluate(button => {
    const icon = button.querySelector('svg')!.getBoundingClientRect();
    const label = button.querySelector('span')!.getBoundingClientRect();
    const bounds = button.getBoundingClientRect();
    return Math.abs((icon.left + label.right) / 2 - (bounds.left + bounds.right) / 2);
  });
  expect(contentOffset).toBeLessThanOrEqual(1);
  const addWorkspace = sidebar.getByRole('button', { name: 'Add workspace', exact: true });
  await expect(addWorkspace).toHaveCSS('width', '28px');
  await expect(addWorkspace).toHaveCSS('height', '28px');
  await expect(addWorkspace).toHaveText('');
  await expect(sidebar.locator('.agent-workspace-toolbar')).toHaveText('Workspaces');
  await expect(sidebar.locator('.sidebar-section-heading')).toHaveCount(3);
  for (const heading of await sidebar.locator('.sidebar-section-heading').all()) await expect(heading).toHaveCSS('border-bottom-width', '0px');
  await expect(sidebar.locator('nav > section').first()).toHaveCSS('border-top-width', '0px');
  for (const section of await sidebar.locator('nav > section + section').all()) await expect(section).toHaveCSS('border-top-width', '1px');
  expect(await sidebar.locator('.sidebar-group-toggle').evaluate(element => element.lastElementChild?.tagName.toLowerCase())).toBe('svg');
  const caretGap = await sidebar.locator('.sidebar-group-toggle').evaluate(element => element.querySelector('svg')!.getBoundingClientRect().left - element.querySelector('span')!.getBoundingClientRect().right);
  expect(caretGap).toBeGreaterThanOrEqual(0);
  expect(caretGap).toBeLessThanOrEqual(8);
  await expect(sidebar.getByRole('button', { name: 'Tools', exact: true })).toHaveAttribute('aria-expanded', 'true');
  await expect(sidebar.getByRole('button', { name: 'Downloads', exact: true })).toBeVisible();
  await expect(sidebar.getByRole('button', { name: 'Applications', exact: true })).toHaveCount(0);
  await expect(sidebar.getByRole('button', { name: 'Settings', exact: true })).toHaveCount(0);
  await expect(sidebar.getByRole('button', { name: 'Logs', exact: true })).toBeVisible();
  const sections = await sidebar.locator('nav > section').allTextContents();
  expect(sections.at(-1)).toContain('Agent');
  expect(sections[1]).toContain('Logs');
  await sidebar.getByRole('button', { name: 'Review changes', exact: true }).click();
  await expect(page).toHaveURL(/session=session-1/);
  await expect(title).toHaveText('Review changes');
  await expect(title.locator('h1')).toHaveAttribute('title', 'Review changes');
  await expect(page.getByText('Workspace restored.', { exact: true })).toBeVisible();
  await page.getByRole('textbox', { name: 'Message the agent' }).fill('Inspect the project');
  await page.getByRole('button', { name: 'Send', exact: true }).click();
  await expect.poll(() => calls.filter(item => item.method === 'session/prompt').length).toBe(1);
  expect(calls.find(item => item.method === 'session/prompt')?.args).toMatchObject({ request: { sessionId: 'session-1', mode: 'queue', content: [{ type: 'text', text: 'Inspect the project' }] } });
  state.sessions[0].projections!.values!.title = 'Updated review';
  await expect(title).toHaveText('Updated review');
  if (testInfo.project.name === 'mobile') await page.getByRole('button', { name: 'Open sidebar', exact: true }).click();
  await expect(sidebar.locator('.appearance-controls')).toHaveCount(0);
  await newConversation.click();
  await expect(page).toHaveURL(/workspace=workspace-1&session=blank-1/);
  await expect(title).toHaveText('Untitled conversation');
  expect(state.sessions.filter(item => item.blank)).toHaveLength(1);
  expect(failures).toEqual([]);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});

test('adds a DSH workspace through the server folder browser', async ({ page }, testInfo) => {
  const server = await mockStudioServer(page);
  const state: AgentState = { running: true, workspaces: [], sessions: [], models: [], approvals: [] };
  await page.route('**/api/v1/agent/state', route => route.fulfill({ json: state }));
  await page.route('**/api/v1/agent/start', route => route.fulfill({ json: state }));
  let opened = 0;
  await page.route('**/api/v1/agent/workspaces/created/session', route => {
    opened += 1;
    if (!state.sessions.length) {
      state.sessions.push({ sessionId: 'empty-session', running: false, blank: true });
      state.workspaces[0].sessionIds.push('empty-session');
    }
    return route.fulfill({ json: { sessionId: 'empty-session' } });
  });
  await page.route('**/api/v1/agent/sessions/empty-session/events', route => route.fulfill({ contentType: 'application/x-ndjson', body: JSON.stringify({ type: 'snapshot', cursor: 0, records: [] }) + '\n' }));
  await page.route(/\/api\/v1\/models\/directories(?:\?.*)?$/, route => {
    const child = new URL(route.request().url()).searchParams.get('directory_id') === 'projects';
    return route.fulfill({ json: { current_id: child ? 'projects' : 'root', current_name: child ? 'Projects' : '/', current_path: child ? '/Projects' : '/',
      parent_id: child ? 'root' : null, model_file_count: 0, data: child ? [] : [{ id: 'projects', name: 'Projects', model_file_count: 0 }] } });
  });
  const registrations: unknown[] = [];
  await page.route('**/api/v1/agent/rpc', route => {
    const body = route.request().postDataJSON(); registrations.push(body);
    state.workspaces.push({ workspaceId: 'created', title: 'Projects', path: '/Projects', sessionIds: [] });
    return route.fulfill({ json: { workspace: state.workspaces[0], created: true } });
  });
  await page.goto('/agent');
  if (testInfo.project.name === 'mobile') await page.getByRole('button', { name: 'Open sidebar', exact: true }).click();
  await page.getByRole('button', { name: 'Add workspace', exact: true }).click();
  const dialog = page.getByRole('dialog', { name: 'Choose workspace folder' });
  await expect(dialog.getByRole('textbox', { name: 'Current directory' })).toHaveValue('/');
  await dialog.getByRole('button', { name: 'Projects', exact: true }).click();
  await expect(dialog.getByRole('textbox', { name: 'Current directory' })).toHaveValue('/Projects');
  await dialog.getByRole('button', { name: 'Up', exact: true }).click();
  await expect(dialog.getByRole('textbox', { name: 'Current directory' })).toHaveValue('/');
  await dialog.getByRole('button', { name: 'Projects', exact: true }).click();
  await dialog.getByRole('button', { name: 'Select folder', exact: true }).click();
  await expect(dialog).toHaveCount(0);
  expect(registrations).toEqual([{ method: 'workspace/create', args: { request: { path: '/Projects' } } }]);
  await expect(page).toHaveURL(/workspace=created/);
  await expect(page).toHaveURL(/session=empty-session/);
  expect(opened).toBe(1);
  expect(state.models).toHaveLength(0);
  if (testInfo.project.name === 'mobile') await page.getByRole('button', { name: 'Open sidebar', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Open blank conversation · Projects', exact: true })).toBeVisible();
  await page.locator('.agent-workspace-heading').getByRole('button', { name: 'Projects', exact: true }).click();
  await expect(page).toHaveURL(/session=empty-session/);
  expect(state.sessions).toHaveLength(1);
  await page.reload();
  await expect(page.getByRole('heading', { name: 'Untitled conversation', exact: true })).toBeVisible();
  expect(state.sessions).toHaveLength(1);
  expect(server.requests.some(request => request === 'POST /api/v1/models/directories/register')).toBe(false);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});

test('a late workspace response cannot replace a newer workspace selection', async ({ page }, testInfo) => {
  await mockStudioServer(page);
  const state: AgentState = { running: true, workspaces: ['Alpha', 'Beta'].map(title => ({ workspaceId: title.toLowerCase(), title, path: `/${title}`, sessionIds: [] })), sessions: [], models: [], approvals: [] };
  await page.route('**/api/v1/agent/state', route => route.fulfill({ json: state }));
  await page.route('**/api/v1/agent/start', route => route.fulfill({ json: state }));
  let release!: () => void;
  const held = new Promise<void>(resolve => { release = resolve; });
  let started = false;
  await page.route('**/api/v1/agent/workspaces/*/session', async route => {
    const id = route.request().url().includes('/alpha/') ? 'alpha' : 'beta';
    if (id === 'alpha') { started = true; await held; }
    const sessionId = `${id}-blank`;
    state.workspaces.find(item => item.workspaceId === id)!.sessionIds = [sessionId];
    state.sessions.push({ sessionId, blank: true, running: false });
    await route.fulfill({ json: { sessionId } });
  });
  await page.route('**/api/v1/agent/sessions/*/events', route => route.fulfill({ contentType: 'application/x-ndjson', body: JSON.stringify({ type: 'snapshot', cursor: 0, records: [] }) + '\n' }));
  await page.goto('/');
  if (testInfo.project.name === 'mobile') await page.getByRole('button', { name: 'Open sidebar', exact: true }).click();
  await page.getByRole('button', { name: 'Alpha', exact: true }).click();
  await expect.poll(() => started).toBe(true);
  await page.getByRole('combobox', { name: 'Workspace', exact: true }).selectOption('beta');
  await expect(page).toHaveURL(/workspace=beta&session=beta-blank/);
  release();
  await expect.poll(() => state.sessions.length).toBe(2);
  await expect(page).toHaveURL(/workspace=beta&session=beta-blank/);
});

test('expanded workspaces show blank conversations without selecting their folders', async ({ page }, testInfo) => {
  await mockStudioServer(page);
  const state: AgentState = { running: true,
    workspaces: ['Alpha', 'Beta'].map(title => ({ workspaceId: title.toLowerCase(), title, path: `/${title}`, sessionIds: [`${title.toLowerCase()}-blank`, `${title.toLowerCase()}-history`] })),
    sessions: ['alpha', 'beta'].flatMap(id => [
      { sessionId: `${id}-blank`, blank: true, running: false },
      { sessionId: `${id}-history`, blank: false, running: false, projections: { values: { title: `${id} history` } } },
    ]), models: [], approvals: [] };
  await page.route('**/api/v1/agent/state', route => route.fulfill({ json: state }));
  await page.route('**/api/v1/agent/start', route => route.fulfill({ json: state }));
  const opened: string[] = [];
  await page.route('**/api/v1/agent/workspaces/*/session', route => {
    const id = route.request().url().includes('/alpha/') ? 'alpha' : 'beta'; opened.push(id);
    return route.fulfill({ json: { sessionId: `${id}-blank` } });
  });
  await page.route('**/api/v1/agent/sessions/*/events', route => route.fulfill({ contentType: 'application/x-ndjson', body: JSON.stringify({ type: 'snapshot', cursor: 0, records: [] }) + '\n' }));
  await page.goto('/agent');
  await expect(page.getByRole('combobox', { name: 'Workspace', exact: true })).toHaveValue('');
  if (testInfo.project.name === 'mobile') await page.getByRole('button', { name: 'Open sidebar', exact: true }).click();
  const sidebar = page.locator('#studio-sidebar');
  await expect(sidebar.getByRole('button', { name: 'alpha history', exact: true })).toBeVisible();
  await expect(sidebar.getByRole('button', { name: 'beta history', exact: true })).toBeVisible();
  await expect(sidebar.getByRole('button', { name: 'Open blank conversation · Alpha', exact: true })).toBeVisible();
  await expect(sidebar.getByRole('button', { name: 'Open blank conversation · Beta', exact: true })).toBeVisible();
  const alphaToggle = sidebar.getByRole('button', { name: 'Expand workspace Alpha', exact: true });
  await alphaToggle.click();
  await expect(sidebar.getByRole('button', { name: 'Open blank conversation · Alpha', exact: true })).toHaveCount(0);
  await alphaToggle.click();
  await expect(alphaToggle).toHaveAttribute('aria-expanded', 'true');
  await expect(sidebar.getByRole('button', { name: 'Open blank conversation · Alpha', exact: true })).toBeVisible();
  expect(opened).toEqual([]);
  await expect(page).toHaveURL('/agent');
  await sidebar.getByRole('button', { name: 'Alpha', exact: true }).click();
  await expect(page).toHaveURL(/workspace=alpha&session=alpha-blank/);
  if (testInfo.project.name === 'mobile') await page.getByRole('button', { name: 'Open sidebar', exact: true }).click();
  await expect(sidebar.getByRole('button', { name: 'Open blank conversation · Alpha', exact: true })).toBeVisible();
  await expect(sidebar.getByRole('button', { name: 'Open blank conversation · Beta', exact: true })).toBeVisible();
  const betaToggle = sidebar.getByRole('button', { name: 'Expand workspace Beta', exact: true });
  await betaToggle.click(); await betaToggle.click();
  await expect(sidebar.getByRole('button', { name: 'Open blank conversation · Beta', exact: true })).toBeVisible();
  expect(opened).toEqual(['alpha']);
  await sidebar.getByRole('button', { name: 'Open blank conversation · Beta', exact: true }).click();
  await expect(page).toHaveURL(/workspace=beta&session=beta-blank/);
  expect(opened).toEqual(['alpha']);
});

test('DSH composer persists thinking, mode, permissions and MTP with real request shapes', async ({ page }) => {
  await mockStudioServer(page);
  const state: AgentState = { running: true,
    workspaces: [{ workspaceId: 'project', title: 'Project', path: '/project', sessionIds: ['first'] }],
    sessions: [{ sessionId: 'first', blank: true, running: false }], models: [
      { id: 'model-a', name: 'Model A', input: ['text', 'image'], reasoningEfforts: { off: 'none', low: 'low', high: 'high' }, mtpAvailable: true },
      { id: 'model-b', name: 'Model B', input: ['text'], mtpAvailable: false }], approvals: [] };
  const controls = { values: { modelSelection: { next: { provider: 'tyloinfer', model: 'model-a', reasoningEffort: undefined as string | undefined } },
    plan: { active: false, pending: null }, permissions: { currentValue: 'workspace-write' } },
    commands: [{ name: 'plan', description: 'Enter or leave plan mode' }, { name: 'goal', description: 'Set a goal', input: { hint: '<objective>' } }],
    permissions: ['read-only', 'workspace-write', 'danger-full-access'].map(value => ({ value, name: value })) };
  await page.route('**/api/v1/agent/state', route => route.fulfill({ json: state }));
  await page.route('**/api/v1/agent/start', route => route.fulfill({ json: state }));
  await page.route('**/api/v1/agent/sessions/first/controls', route => route.fulfill({ json: controls }));
  await page.route('**/api/v1/agent/sessions/first/events', route => route.fulfill({ contentType: 'application/x-ndjson', body: JSON.stringify({ type: 'snapshot', cursor: 0, records: [] }) + '\n' }));
  const calls: { method: string; args: { request: Record<string, unknown> } }[] = [];
  let rejectSelection = false;
  await page.route('**/api/v1/agent/rpc', route => {
    const body = route.request().postDataJSON(); calls.push(body);
    if (body.method === 'session/selectModel') {
      if (rejectSelection) { rejectSelection = false; return route.fulfill({ status: 400, json: { error: { code: 'agent_operation_failed', message: 'Model selection refused', retryable: true, details: {} } } }); }
      const { sessionId: _, ...selection } = body.args.request; controls.values.modelSelection.next = selection;
    }
    return route.fulfill({ json: {} });
  });
  const commands: string[] = [];
  await page.route('**/api/v1/agent/sessions/first/commands', route => {
    const line = route.request().postDataJSON().line; commands.push(line);
    if (line.startsWith('/permission ')) controls.values.permissions.currentValue = line.slice('/permission '.length);
    if (line === '/plan' || line === '/plan off') controls.values.plan.active = line === '/plan';
    return route.fulfill({ json: { result: { kind: 'success', text: 'Applied' } } });
  });
  let uploaded = '';
  await page.route('**/api/v1/agent/sessions/first/files?*', route => {
    uploaded = route.request().postData() || '';
    return route.fulfill({ json: { receiptId: 'uploaded-file', file: { name: 'note.txt', bytes: 4 } } });
  });
  await page.goto('/agent?workspace=project&session=first');
  const composer = page.locator('.agent-composer');
  await composer.getByRole('button', { name: 'MTP · Auto', exact: true }).click();
  await page.getByRole('menuitemradio', { name: 'Off', exact: true }).click();
  await expect(composer.getByRole('button', { name: 'MTP · Off', exact: true })).toBeEnabled();
  await composer.getByRole('button', { name: 'Model A', exact: true }).click();
  await page.getByRole('menuitemradio', { name: 'High', exact: true }).click();
  await expect.poll(() => controls.values.modelSelection.next.reasoningEffort).toBe('high');
  await expect(composer.getByRole('button', { name: 'Model A', exact: true })).toBeEnabled();
  await composer.getByRole('button', { name: 'Agent', exact: true }).click();
  await page.getByRole('menuitemradio', { name: 'Plan', exact: true }).click();
  await expect(composer.getByRole('button', { name: 'Plan', exact: true })).toBeEnabled();
  await composer.getByRole('button', { name: 'Workspace access', exact: true }).click();
  const permissionMenu = page.getByRole('menu', { name: 'Workspace access', exact: true });
  await expect(composer.getByRole('button', { name: 'Workspace access', exact: true }).locator('.agent-permission-shield')).toBeVisible();
  for (const option of await permissionMenu.getByRole('menuitemradio').all()) {
    await expect(option.locator('.agent-permission-shield')).toBeVisible();
    await expect(option.locator('.agent-permission-shield')).toHaveAttribute('width', '16');
  }
  const shieldPaths = await permissionMenu.locator('.agent-permission-shield').evaluateAll(icons => icons.map(icon => icon.innerHTML));
  expect(new Set(shieldPaths).size).toBe(3);
  await expect(permissionMenu.getByRole('menuitemradio', { name: 'Read only', exact: true }).locator('svg')).toHaveAttribute('data-permission', 'read-only');
  await expect(permissionMenu.getByRole('menuitemradio', { name: 'Full access', exact: true }).locator('svg')).toHaveAttribute('data-permission', 'danger-full-access');
  await page.getByRole('menuitemradio', { name: 'Full access', exact: true }).click();
  const confirmation = page.getByRole('dialog', { name: 'Allow full access?' });
  await expect(confirmation).toBeVisible();
  expect(commands).not.toContain('/permission danger-full-access');
  await confirmation.getByRole('button', { name: 'Cancel', exact: true }).click();
  await composer.getByRole('button', { name: 'Workspace access', exact: true }).click();
  await page.getByRole('menuitemradio', { name: 'Read only', exact: true }).click();
  await expect(composer.getByRole('button', { name: 'Read only', exact: true })).toBeEnabled();
  await expect(composer.getByRole('button', { name: 'Read only', exact: true }).locator('.agent-permission-shield')).toHaveAttribute('data-permission', 'read-only');
  await composer.getByRole('button', { name: 'Add and commands', exact: true }).click();
  const menu = page.getByRole('menu', { name: 'Add and commands', exact: true });
  await expect(menu.getByRole('menuitem', { name: 'Attach files or images', exact: true })).toBeVisible();
  const bounds = await menu.boundingBox();
  expect(bounds!.x).toBeGreaterThanOrEqual(0);
  expect(bounds!.x + bounds!.width).toBeLessThanOrEqual(page.viewportSize()!.width);
  await menu.getByRole('textbox', { name: 'Search commands' }).fill('goal');
  await menu.getByRole('menuitem', { name: '/goal Set a goal', exact: true }).click();
  await expect(composer.getByRole('textbox', { name: 'Message the agent' })).toHaveValue('/goal ');
  await composer.getByRole('textbox', { name: 'Message the agent' }).fill('/goal Ship the feature');
  await composer.getByRole('button', { name: 'Send', exact: true }).click();
  await expect.poll(() => commands.includes('/goal Ship the feature')).toBe(true);
  expect(calls.filter(item => item.method === 'session/prompt')).toHaveLength(0);
  await expect(composer.getByRole('button', { name: 'Add and commands', exact: true })).toBeEnabled();
  await composer.getByLabel('Attach files', { exact: true }).setInputFiles({ name: 'note.txt', mimeType: 'text/plain', buffer: Buffer.from('note') });
  await expect(composer.getByRole('button', { name: 'Remove attachment note.txt' })).toBeVisible();
  expect(uploaded).toBe('note');
  await composer.getByRole('textbox', { name: 'Message the agent' }).fill('Inspect the attached note');
  await composer.getByRole('button', { name: 'Send', exact: true }).click();
  await expect.poll(() => calls.filter(item => item.method === 'session/prompt').length).toBe(1);
  expect(calls.find(item => item.method === 'session/prompt')?.args.request.content).toEqual([
    { type: 'text', text: 'Inspect the attached note' }, { type: 'file', receiptId: 'uploaded-file' }]);
  expect(calls.filter(item => item.method === 'session/selectModel').at(-1)?.args.request).toMatchObject({ provider: 'tyloinfer-mtp-off', model: 'model-a', reasoningEffort: 'high' });
  await page.reload();
  await expect(composer.getByRole('button', { name: 'MTP · Off', exact: true })).toBeEnabled();
  await expect(composer.getByRole('button', { name: 'Plan', exact: true })).toBeEnabled();
  await expect(composer.getByRole('button', { name: 'Read only', exact: true })).toBeEnabled();
  await composer.getByRole('button', { name: 'Model A', exact: true }).click();
  await expect(page.getByRole('menuitemradio', { name: 'High', exact: true })).toHaveAttribute('aria-checked', 'true');
  rejectSelection = true;
  await page.getByRole('menuitemradio', { name: 'Model B', exact: true }).click();
  await expect(page.locator('.agent-composer-error')).toContainText('Model selection refused');
  await expect(composer.getByRole('button', { name: 'Model A', exact: true })).toBeEnabled();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});
