import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdir, mkdtemp, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { once } from 'node:events';
import { chromium } from '@playwright/test';

// 参数必须指向专用构建；每次运行新建数据与证据目录，不连接用户的服务。
const [serverPath, webPath, evidenceRoot] = process.argv.slice(2);
assert(serverPath && webPath && evidenceRoot, 'Usage: node e2e/native-capabilities.acceptance.mjs SERVER WEB_ROOT EVIDENCE_ROOT');
await mkdir(resolve(evidenceRoot), { recursive: true });
const output = await mkdtemp(resolve(evidenceRoot, 'run-'));
const server = spawn(resolve(serverPath), ['--port', '0', '--data-dir', resolve(output, 'data'),
  '--web-root', resolve(webPath), '--exit-on-stdin-close'], { windowsHide: true, stdio: ['pipe', 'pipe', 'pipe'] });
let stdout = '';
let stderr = '';
server.stdout.on('data', chunk => { stdout += chunk; });
server.stderr.on('data', chunk => { stderr += chunk; });
const listening = await new Promise((resolveListening, reject) => {
  const timer = setTimeout(() => reject(new Error(`Native server did not start: ${stderr}`)), 15000);
  server.once('error', cause => { clearTimeout(timer); reject(cause); });
  server.once('exit', code => { clearTimeout(timer); reject(new Error(`Native server exited: ${code}; ${stderr}`)); });
  server.stdout.on('data', () => {
    const event = stdout.split('\n').find(line => line.includes('"event":"listening"'));
    if (event) { clearTimeout(timer); resolveListening(JSON.parse(event)); }
  });
}).catch(cause => { server.stdin.end(); throw cause; });
const baseURL = `http://127.0.0.1:${listening.port}`;
let browser;
const requests = [];
const httpErrors = [];
const consoleErrors = [];
const report = { baseURL, output, checks: [], requests, httpErrors, consoleErrors,
  inference: 'Not tested: this isolated infrastructure build has no GPU inference backend.' };
try {
  browser = await chromium.launch();
  const context = await browser.newContext({ locale: 'en-US', viewport: { width: 1440, height: 1000 } });
  const page = await context.newPage();
  page.setDefaultTimeout(15000);
  page.setDefaultNavigationTimeout(15000);
  page.on('request', request => requests.push({ method: request.method(), url: request.url() }));
  page.on('response', response => { if (response.status() >= 400) httpErrors.push({ status: response.status(), url: response.url() }); });
  page.on('pageerror', error => consoleErrors.push(error.message));
  await page.addInitScript(() => {
    localStorage.setItem('tyloinfer.studio.generation.v1', JSON.stringify({ language: 'en' }));
    window.cspViolations = [];
    document.addEventListener('securitypolicyviolation', event => window.cspViolations.push({
      blockedURI: event.blockedURI, directive: event.effectiveDirective,
    }));
  });
  const initial = await page.goto(`${baseURL}/chat`);
  report.csp = initial.headers()['content-security-policy'];
  assert(report.csp.includes("connect-src 'self' https://api.github.com ws: wss:;"));
  await page.getByRole('textbox', { name: 'Message', exact: true }).waitFor();
  report.checks.push('Text chat route renders against the native service with no loaded model');
  await page.screenshot({ path: resolve(output, 'chat.png') });

  await page.goto(`${baseURL}/models`);
  await page.getByRole('button', { name: 'Add model', exact: true }).waitFor();
  await page.getByRole('button', { name: 'Add model', exact: true }).click();
  await page.getByRole('dialog').waitFor();
  await page.getByRole('textbox', { name: 'Current directory' }).waitFor();
  await page.screenshot({ path: resolve(output, 'model-directory.png') });
  report.checks.push('Model management opens the real native directory browser');

  for (const [path, detail] of [['resources', 'resource monitoring or cache management'],
    ['analysis', 'architecture or tensor analysis'], ['evaluations', 'datasets, quality evaluations or benchmark jobs']]) {
    await page.goto(`${baseURL}/${path}`);
    await page.getByText(new RegExp(detail)).waitFor();
    await page.screenshot({ path: resolve(output, `${path}.png`) });
  }
  await page.goto(`${baseURL}/runtime`);
  await page.getByRole('combobox', { name: 'Authentication', exact: true }).waitFor();
  assert(await page.getByRole('spinbutton', { name: 'OpenAI port' }).isDisabled());
  await page.getByRole('tab', { name: 'Cache', exact: true }).click();
  await page.getByText('Cache settings are unavailable on the native server.').waitFor();
  await page.getByRole('tab', { name: 'Advanced', exact: true }).click();
  await page.getByText('Cluster nodes and remote routing are unavailable on the native server.').waitFor();
  await page.screenshot({ path: resolve(output, 'runtime.png') });

  await page.goto(`${baseURL}/model-hub`);
  await page.getByRole('textbox', { name: 'Search models' }).waitFor();
  assert(await page.getByRole('button', { name: 'Search', exact: true }).isDisabled());
  await page.getByRole('tab', { name: 'Recommended models', exact: true }).click();
  await page.getByText(/does not provide recommendations/).waitFor();
  await page.waitForTimeout(6500);
  const unsupported = /\/api\/v1\/(?:runtime\/(?:listener|memory-policy|resources|cache|serving|context-policy|inference-policy)|cluster|datasets|evaluations|hub\/(?:official|models\?|download-reuse)|models\/[^/]+\/(?:analysis|cache-profile|deletion|delete))/;
  assert.deepEqual(requests.filter(request => unsupported.test(request.url)), []);
  assert.deepEqual(httpErrors.filter(response => response.url.startsWith(baseURL)), []);
  assert.deepEqual(consoleErrors, []);
  report.checks.push('Unsupported routes and settings render explicit notices with no unsupported requests or local HTTP errors');

  await page.goto(`${baseURL}/versions`);
  await page.getByRole('heading', { name: 'Version manager', exact: true }).waitFor();
  report.github = await page.evaluate(async () => {
    try {
      const response = await fetch('https://api.github.com/repos/Tylogi/TyloInfer/releases?per_page=100', { signal: AbortSignal.timeout(15000) });
      return { status: response.status, reachedNetwork: true };
    } catch (cause) { return { error: String(cause), reachedNetwork: false }; }
  });
  assert.deepEqual(await page.evaluate(() => window.cspViolations), []);
  report.checks.push('GitHub version check is not blocked by the local CSP');
  // 负向来源不发实际网络请求；应在 CSP 层阻止。
  await page.evaluate(async () => { await fetch('https://example.invalid/csp-probe').catch(() => {}); });
  await page.waitForFunction(() => window.cspViolations.some(event => event.blockedURI === 'https://example.invalid/csp-probe'));
  report.checks.push('An unrelated external HTTP origin remains blocked by CSP');
  await page.screenshot({ path: resolve(output, 'versions.png') });

  // 用同一个隔离服务的另一主机名验证跨来源跳转，不启动或探测其它用户服务。
  await page.goto(`${baseURL}/runtime`);
  await page.getByRole('combobox', { name: 'Service connection', exact: true }).selectOption('remote');
  const remote = page.getByRole('textbox', { name: 'Remote endpoint', exact: true });
  await remote.fill(`http://localhost:${listening.port}/prefix/v1`);
  await remote.press('Enter');
  await page.getByRole('alert').filter({ hasText: 'Path prefixes are not supported.' }).waitFor();
  assert.equal(page.url(), `${baseURL}/runtime`);
  report.checks.push('Unsupported web URL prefixes produce a clear error without navigating');
  await remote.fill(`http://localhost:${listening.port}`);
  await remote.press('Enter');
  await page.waitForURL(`http://localhost:${listening.port}/runtime`);
  await page.getByRole('combobox', { name: 'Authentication', exact: true }).waitFor();
  assert.deepEqual(httpErrors.filter(response => new URL(response.url).port === String(listening.port)), []);
  report.checks.push('A user-selected remote web connection navigates to its own origin without broadening CSP');
  report.status = 'passed';
} catch (cause) {
  report.status = 'failed';
  report.error = cause.stack || String(cause);
  throw cause;
} finally {
  await browser?.close();
  if (server.exitCode === null) {
    const closed = once(server, 'exit');
    server.stdin.end();
    await closed;
  }
  await writeFile(resolve(output, 'server.stdout.log'), stdout);
  await writeFile(resolve(output, 'server.stderr.log'), stderr);
  await writeFile(resolve(output, 'report.json'), JSON.stringify(report, null, 2));
  console.log(JSON.stringify({ status: report.status, output, checks: report.checks, github: report.github }));
}
