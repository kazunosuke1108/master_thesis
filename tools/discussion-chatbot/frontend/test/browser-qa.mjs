/** Optional browser QA. Playwright stays outside the application dependencies; see README.md. */
import assert from 'node:assert/strict';
import { mkdir } from 'node:fs/promises';
import { isAbsolute, join } from 'node:path';
import { pathToFileURL } from 'node:url';

const baseURL = process.env.QA_BASE_URL || 'http://127.0.0.1:5173';
const output = process.env.QA_OUTPUT_DIR || '/tmp/req005-browser-qa';
const modulePath = process.env.BROWSER_QA_MODULE;
if (modulePath && !isAbsolute(modulePath)) throw new Error('BROWSER_QA_MODULEにはモジュールファイルの絶対パスを指定してください。');
const { chromium } = await import(modulePath ? pathToFileURL(modulePath).href : 'playwright-core').catch(() => {
  throw new Error('Playwright Coreが見つかりません。test/README.mdの手順で別ディレクトリへ導入し、BROWSER_QA_MODULEを指定してください。');
});
await mkdir(output, { recursive: true });
const browser = await chromium.launch({
  executablePath: process.env.BROWSER_EXECUTABLE || '/usr/bin/google-chrome',
  headless: true,
  args: process.env.QA_NO_SANDBOX === '1' ? ['--no-sandbox'] : [],
});
const time = () => new Date().toISOString();
const roles = [
  { id: 'pm', label: 'PM', description: '相談の窓口' },
  { id: 'evaluator', label: '評価担当', description: '評価の相談' },
  { id: 'reviewer', label: 'レビュー担当', description: '主張を検討' },
];
const room = (title) => ({ id: 'r1', title, reqId: 'REQ-005', createdAt: time(), updatedAt: time(), paths: [] });
const message = (id, role, content) => ({ id, roomId: 'r1', role, content, status: 'complete', createdAt: time() });
const draft = (id, content) => ({ id, content, target: 'report', status: 'draft', createdAt: time() });
const editor = page => page.getByLabel('Markdownの下書き', { exact: true });
const input = page => page.getByLabel('メッセージ', { exact: true });

async function withPage(name, run) {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  page.setDefaultTimeout(10000);
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  try {
    await run(page);
    assert.deepEqual(errors, [], 'ブラウザ内の実行エラー');
    console.log(`PASS: ${name}`);
  } catch (error) {
    await page.screenshot({ path: join(output, `${name}-failure.png`), fullPage: true }).catch(() => {});
    throw error;
  } finally { await page.close(); }
}

async function mockAPI(page, state) {
  await page.route('**/api/**', async route => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    const method = request.method();
    const data = request.postDataJSON();
    const json = value => route.fulfill({ contentType: 'application/json', body: JSON.stringify(value) });
    if (path === '/api/health') return json({ status: 'ok', provider: 'mock', ai: { ready: true, authenticated: true }, roles, version: 'qa' });
    if (path === '/api/sources') return json({ sources: [{ path: 'reports/example.md', label: '議論資料' }] });
    if (path === '/api/rooms' && method === 'GET') return json({ rooms: state.rooms });
    if (path === '/api/rooms' && method === 'POST') { const value = room(data.title); state.rooms.push(value); return json({ room: value }); }
    if (path === '/api/rooms/r1') return json({ room: state.rooms[0], messages: state.messages, runs: [], drafts: state.drafts, sources: state.sources });
    if (path.endsWith('/events')) return route.fulfill({ contentType: 'text/event-stream', body: ': connected\n\n' });
    if (path.endsWith('/messages')) {
      state.sendAttempts++;
      state.messages.push(message('m1', 'user', data.content), message('m2', 'pm', '## 一緒に考えましょう\n\nUI検証用の返答です。\n\n<script>window.unsafe=true</script>\n\n![参照図](https://tracker.invalid/pixel?context=test)'));
      return json({ runId: 'run1', message: state.messages[0] });
    }
    if (path.endsWith('/sources')) {
      state.rooms[0].paths = data.paths;
      state.sources = data.paths.map(path => ({ path, hash: '123' }));
      return json({ sources: state.sources });
    }
    if (path.endsWith('/drafts')) {
      const value = draft('d1', '# 議論メモ\n\nこれは下書きです。');
      state.drafts.push(value);
      return json({ draft: value });
    }
    if (path === '/api/drafts/d1/save') {
      await new Promise(resolve => setTimeout(resolve, state.saveDelay || 0));
      state.drafts[0] = { ...state.drafts[0], content: data.content, status: 'saved', savedPath: 'reports/example.md' };
      return json({ draft: state.drafts[0] });
    }
    throw new Error(`想定外の要求: ${method} ${path}`);
  });
}

async function mobileCheck(page, name, expectedMessage) {
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
  await page.screenshot({ path: join(output, `${name}-desktop.png`), fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  // Open and close the panel, then reselect the current room. This guards against clearing its snapshot.
  if (await page.locator('.detail-panel.is-open').count()) await page.getByRole('button', { name: 'メモと資料を閉じる' }).click();
  await page.getByRole('button', { name: 'メモと資料を開く' }).click();
  await page.getByRole('button', { name: '参照資料', exact: true }).click();
  await page.getByRole('button', { name: 'メモと資料を閉じる' }).click();
  await page.getByRole('button', { name: '話題一覧を開く' }).click();
  await page.getByRole('navigation', { name: '話題一覧' }).locator('.room-link.selected').click();
  await page.waitForTimeout(300); // Wait for the navigation closing transition before taking a screenshot.
  await expectedMessage.waitFor();
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
  await page.screenshot({ path: join(output, `${name}-mobile.png`), fullPage: true });
}

async function mockFlow(page) {
  const state = { rooms: [], messages: [], drafts: [], sources: [], sendAttempts: 0 };
  await mockAPI(page, state);
  const externalImages = [];
  page.on('request', request => { if (request.url().includes('tracker.invalid')) externalImages.push(request.url()); });
  await page.goto(baseURL);
  await page.getByRole('heading', { name: /研究のこと/ }).waitFor();
  await page.getByRole('button', { name: '最初の話題をつくる' }).click();
  await page.getByLabel('話題の名前').fill('ジャーナル投稿の評価について');
  await page.getByRole('button', { name: '話し始める' }).click();
  await input(page).fill('今の評価で十分かな？');
  await page.getByRole('button', { name: 'メッセージを送信' }).click();
  const rendered = page.getByRole('heading', { name: '一緒に考えましょう' });
  await rendered.waitFor();
  assert.equal(await page.evaluate(() => window.unsafe), undefined);
  assert.deepEqual(externalImages, [], 'Markdownの画像URLが自動取得されていないこと');
  assert.equal(await page.locator('.markdown img').count(), 0);
  assert.equal(state.sendAttempts, 1);
  await page.getByRole('button', { name: '会話から下書きをつくる' }).click();
  await editor(page).fill('# 議論メモ\n\n確認した要点です。');
  await page.getByRole('button', { name: 'プレビュー', exact: true }).click();
  await page.getByRole('heading', { name: '議論メモ' }).waitFor();
  await page.getByRole('button', { name: '内容を確認して保存' }).click();
  await page.getByText('reports/example.md', { exact: true }).waitFor();
  assert.equal(state.drafts[0].content, '# 議論メモ\n\n確認した要点です。');
  await page.getByRole('button', { name: '参照資料', exact: true }).click();
  await page.getByRole('checkbox', { name: /議論資料/ }).check();
  await page.getByRole('button', { name: '選択した資料を反映' }).click();
  await page.waitForFunction(() => document.querySelector('.sources-save')?.disabled);
  assert.equal(state.sources.length, 1);
  await mobileCheck(page, 'mock', rendered);
}

async function draftRace(page) {
  const state = {
    rooms: [room('保存競合の動作確認')], messages: [message('m1', 'user', '相談です')], sources: [],
    drafts: [{ ...draft('d1', '第一の下書き'), proofreading: '検査結果：確認候補があります。' }, draft('d2', '第二の下書き')],
    saveDelay: 800,
  };
  await mockAPI(page, state);
  await page.goto(baseURL);
  const select = page.getByLabel('保存したメモと下書き');
  await select.selectOption('d1');
  await editor(page).fill('第一の編集後の内容');
  await page.getByRole('button', { name: '内容を確認して保存' }).click();
  assert.equal(await select.isDisabled(), true);
  assert.equal(await editor(page).getAttribute('readonly'), '');
  assert.equal(await page.getByRole('button', { name: '会話から下書きをつくる' }).isDisabled(), true);
  await page.getByText('reports/example.md', { exact: true }).waitFor();
  await select.selectOption('d2');
  assert.equal(await editor(page).inputValue(), '第二の下書き');
  await select.selectOption('d1');
  await page.getByText('校正の確認結果', { exact: true }).click();
  await page.getByText('検査結果：確認候補があります。', { exact: true }).waitFor();
}

async function integration(page) {
  if (process.env.QA_TEMP_WORKSPACE_CONFIRMED !== '1') throw new Error('実API確認は専用の一時workspace/dataで実行し、QA_TEMP_WORKSPACE_CONFIRMED=1を指定してください。');
  const health = await (await fetch(new URL('/api/health', baseURL))).json();
  assert.equal(health.provider, 'mock', '実API確認にはmock backendが必要です');
  await page.goto(baseURL);
  await page.getByRole('button', { name: '新しい話題', exact: true }).click();
  const title = `画面動作確認（模擬応答） ${Date.now()}`;
  await page.getByLabel('話題の名前').fill(title);
  await page.getByRole('button', { name: '話し始める' }).click();
  await page.getByRole('heading', { name: title }).waitFor();
  await page.getByLabel('話しかける相手').selectOption('evaluator');
  await input(page).fill('評価担当の画面動作を確認しています。');
  await page.getByRole('button', { name: 'メッセージを送信' }).click();
  const evaluator = page.locator('.message-evaluator.message-complete').first();
  await evaluator.waitFor();
  await page.getByRole('checkbox', { name: 'みんなで議論' }).check();
  await page.getByLabel('話しかける相手').selectOption('pm');
  await input(page).fill('みんなで議論する画面動作を確認します。');
  await page.getByRole('button', { name: 'メッセージを送信' }).click();
  await page.waitForFunction(() => document.querySelectorAll('.message-pm.message-complete').length >= 2);
  await page.getByRole('button', { name: '参照資料', exact: true }).click();
  const source = process.env.QA_SOURCE_PATH || 'reports/qa-reference.md';
  const name = source.split('/').at(-1).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  await page.getByRole('checkbox', { name: new RegExp(name) }).check();
  await page.getByRole('button', { name: '選択した資料を反映' }).click();
  await page.waitForFunction(() => document.querySelector('.sources-save')?.disabled);
  assert.equal(await page.getByRole('alert').count(), 0);
  await page.getByRole('button', { name: '残しておく', exact: true }).click();
  await page.getByRole('button', { name: '会話から下書きをつくる' }).click();
  await editor(page).waitFor();
  assert((await editor(page).inputValue()).includes('未承認'));
  // Never click Save in integration mode: no Markdown report is written to the real workspace.
  await input(page).fill('応答停止の動作確認です。');
  await page.getByRole('button', { name: 'メッセージを送信' }).click();
  await page.getByRole('button', { name: '停止', exact: true }).click();
  const stopped = page.locator('.run-notice').filter({ hasText: '返答を中断しました。' });
  await stopped.waitFor();
  await stopped.getByRole('button', { name: '再試行' }).click();
  await page.waitForFunction(() => !document.querySelector('.run-bar'));
  await page.reload();
  await page.getByRole('heading', { name: title }).waitFor();
  await evaluator.waitFor();
  await mobileCheck(page, 'integration', evaluator);
  console.log('統合確認のルームID:', await page.evaluate(() => localStorage.getItem('research-room-selected')));
}

try {
  await withPage('mock-flow', mockFlow);
  await withPage('draft-save-race', draftRace);
  if (process.env.QA_INTEGRATION === '1') await withPage('integration', integration);
  console.log(`画像出力: ${output}`);
} finally { await browser.close(); }
