import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';
import { mkdir } from 'node:fs/promises';
import { createInterface } from 'node:readline';
import { setTimeout as delay } from 'node:timers/promises';

export type RoleId = 'pm' | 'researcher' | 'architect' | 'implementer' | 'evaluator' | 'reviewer' | 'editor';
export interface GenerateOptions {
  key: string;
  instructions: string;
  input: string;
  signal?: AbortSignal;
  onDelta?: (delta: string) => void;
  schema?: Record<string, unknown>;
}
export interface AiProvider {
  status(): Promise<{ ready: boolean; authenticated: boolean; error?: string }>;
  generate(options: GenerateOptions): Promise<{ text: string; threadId?: string; model?: string; usage?: unknown }>;
  close(): Promise<void>;
}
export interface ProviderConfig {
  provider: 'codex' | 'mock';
  codexBin: string;
  codexHome: string;
  workspace: string;
  model?: string;
  timeoutMs: number;
}

type ObjectValue = Record<string, any>;
type Pending = { method: string; resolve: (value: any) => void; reject: (error: Error) => void; timer: NodeJS.Timeout };
type ActiveTurn = {
  threadId: string;
  turnId?: string;
  text: string;
  finalText?: string;
  usage?: unknown;
  onDelta?: (delta: string) => void;
  resolve: (value: { text: string; threadId: string; model?: string; usage?: unknown }) => void;
  reject: (error: Error) => void;
};

function failure(message: string, name = 'Error'): Error {
  const error = new Error(message);
  error.name = name;
  return error;
}
function interrupted(): Error {
  return failure('応答を停止しました。', 'AbortError');
}
function safeRpcError(error: unknown, method?: string): Error {
  // App-server error text can contain URLs, request bodies or authentication material.
  // Return a fixed classification instead of exposing arbitrary subprocess output.
  const text = JSON.stringify(error ?? '').toLowerCase();
  if (/auth|unauthorized|401|login/.test(text)) return failure('Codexの認証が必要です。アプリ専用のCodexへログインしてください。');
  if (/rate.?limit|usage.?limit|429|quota/.test(text)) return failure('Codexの利用上限に達しました。時間をおいて再試行してください。');
  const code = error && typeof error === 'object' && 'code' in error && typeof error.code === 'number' ? error.code : undefined;
  if (code === -32600 || code === -32602) return failure(`Codexの通信形式がCLIと一致しません（${method ?? 'request'} / ${code}）。アプリを更新して再起動してください。`);
  return failure(`Codexへの要求に失敗しました（${method ?? 'response'}${code !== undefined ? ` / ${code}` : ''}）。CLIの版とアプリ専用設定を確認してください。`);
}

/** A private stdio process: no ports, host auth copy, arbitrary shell, or raw logs. */
class CodexProvider implements AiProvider {
  private process?: ChildProcessWithoutNullStreams;
  private starting?: Promise<void>;
  private pending = new Map<number, Pending>();
  private active = new Map<string, ActiveTurn>();
  private threads = new Map<string, { id: string; instructions: string }>();
  private busyKeys = new Set<string>();
  private nextId = 0;
  private model?: string;
  private closed = false;

  constructor(private readonly config: ProviderConfig) {}

  private async start(): Promise<void> {
    if (this.closed) throw failure('AI接続は終了しています。');
    if (this.starting) return this.starting;
    if (this.process) return;
    this.starting = this.launch();
    try { await this.starting; } finally { this.starting = undefined; }
  }

  private async launch(): Promise<void> {
    await mkdir(this.config.codexHome, { recursive: true, mode: 0o700 });
    if (this.closed) throw failure('AI接続は終了しています。');
    const disabled = [
      'shell_tool', 'unified_exec', 'apps', 'plugins', 'remote_plugin', 'hooks',
      'browser_use', 'computer_use', 'image_generation', 'view_image',
      'multi_agent', 'multi_agent_v2', 'skill_search', 'sleep_tool',
    ];
    const args = ['app-server', '--listen', 'stdio://'];
    for (const feature of disabled) args.push('--disable', feature);
    args.push('-c', 'web_search="disabled"', '-c', 'approval_policy="never"', '-c', 'sandbox_mode="read-only"', '-c', 'cli_auth_credentials_store="file"');
    // A dedicated CODEX_HOME must be used. No parent-process provider tokens or
    // command injection via a shell are passed through by this adapter.
    const env: NodeJS.ProcessEnv = {
      PATH: process.env.PATH,
      HOME: this.config.codexHome,
      CODEX_HOME: this.config.codexHome,
      LANG: 'C.UTF-8',
      TERM: 'dumb',
      ...(process.env.SSL_CERT_FILE ? { SSL_CERT_FILE: process.env.SSL_CERT_FILE } : {}),
      ...(process.env.HTTPS_PROXY ? { HTTPS_PROXY: process.env.HTTPS_PROXY } : {}),
      ...(process.env.HTTP_PROXY ? { HTTP_PROXY: process.env.HTTP_PROXY } : {}),
      ...(process.env.NO_PROXY ? { NO_PROXY: process.env.NO_PROXY } : {}),
    };
    const child = spawn(this.config.codexBin, args, {
      cwd: this.config.workspace, env, stdio: ['pipe', 'pipe', 'pipe'], shell: false,
    });
    this.process = child;
    child.stderr.resume(); // Never relay CLI logs or authentication material to the browser.
    child.stdin.on('error', () => this.disconnected(child));
    child.on('error', () => this.disconnected(child));
    child.on('close', () => this.disconnected(child));
    const lines = createInterface({ input: child.stdout });
    lines.on('line', (line) => {
      if (this.process !== child) return;
      try { this.receive(JSON.parse(line)); } catch { /* Ignore non-protocol log lines. */ }
    });
    try {
      await this.rpc('initialize', { clientInfo: { name: 'research-discussion', title: '研究相談', version: '0.1.0' } });
      this.send({ method: 'initialized', params: {} });
    } catch (error) {
      this.terminate();
      throw error;
    }
  }

  private disconnected(child: ChildProcessWithoutNullStreams): void {
    if (this.process !== child) return;
    this.process = undefined;
    const error = failure('Codexとの接続が終了しました。再試行してください。');
    for (const pending of this.pending.values()) { clearTimeout(pending.timer); pending.reject(error); }
    this.pending.clear();
    for (const turn of this.active.values()) turn.reject(error);
    this.active.clear();
    // A new process gets new threads. Complete context is supplied by the backend.
    this.threads.clear();
  }

  private send(message: ObjectValue): void {
    if (!this.process || this.process.stdin.destroyed) throw failure('Codexに接続できません。');
    this.process.stdin.write(`${JSON.stringify(message)}\n`);
  }

  private rpc(method: string, params: ObjectValue): Promise<any> {
    const id = ++this.nextId;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(failure('Codexへの接続がタイムアウトしました。'));
      }, Math.min(this.config.timeoutMs, 30_000));
      this.pending.set(id, { method, resolve, reject, timer });
      try { this.send({ id, method, params }); }
      catch (error) { clearTimeout(timer); this.pending.delete(id); reject(error); }
    });
  }

  private receive(message: ObjectValue): void {
    if (message.id !== undefined && message.method) {
      this.rejectServerRequest(message);
      return;
    }
    if (message.id !== undefined) {
      const pending = this.pending.get(message.id);
      if (!pending) return;
      this.pending.delete(message.id);
      clearTimeout(pending.timer);
      if (message.error) pending.reject(safeRpcError(message.error, pending.method));
      else pending.resolve(message.result);
      return;
    }
    const params = message.params ?? {};
    const active = this.active.get(params.threadId);
    if (!active) return;
    if (params.turnId && active.turnId && params.turnId !== active.turnId) return;
    if (message.method === 'turn/started' && params.turn?.id) active.turnId = params.turn.id;
    if (message.method === 'item/agentMessage/delta' && typeof params.delta === 'string') {
      if (active.text.length + params.delta.length > 32_000) {
        active.reject(failure('応答の長さが上限に達しました。質問を分けて再試行してください。'));
        if (active.turnId) void this.rpc('turn/interrupt', { threadId: active.threadId, turnId: active.turnId }).catch(() => {});
        return;
      }
      active.text += params.delta;
      active.onDelta?.(params.delta);
    }
    if (message.method === 'item/completed' && params.item?.type === 'agentMessage') {
      const item = params.item;
      if (item.phase === 'final_answer' || !active.finalText) active.finalText = item.text;
    }
    if (message.method === 'thread/tokenUsage/updated') active.usage = params.tokenUsage;
    if (message.method === 'turn/completed') {
      if (active.turnId && params.turn?.id !== active.turnId) return;
      if (params.turn?.status === 'interrupted') active.reject(interrupted());
      else if (params.turn?.status === 'failed') active.reject(safeRpcError(params.turn.error));
      else active.resolve({ text: active.finalText || active.text, threadId: active.threadId, model: this.model, usage: active.usage });
    }
    // Reasoning, tool inputs and tool output are intentionally never forwarded.
  }

  private rejectServerRequest(message: ObjectValue): void {
    const method = String(message.method);
    if (method === 'item/permissions/requestApproval') {
      this.send({ id: message.id, result: { permissions: {}, scope: 'turn' } });
    } else if (/requestApproval$|execCommandApproval$|applyPatchApproval$/.test(method)) {
      this.send({ id: message.id, result: { decision: 'decline' } });
    } else if (method === 'item/tool/call') {
      this.send({ id: message.id, result: { contentItems: [{ type: 'inputText', text: 'このアプリではツール実行は許可されていません。' }], success: false } });
    } else if (method === 'item/tool/requestUserInput') {
      this.send({ id: message.id, result: { answers: {} } });
    } else if (/elicitation/.test(method)) {
      this.send({ id: message.id, result: { action: 'decline', content: null } });
    } else {
      this.send({ id: message.id, error: { code: -32601, message: 'Unsupported server request' } });
    }
  }

  async status(): Promise<{ ready: boolean; authenticated: boolean; error?: string }> {
    try {
      await this.start();
      const result = await this.rpc('account/read', { refreshToken: false });
      const authenticated = Boolean(result.account) || result.requiresOpenaiAuth === false;
      return authenticated
        ? { ready: true, authenticated: true }
        : { ready: false, authenticated: false, error: 'アプリ専用のCodexへログインしてください。' };
    } catch (error) {
      return { ready: false, authenticated: false, error: error instanceof Error ? error.message : 'Codexに接続できません。' };
    }
  }

  private async chooseModel(): Promise<string> {
    if (this.model) return this.model;
    if (this.config.model) { this.model = this.config.model; return this.model; }
    const result = await this.rpc('model/list', { limit: 100 });
    const models = (Array.isArray(result.data) ? result.data : []) as ObjectValue[];
    const selected = models.find((model) => model.isDefault && !model.hidden) ?? models.find((model) => !model.hidden);
    if (!selected || typeof selected.model !== 'string') throw failure('利用可能なCodexモデルがありません。');
    this.model = selected.model;
    return this.model;
  }

  async generate(options: GenerateOptions): Promise<{ text: string; threadId?: string; model?: string; usage?: unknown }> {
    if (options.signal?.aborted) throw interrupted();
    if (this.busyKeys.has(options.key)) throw failure('この担当は応答中です。完了を待ってください。');
    this.busyKeys.add(options.key);
    let active: ActiveTurn | undefined;
    let timer: NodeJS.Timeout | undefined;
    let abort: (() => void) | undefined;
    try {
      await this.start();
      if (options.signal?.aborted) throw interrupted();
      const model = await this.chooseModel();
      const settings = {
        cwd: this.config.workspace, approvalPolicy: 'never', sandbox: 'read-only', model,
        developerInstructions: options.instructions,
      };
      let thread = this.threads.get(options.key);
      if (thread && thread.instructions === options.instructions) {
        await this.rpc('thread/resume', { ...settings, threadId: thread.id, excludeTurns: true });
      } else {
        const result = await this.rpc('thread/start', settings);
        if (typeof result.thread?.id !== 'string') throw failure('Codexの会話を開始できませんでした。');
        thread = { id: result.thread.id, instructions: options.instructions };
        this.threads.set(options.key, thread);
      }
      if (options.signal?.aborted) { this.threads.delete(options.key); throw interrupted(); }
      const threadId = thread.id;
      const completed = new Promise<{ text: string; threadId: string; model?: string; usage?: unknown }>((resolve, reject) => {
        active = { threadId, text: '', onDelta: options.onDelta, resolve, reject };
        this.active.set(threadId, active);
      });
      // Register a handler before awaiting turn/start: the server can stream and
      // even complete before it returns the start acknowledgement.
      void completed.catch(() => {});
      let cancelled = false;
      const cancel = (error: Error) => {
        cancelled = true;
        if (!active) return;
        this.threads.delete(options.key);
        if (active.turnId) void this.rpc('turn/interrupt', { threadId, turnId: active.turnId }).catch(() => {});
        active.reject(error);
        if (!active.turnId) this.terminate();
      };
      abort = () => cancel(interrupted());
      options.signal?.addEventListener('abort', abort, { once: true });
      timer = setTimeout(() => {
        cancel(failure('応答時間の上限に達しました。必要に応じて再試行してください。', 'TimeoutError'));
      }, this.config.timeoutMs);
      // CLI 0.163.0 rejects legacy readOnly.access, even though some online
      // examples still show it. Tools are disabled at launch and approvals denied.
      const started = this.rpc('turn/start', {
        threadId, input: [{ type: 'text', text: options.input }],
        approvalPolicy: 'never', sandboxPolicy: { type: 'readOnly' },
        ...(options.schema ? { outputSchema: options.schema } : {}),
      }).then((response) => {
        if (active && response.turn?.id) active.turnId = response.turn.id;
        if (cancelled && active?.turnId) {
          void this.rpc('turn/interrupt', { threadId, turnId: active.turnId }).catch(() => {});
        }
        return completed;
      }, (error) => {
        // A failed acknowledgement may leave the server running a turn whose
        // ID is unknown. End this process instead of silently orphaning work.
        active?.reject(error);
        this.terminate();
        throw error;
      });
      return await Promise.race([completed, started]);
    } catch (error) {
      this.threads.delete(options.key);
      throw error;
    } finally {
      if (timer) clearTimeout(timer);
      if (abort) options.signal?.removeEventListener('abort', abort);
      if (active) this.active.delete(active.threadId);
      this.busyKeys.delete(options.key);
    }
  }

  private terminate(): void {
    const child = this.process;
    if (!child) return;
    this.disconnected(child);
    child.kill('SIGTERM');
    const killTimer = setTimeout(() => { if (child.exitCode === null) child.kill('SIGKILL'); }, 1_000);
    killTimer.unref();
  }

  async close(): Promise<void> {
    this.closed = true;
    this.terminate();
  }
}

function mockValue(schema: ObjectValue, key = '', input = ''): unknown {
  if (Array.isArray(schema.enum)) return schema.enum[0];
  if (schema.type === 'object' || schema.properties) {
    return Object.fromEntries(Object.entries(schema.properties ?? {}).map(([name, value]) => [name, mockValue(value as ObjectValue, name, input)]));
  }
  if (schema.type === 'array') return /participants|recipients|roles|agents/.test(key) && input.includes('[DISCUSS=true]') ? ['evaluator', 'reviewer'] : [];
  if (schema.type === 'boolean') return false;
  if (schema.type === 'number' || schema.type === 'integer') return 0;
  return /content|markdown/.test(key)
    ? '## 未承認の提案（動作確認用）\n\nこれはmockモードの応答です。研究上の結論・合意・実測結果を示しません。\n\n## 未確認事項\n\n実際の研究相談はCodex接続後に行ってください。'
    : '動作確認用の下書き';
}

class MockProvider implements AiProvider {
  private closed = false;
  async status() { return { ready: !this.closed, authenticated: !this.closed }; }
  async generate(options: GenerateOptions) {
    if (this.closed) throw failure('AI接続は終了しています。');
    const role = options.key.split(':').at(-1) ?? 'pm';
    const text = options.schema
      ? JSON.stringify(mockValue(options.schema, '', options.input))
      : `【動作確認用・${role}】ご相談を受け取りました。これはmockモードの応答です。実際の議論にはCodex接続が必要です。`;
    for (const chunk of text.match(/.{1,12}/gs) ?? []) {
      if (options.signal?.aborted) throw interrupted();
      await delay(15, undefined, { signal: options.signal });
      options.onDelta?.(chunk);
    }
    return { text, model: 'mock', threadId: `mock:${options.key}` };
  }
  async close() { this.closed = true; }
}

export function createProvider(config: ProviderConfig): AiProvider {
  return config.provider === 'mock' ? new MockProvider() : new CodexProvider(config);
}
