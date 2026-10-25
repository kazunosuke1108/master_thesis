export type Role = { id: string; label: string; description: string };
export type Health = { status: string; provider: 'codex' | 'mock'; ai: { ready: boolean; authenticated: boolean; error?: string }; roles: Role[]; version: string };
export type Room = { id: string; title: string; reqId: string; createdAt: string; updatedAt: string; paths?: string[] };
export type Message = { id: string; roomId: string; role: string; content: string; status: 'complete' | 'streaming' | 'interrupted' | 'failed'; createdAt: string; runId?: string };
export type Run = { id: string; status: 'queued' | 'running' | 'complete' | 'interrupted' | 'failed'; error?: string };
export type Draft = { id: string; content: string; target: 'report' | 'checkpoint'; status: 'draft' | 'saved'; savedPath?: string; createdAt: string; proofreading?: string };
export type Source = { path: string; hash?: string; label?: string; truncated?: boolean; automatic?: boolean; unread?: boolean };
export type Snapshot = { room: Room; messages: Message[]; runs: Run[]; drafts: Draft[]; sources: Source[]; contextWarning?: string };
export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...init?.headers },
  });
  const value = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(value.error || `通信に失敗しました（${response.status}）`);
  return value as T;
}
export function post<T>(path: string, value: unknown = {}) {
  return api<T>(path, { method: 'POST', body: JSON.stringify(value) });
}
