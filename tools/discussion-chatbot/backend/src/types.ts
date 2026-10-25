export const roles = [
  {id: 'pm', label: 'PM', description: '相談相手・進行と論点の整理'},
  {id: 'researcher', label: '調査担当', description: '先行研究と出典の確認'},
  {id: 'architect', label: '設計担当', description: '数理モデルと実装の整合'},
  {id: 'implementer', label: '実装担当', description: 'コードと実装方法の相談'},
  {id: 'evaluator', label: '評価担当', description: '実験条件と評価の検討'},
  {id: 'reviewer', label: 'レビュー担当', description: '主張と根拠の独立した批判的検討'},
  {id: 'editor', label: '編集担当', description: '原稿の論理と文章の確認'},
] as const;
export type Role = typeof roles[number]['id'];
export type RunStatus = 'queued' | 'running' | 'complete' | 'interrupted' | 'failed';
export interface Room { id: string; title: string; reqId: string; createdAt: string; updatedAt: string; paths: string[] }
export interface Message {id: string; roomId: string; role: string; content: string; status: string; createdAt: string; runId?: string}
export interface Run {id: string; roomId: string; messageId: string; status: RunStatus; error?: string; recipients: Role[]; discuss: boolean; createdAt: string; retryOf?: string; completedRoles: Role[]; nextRole?: Role; steps?: Role[]; cursor?: number; sources?: Omit<Source,'text'>[]; startedAt?:string; codeRevision?:string; contextWarning?:string}
export interface Draft {id: string; roomId: string; content: string; target: 'report'|'checkpoint'; status: 'draft'|'saved'; savedPath?: string; createdAt: string; baseline: string; sourceIds: string[]; sourceHash: string; sources?: Omit<Source,'text'>[]; codeRevision?:string; proofreading?:string}
export interface Source {path: string; hash: string; text: string; truncated: boolean; unread?:boolean}
export const now = () => new Date().toLocaleString('sv-SE', {timeZone:'Asia/Tokyo'}).replace(' ', 'T') + '+09:00';
export function requireString(value: unknown, name: string, max: number): string {
  if (typeof value !== 'string' || !value.trim() || value.length > max) throw new Error(`${name}を1〜${max}文字で指定してください。`);
  return value.trim();
}
export function reqId(value: unknown): string {
  const id = requireString(value, 'REQ ID', 20);
  if (!/^REQ-\d{3,}$/.test(id)) throw new Error('REQ-005の形式で指定してください。');
  return id;
}
