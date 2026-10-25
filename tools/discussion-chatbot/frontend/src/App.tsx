import { useCallback, useEffect, useRef, useState } from 'react';
import type { FormEvent, KeyboardEvent } from 'react';
import ReactMarkdown from 'react-markdown';
import { ArrowDown, ArrowUp, BookOpen, Check, ChevronRight, Circle, FileText, FlaskConical, Menu, MessageCircle, Plus, RefreshCw, Save, Square, Users, X } from 'lucide-react';
import { api, post } from './api';
import type { Draft, Health, Message, Role, Room, Snapshot, Source } from './api';

const fallbackRoles: Role[] = [
  { id: 'pm', label: 'PM', description: '相談を受け、必要な担当と一緒に考えます' },
  { id: 'researcher', label: '調査担当', description: '先行研究と根拠を確かめます' },
  { id: 'architect', label: '設計担当', description: '数理モデルと実装の対応を考えます' },
  { id: 'implementer', label: '実装担当', description: 'コードと実装方法を検討します' },
  { id: 'evaluator', label: '評価担当', description: '評価方法と結果の解釈を考えます' },
  { id: 'reviewer', label: '査読担当', description: '主張と根拠を批判的に検討します' },
  { id: 'editor', label: '校正担当', description: '文章の伝わり方を整えます' },
];
const statusLabel = { complete: '', streaming: '返答中', interrupted: '中断', failed: '失敗', queued: '順番待ち', running: '返答中' };
const readableError = (error: unknown) => error instanceof Error ? error.message : '通信に失敗しました。もう一度お試しください。';
const clock = (value: string) => new Date(value).toLocaleTimeString('ja-JP', { hour: '2-digit', minute: '2-digit' });

function Avatar({ role, small = false }: { role: string; small?: boolean }) {
  return <span className={`avatar avatar-${role} ${small ? 'avatar-small' : ''}`} aria-hidden="true">{role === 'user' ? '私' : role === 'pm' ? 'PM' : role.slice(0, 2).toUpperCase()}</span>;
}

function Markdown({ children }: { children: string }) {
  return <div className="markdown"><ReactMarkdown components={{ a: ({ children: text, href }) => <a href={href} target="_blank" rel="noopener noreferrer">{text}</a>, img: ({ alt }) => <span className="markdown-image">［画像：{alt || '説明なし'}・自動表示を省略］</span> }}>{children}</ReactMarkdown></div>;
}

function ChatMessage({ message, roles }: { message: Message; roles: Role[] }) {
  const label = message.role === 'user' ? 'あなた' : roles.find(role => role.id === message.role)?.label || message.role;
  return <article className={`message message-${message.role} message-${message.status}`}>
    <Avatar role={message.role} />
    <div className="message-body">
      <div className="message-meta"><strong>{label}</strong>{message.role !== 'user' && <span className="ai-label">AI</span>}<time dateTime={message.createdAt}>{clock(message.createdAt)}</time>{message.status !== 'complete' && <span className={`message-status ${message.status}`}>{statusLabel[message.status]}</span>}</div>
      {message.content ? <Markdown>{message.content}</Markdown> : message.status === 'streaming' ? <span className="typing" aria-label="返答を準備しています"><i /><i /><i /></span> : <p className="muted">返答はありません。</p>}
    </div>
  </article>;
}

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [rooms, setRooms] = useState<Room[]>([]);
  const [roomId, setRoomId] = useState('');
  const currentRoom = useRef('');
  const snapshotSequence = useRef(0);
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [initialLoading, setInitialLoading] = useState(true);
  const [roomLoading, setRoomLoading] = useState(false);
  const [error, setError] = useState('');
  const [connected, setConnected] = useState(false);
  const [mobileNav, setMobileNav] = useState(false);
  const [panel, setPanel] = useState<'notes' | 'sources'>('notes');
  const [panelOpen, setPanelOpen] = useState(false);
  const [newRoomOpen, setNewRoomOpen] = useState(false);
  const [newTitle, setNewTitle] = useState('');
  const [newReq, setNewReq] = useState('REQ-005');
  const [creating, setCreating] = useState(false);
  const [input, setInput] = useState('');
  const [recipient, setRecipient] = useState('pm');
  const [discuss, setDiscuss] = useState(false);
  const [sending, setSending] = useState(false);
  const sendKey = useRef({ content: '', id: '' });
  const [runBusy, setRunBusy] = useState('');
  const [draftBusy, setDraftBusy] = useState(false);
  const [draftTarget, setDraftTarget] = useState<'report' | 'checkpoint'>('report');
  const [selectedDraft, setSelectedDraft] = useState('');
  const selectedDraftRef = useRef('');
  selectedDraftRef.current = selectedDraft;
  const [draftText, setDraftText] = useState('');
  const [draftPreview, setDraftPreview] = useState(false);
  const [savingDraft, setSavingDraft] = useState(false);
  const [availableSources, setAvailableSources] = useState<Source[]>([]);
  const [sourcePaths, setSourcePaths] = useState<string[]>([]);
  const [sourceDirty, setSourceDirty] = useState(false);
  const [savingSources, setSavingSources] = useState(false);
  const [atBottom, setAtBottom] = useState(true);
  const scrollRef = useRef<HTMLDivElement>(null);
  const composerRef = useRef<HTMLTextAreaElement>(null);
  const roles = health?.roles?.length ? health.roles : fallbackRoles;
  const activeRun = snapshot?.runs.find(run => run.status === 'running');
  const queuedRuns = snapshot?.runs.filter(run => run.status === 'queued') || [];
  const draft = snapshot?.drafts.find(item => item.id === selectedDraft);
  const aiReady = !!health?.ai?.ready && (health.provider === 'mock' || health.ai.authenticated);

  const selectRoom = useCallback((id: string) => {
    if (currentRoom.current === id) { setMobileNav(false); return; }
    currentRoom.current = id;
    snapshotSequence.current += 1;
    setRoomId(id); setSnapshot(null); setError(''); setMobileNav(false); setInput('');
    setSelectedDraft(''); setDraftText(''); setSourceDirty(false); setSourcePaths([]); setAtBottom(true);
    setSending(false); setDraftBusy(false); setSavingDraft(false); setSavingSources(false);
    localStorage.setItem('research-room-selected', id);
  }, []);

  const refreshRoom = useCallback(async (id: string) => {
    const sequence = ++snapshotSequence.current;
    const value = await api<Snapshot>(`/rooms/${encodeURIComponent(id)}`);
    if (currentRoom.current === id && sequence === snapshotSequence.current) {
      setSnapshot(value);
      setRooms(items => items.map(item => item.id === id ? value.room : item));
    }
    return value;
  }, []);

  const refreshHealth = useCallback(async () => {
    const value = await api<Health>('/health'); setHealth(value);
  }, []);

  const initialize = useCallback(async () => {
    setInitialLoading(true); setError('');
    const results = await Promise.allSettled([
      refreshHealth(),
      api<{ rooms: Room[] }>('/rooms').then(value => {
        setRooms(value.rooms);
        if (!currentRoom.current && value.rooms.length) {
          const previous = localStorage.getItem('research-room-selected');
          selectRoom(value.rooms.find(item => item.id === previous)?.id || value.rooms[0].id);
        }
      }),
      api<{ sources: Source[] }>('/sources').then(value => setAvailableSources(value.sources)),
    ]);
    const failed = results.find(value => value.status === 'rejected');
    if (failed?.status === 'rejected') setError(readableError(failed.reason));
    setInitialLoading(false);
  }, [refreshHealth, selectRoom]);

  useEffect(() => { void initialize(); }, [initialize]);
  useEffect(() => {
    const timer = window.setInterval(() => { void refreshHealth().catch(() => {}); }, 30000);
    return () => window.clearInterval(timer);
  }, [refreshHealth]);

  useEffect(() => {
    if (!roomId) return;
    setRoomLoading(true); setConnected(false);
    void refreshRoom(roomId).catch(value => { if (currentRoom.current === roomId) setError(readableError(value)); }).finally(() => { if (currentRoom.current === roomId) setRoomLoading(false); });
    let timer: ReturnType<typeof setTimeout> | undefined;
    const events = new EventSource(`/api/rooms/${encodeURIComponent(roomId)}/events`);
    events.onopen = () => { setConnected(true); void refreshRoom(roomId).catch(() => {}); };
    events.onerror = () => setConnected(false);
    events.addEventListener('update', () => {
      if (timer) return;
      timer = setTimeout(() => {
        timer = undefined;
        void refreshRoom(roomId).catch(value => { if (currentRoom.current === roomId) setError(readableError(value)); });
      }, 100);
    });
    return () => { events.close(); if (timer) clearTimeout(timer); };
  }, [roomId, refreshRoom]);

  useEffect(() => {
    if (snapshot && !sourceDirty) setSourcePaths(snapshot.room.paths ?? snapshot.sources.filter(source => availableSources.some(item => item.path === source.path)).map(source => source.path));
  }, [snapshot, sourceDirty, availableSources]);

  useEffect(() => {
    if (snapshot && !selectedDraft && snapshot.drafts.length) {
      const latest = snapshot.drafts[snapshot.drafts.length - 1];
      setSelectedDraft(latest.id); setDraftText(latest.content);
    }
  }, [snapshot, selectedDraft]);

  useEffect(() => {
    if (atBottom && scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [snapshot?.messages, atBottom]);

  async function createRoom(event: FormEvent) {
    event.preventDefault(); if (!newTitle.trim() || creating) return;
    setCreating(true); setError('');
    try {
      const value = await post<{ room: Room }>('/rooms', { title: newTitle.trim(), reqId: newReq.trim() || 'REQ-005' });
      setRooms(items => [value.room, ...items]); selectRoom(value.room.id);
      setNewRoomOpen(false); setNewTitle('');
    } catch (value) { setError(readableError(value)); } finally { setCreating(false); }
  }

  async function sendMessage(event?: FormEvent) {
    event?.preventDefault(); if (!input.trim() || !roomId || sending || !aiReady) return;
    const id = roomId; const content = input.trim();
    if (sendKey.current.content !== content) sendKey.current = { content, id: crypto.randomUUID() };
    setSending(true); setError(''); setAtBottom(true);
    try {
      await post(`/rooms/${encodeURIComponent(id)}/messages`, { content, clientId: sendKey.current.id, recipients: recipient === 'pm' ? [] : [recipient], discuss });
      if (currentRoom.current === id) { setInput(''); sendKey.current = { content: '', id: '' }; await refreshRoom(id); composerRef.current?.focus(); }
    } catch (value) { if (currentRoom.current === id) setError(readableError(value)); }
    finally { if (currentRoom.current === id) setSending(false); }
  }

  function composerKey(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); void sendMessage(); }
  }

  async function actOnRun(id: string, action: 'stop' | 'retry') {
    const room = roomId; setRunBusy(id); setError('');
    try { await post(`/runs/${encodeURIComponent(id)}/${action}`); await refreshRoom(room); }
    catch (value) { setError(readableError(value)); } finally { setRunBusy(''); }
  }

  async function makeDraft() {
    const id = roomId; if (!id || draftBusy) return;
    setDraftBusy(true); setPanel('notes'); setPanelOpen(true); setError('');
    try {
      const value = await post<{ draft: Draft }>(`/rooms/${encodeURIComponent(id)}/drafts`, { target: draftTarget });
      if (currentRoom.current === id) { await refreshRoom(id); setSelectedDraft(value.draft.id); setDraftText(value.draft.content); setDraftPreview(false); }
    } catch (value) { if (currentRoom.current === id) setError(readableError(value)); }
    finally { if (currentRoom.current === id) setDraftBusy(false); }
  }

  async function saveDraft() {
    if (!draft || !draftText.trim() || savingDraft) return;
    const id = roomId; setSavingDraft(true); setError('');
    try {
      const value = await post<{ draft: Draft }>(`/drafts/${encodeURIComponent(draft.id)}/save`, { content: draftText });
      if (currentRoom.current === id) { await refreshRoom(id); if (selectedDraftRef.current === value.draft.id) setDraftText(value.draft.content); }
    } catch (value) { if (currentRoom.current === id) setError(readableError(value)); }
    finally { if (currentRoom.current === id) setSavingDraft(false); }
  }

  async function saveSources() {
    const id = roomId; setSavingSources(true); setError('');
    try { await post(`/rooms/${encodeURIComponent(id)}/sources`, { paths: sourcePaths }); if (currentRoom.current === id) { setSourceDirty(false); await refreshRoom(id); } }
    catch (value) { if (currentRoom.current === id) setError(readableError(value)); }
    finally { if (currentRoom.current === id) setSavingSources(false); }
  }

  const participantIds = new Set(['pm', ...(snapshot?.messages.filter(item => item.role !== 'user').map(item => item.role) || [])]);
  return <div className="app-shell">
    {mobileNav && <button className="mobile-backdrop" aria-label="話題一覧を閉じる" onClick={() => setMobileNav(false)} />}
    <aside className={`sidebar ${mobileNav ? 'is-open' : ''}`}>
      <a className="brand" href="/" aria-label="研究室 ホーム"><span className="brand-mark"><FlaskConical size={22} strokeWidth={1.7} /></span><span>研究室<small>RESEARCH ROOM</small></span></a>
      <div className="sidebar-intro">考えの途中から、話そう。</div>
      <button className="new-room" onClick={() => { setNewRoomOpen(true); setMobileNav(false); }}><Plus size={17} />新しい話題</button>
      <div className="section-eyebrow">YOUR CONVERSATIONS</div>
      <nav className="room-list" aria-label="話題一覧">
        {initialLoading ? <p className="sidebar-empty">読み込んでいます…</p> : !rooms.length ? <p className="sidebar-empty">まだ話題はありません。<br />気になることから始めましょう。</p> : rooms.map(room => <button className={`room-link ${room.id === roomId ? 'selected' : ''}`} key={room.id} onClick={() => selectRoom(room.id)}><MessageCircle size={16} /><span><strong>{room.title}</strong><small>{room.reqId}</small></span>{room.id === roomId && <span className="room-current" />}</button>)}
      </nav>
      <div className="sidebar-footer"><span className={`connection-dot ${aiReady ? 'ready' : ''}`} /><span>{aiReady ? health?.provider === 'mock' ? '動作確認モード' : 'Codex に接続' : 'AI 接続を確認中'}<small>個人の研究ワークスペース</small></span><button className="icon-button" aria-label="接続を再確認" onClick={() => void initialize()}><RefreshCw size={15} /></button></div>
    </aside>

    <main className="main-area">
      <header className="topbar"><div className="breadcrumb"><button className="icon-button mobile-only" aria-label="話題一覧を開く" onClick={() => setMobileNav(true)}><Menu size={20} /></button><span>研究室</span><ChevronRight size={13} /><strong>{snapshot?.room.title || '新しい対話'}</strong></div><div className="topbar-actions"><span className="local-badge"><Circle size={7} fill="currentColor" />LOCAL</span><button className={`icon-button panel-toggle ${panelOpen ? 'active' : ''}`} aria-label="メモと資料を開く" onClick={() => setPanelOpen(value => !value)}><FileText size={18} /></button></div></header>
      {error && <div className="error-banner" role="alert"><span>{error}</span><button onClick={() => setError('')} aria-label="エラー表示を閉じる"><X size={16} /></button></div>}
      {health?.provider === 'mock' && <div className="notice-banner">動作確認モードです。ここでの応答は模擬応答で、実際のAIによる研究相談ではありません。</div>}
      {health && !aiReady && <div className="notice-banner">AIへ接続できません。{health.ai.error || 'Codexの認証とバックエンドの状態を確認してください。'}<button className="text-button" onClick={() => void refreshHealth().catch(value => setError(readableError(value)))}>再確認</button></div>}
      <div className="workspace">
        <section className="conversation" aria-label="会話">
          {!roomId ? <div className="welcome"><span className="welcome-icon"><MessageCircle size={27} strokeWidth={1.5} /></span><span className="section-eyebrow">A PLACE TO THINK TOGETHER</span><h1>研究のこと、<br />気軽に話しませんか。</h1><p>まとまっていない考えも、少し気になることも。<br />PMと専門担当が、一緒に考えます。</p><button className="primary-button" onClick={() => setNewRoomOpen(true)}><Plus size={17} />最初の話題をつくる</button><div className="welcome-footnote"><BookOpen size={14} />研究の経緯は、要求書や報告から共有します。</div></div> : <>
            <div className="conversation-heading"><div><span className="req-tag">{snapshot?.room.reqId || rooms.find(item => item.id === roomId)?.reqId}</span><h1>{snapshot?.room.title || rooms.find(item => item.id === roomId)?.title || '読み込み中…'}</h1><p>途中の考えも、そのまま言葉に。</p></div><div className="participant-stack" aria-label="会話の参加者">{roles.filter(role => participantIds.has(role.id)).map(role => <span title={role.label} key={role.id}><Avatar role={role.id} small /></span>)}</div></div>
            <div className="chat-scroll" ref={scrollRef} onScroll={event => { const el = event.currentTarget; setAtBottom(el.scrollHeight - el.scrollTop - el.clientHeight < 100); }}>
              {roomLoading && !snapshot ? <div className="room-loading"><RefreshCw size={18} className="spinning" />会話を読み込んでいます</div> : !snapshot?.messages.length ? <div className="conversation-empty"><span className="empty-line" /><MessageCircle size={20} strokeWidth={1.4} /><h2>ここから、ひとつずつ。</h2><p>今、何が気になっていますか？<br />研究の相談でも、ちょっとした壁打ちでも。</p></div> : snapshot.messages.map(message => <ChatMessage key={message.id} message={message} roles={roles} />)}
              {snapshot?.runs.filter(run => run.status === 'failed' || run.status === 'interrupted').map(run => <div className="run-notice" key={run.id}><span>{run.status === 'failed' ? '返答を完了できませんでした。' : '返答を中断しました。'}{run.error && <small>{run.error}</small>}</span><button className="text-button" disabled={!!runBusy || !aiReady} onClick={() => void actOnRun(run.id, 'retry')}><RefreshCw size={13} />再試行</button></div>)}
            </div>
            {!atBottom && <button className="scroll-down icon-button" aria-label="最新の発言へ移動" onClick={() => setAtBottom(true)}><ArrowDown size={17} /></button>}
            <div className="composer-area">
              {(activeRun || queuedRuns.length > 0) && <div className="run-bar"><span><i className="pulse-dot" />{activeRun ? '返答しています' : '順番を待っています'}{queuedRuns.length > 0 && ` · ${queuedRuns.length}件待機`}</span><div>{activeRun && <button className="stop-button" disabled={!!runBusy} onClick={() => void actOnRun(activeRun.id, 'stop')}><Square size={11} fill="currentColor" />停止</button>}{queuedRuns.length > 0 && <button className="text-button" disabled={!!runBusy} onClick={() => void actOnRun(queuedRuns[0].id, 'stop')}>待機を取消</button>}</div></div>}
              <form className="composer" onSubmit={event => void sendMessage(event)}>
                <textarea ref={composerRef} value={input} onChange={event => setInput(event.target.value)} onKeyDown={composerKey} placeholder="今、考えていることを…" aria-label="メッセージ" rows={3} maxLength={12000} disabled={sending} />
                <div className="composer-toolbar"><div className="recipient-controls"><label className="recipient-select"><Users size={14} /><select value={recipient} onChange={event => setRecipient(event.target.value)} aria-label="話しかける相手">{roles.map(role => <option key={role.id} value={role.id}>{role.id === 'pm' ? 'PM に相談' : `${role.label}に聞く`}</option>)}</select></label><label className="discussion-toggle"><input type="checkbox" checked={discuss} onChange={event => setDiscuss(event.target.checked)} /><span>みんなで議論</span></label></div><button className="send-button" type="submit" aria-label="メッセージを送信" disabled={!input.trim() || sending || !aiReady}>{sending ? <RefreshCw size={17} className="spinning" /> : <ArrowUp size={19} />}</button></div>
              </form>
              <div className="composer-caption"><span>Enter で送信 · Shift + Enter で改行</span><span className={connected ? '' : 'reconnecting'}>{connected ? '会話を保存しています' : '接続を再試行中…'}</span></div>
            </div>
          </>}
        </section>

        <aside className={`detail-panel ${panelOpen ? 'is-open' : ''}`} aria-label="参加者とメモ">
          <div className="panel-heading"><span>この話題について</span><button className="icon-button close-panel" aria-label="メモと資料を閉じる" onClick={() => setPanelOpen(false)}><X size={17} /></button></div>
          <div className="participants-section"><span className="section-eyebrow">YOUR RESEARCH TEAM</span><div className="team-primary"><Avatar role="pm" /><div><strong>PM</strong><small>一緒に考える、相談の窓口</small></div><span className={`connection-dot ${aiReady ? 'ready' : ''}`} /></div><details className="team-details"><summary>専門担当に相談する<span>{roles.filter(role => role.id !== 'pm').length} 名</span></summary><div className="team-members">{roles.filter(role => role.id !== 'pm').map(role => <button title={role.description} key={role.id} onClick={() => { setRecipient(role.id); composerRef.current?.focus(); }}><Avatar role={role.id} small /><span><strong>{role.label}</strong><small>{role.description}</small></span></button>)}</div></details><p className="team-note">必要な視点を持つ担当が参加します。</p></div>
          <div className="panel-tabs"><button className={panel === 'notes' ? 'active' : ''} onClick={() => setPanel('notes')}><FileText size={15} />残しておく</button><button className={panel === 'sources' ? 'active' : ''} onClick={() => setPanel('sources')}><BookOpen size={15} />参照資料</button></div>
          <div className="panel-content">{panel === 'notes' ? <>
            <p className="panel-copy">話して見えてきたことを、<br />後から読み返せるメモに。</p>
            <div className="draft-create"><select aria-label="メモの保存先の種類" value={draftTarget} onChange={event => setDraftTarget(event.target.value as 'report' | 'checkpoint')}><option value="report">議論メモ（reports）</option><option value="checkpoint">引継ぎ記録</option></select><button className="outline-button" onClick={() => void makeDraft()} disabled={!snapshot?.messages.length || draftBusy || savingDraft || !!activeRun || !aiReady}>{draftBusy ? <RefreshCw size={14} className="spinning" /> : <Plus size={14} />}{draftBusy ? 'PMが要点を整理しています…' : '会話から下書きをつくる'}</button></div>
            {!!snapshot?.drafts.length && <><label className="field-label" htmlFor="draft-select">保存したメモと下書き</label><select id="draft-select" className="draft-select" disabled={savingDraft} value={selectedDraft} onChange={event => { const item = snapshot.drafts.find(value => value.id === event.target.value); setSelectedDraft(event.target.value); setDraftText(item?.content || ''); }}>{snapshot.drafts.map((item, index) => <option key={item.id} value={item.id}>{index + 1}. {item.target === 'report' ? '議論メモ' : '引継ぎ'} · {item.status === 'saved' ? '保存済み' : '下書き'}</option>)}</select></>}
            {draft ? <div className="draft-editor"><div className="draft-editor-bar"><span>{draft.status === 'saved' ? <><Check size={13} />保存済み</> : 'Markdown の下書き'}</span><button className="text-button" onClick={() => setDraftPreview(value => !value)}>{draftPreview ? '編集' : 'プレビュー'}</button></div>{draftPreview ? <div className="draft-preview"><Markdown>{draftText}</Markdown></div> : <textarea aria-label="Markdownの下書き" value={draftText} onChange={event => setDraftText(event.target.value)} readOnly={draft.status === 'saved' || savingDraft} spellCheck={false} />}{draft.proofreading && <details className="proofreading"><summary>校正の確認結果</summary><p>検出内容は見直しの候補です。意味と文脈に照らして判断します。</p><pre>{draft.proofreading}</pre></details>}{draft.savedPath && <p className="saved-path">{draft.savedPath}</p>}{draft.status !== 'saved' && <button className="primary-button save-draft" disabled={!draftText.trim() || savingDraft} onClick={() => void saveDraft()}>{savingDraft ? <RefreshCw size={14} className="spinning" /> : <Save size={14} />}{savingDraft ? '保存しています…' : '内容を確認して保存'}</button>}</div> : <div className="notes-empty"><FileText size={25} strokeWidth={1.2} /><span>メモはまだありません</span><small>残したい話ができたら、下書きをつくれます。</small></div>}
            <p className="panel-footnote">メモの保存だけでは、研究方針を承認した扱いにはなりません。</p>
          </> : <><p className="panel-copy">この話題で一緒に読む資料を選びます。<br />要求書や原稿の版は、発言の根拠になります。</p>{snapshot?.contextWarning && <p className="context-warning">{snapshot.contextWarning}</p>}<div className="source-list">{!availableSources.length && <p className="muted">選択できる資料がありません。</p>}{availableSources.map(source => <label key={source.path}><input type="checkbox" disabled={source.automatic || !roomId || savingSources} checked={source.automatic || sourcePaths.includes(source.path)} onChange={event => { setSourceDirty(true); setSourcePaths(paths => event.target.checked ? [...paths, source.path] : paths.filter(path => path !== source.path)); }} /><span><strong>{source.label || source.path.split('/').pop()}</strong><small>{source.path}</small>{source.automatic && <small>全参加者が自動で参照</small>}{snapshot?.sources.find(item => item.path === source.path)?.truncated && <small className="source-warning">{snapshot?.sources.find(item => item.path === source.path)?.unread ? '上限により未読' : '一部を抜粋して参照'}</small>}</span></label>)}</div><button className="outline-button sources-save" disabled={!roomId || !sourceDirty || savingSources} onClick={() => void saveSources()}>{savingSources ? <RefreshCw size={14} className="spinning" /> : <Check size={14} />}選択した資料を反映</button><p className="panel-footnote">必須の指示・要求書とreference内の共有資料は、自動で参照されます。以前のチャットの記憶は引き継がれません。</p></> }</div>
          <div className="panel-bottom"><FlaskConical size={14} /><span>考える時間を、ひとりきりにしない。</span></div>
        </aside>
      </div>
    </main>
    {newRoomOpen && <div className="modal-backdrop" onMouseDown={event => { if (event.target === event.currentTarget && !creating) setNewRoomOpen(false); }}><section className="modal" role="dialog" aria-modal="true" aria-labelledby="new-room-title"><div className="modal-top"><span className="section-eyebrow">NEW CONVERSATION</span><button className="icon-button" aria-label="新しい話題を閉じる" disabled={creating} onClick={() => setNewRoomOpen(false)}><X size={18} /></button></div><h2 id="new-room-title">今日は、何を話そう。</h2><p>話題はあとから振り返るための目印です。</p>{error && <p className="modal-error" role="alert">{error}</p>}<form onSubmit={event => void createRoom(event)}><label className="field-label" htmlFor="room-title">話題の名前</label><input id="room-title" autoFocus placeholder="例：ジャーナル投稿に向けた評価の相談" value={newTitle} onChange={event => setNewTitle(event.target.value)} maxLength={120} required /><label className="field-label" htmlFor="room-req">関連する要求書</label><input id="room-req" placeholder="REQ-005" value={newReq} onChange={event => setNewReq(event.target.value)} maxLength={20} pattern="REQ-[0-9]{3,}" /><small className="input-help">研究の相談は、その内容に対応する REQ-ID を指定します。</small><button className="primary-button modal-submit" disabled={!newTitle.trim() || creating}>{creating ? <RefreshCw size={15} className="spinning" /> : <MessageCircle size={16} />}{creating ? '作成しています…' : '話し始める'}</button></form></section></div>}
  </div>;
}
