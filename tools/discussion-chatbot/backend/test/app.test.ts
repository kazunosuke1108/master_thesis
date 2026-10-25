import {test} from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {setTimeout as delay} from 'node:timers/promises';
import {createApp,type Config} from '../src/server.js';
import {Store} from '../src/store.js';
import type {AiProvider,GenerateOptions} from '../src/ai.js';
class FakeAi implements AiProvider {
  calls:GenerateOptions[]=[]; failures=0;
  status=async()=>({ready:true,authenticated:true});close=async()=>{};
  async generate(o:GenerateOptions) {
    this.calls.push(o);if(this.failures>0){this.failures--;throw new Error('一時的な接続失敗');}
    if(o.schema){const text=JSON.stringify((o.schema.properties as any).participants?{reply:'担当に聞きます。',participants:['evaluator','reviewer']}:{content:'## 未承認の提案\n\n仮説の検討。\n\n## 未確認事項\n\n実測は未実施。'});return {text,model:'test'};}
    const text='話題を確認しました。';
    for(const chunk of text){await delay(8,undefined,{signal:o.signal});o.onDelta?.(chunk);}
    return {text,model:'test'};
  }
}
async function fixture() {
  const root=await mkdtemp(join(tmpdir(),'req005-app-'));
  for(const dir of ['requests','reports','.codex/research-docs/checkpoints'])await mkdir(join(root,dir),{recursive:true});
  await writeFile(join(root,'AGENTS.md'),'研究の主張を変更しない。');
  await writeFile(join(root,'.codex/research-docs/requirements-workflow.md'),'原文と提案を区別する。');
  await writeFile(join(root,'requests/2026-10-25-001-REQ-005-topic.md'),'# REQ-005\n状態:承認済み');
  const config:Config={workspace:root,dataDir:join(root,'data'),provider:'mock',codexBin:'codex',codexHome:join(root,'codex'),timeoutMs:3000,commonInstructions:'',commonWorkflow:'',port:0,host:'127.0.0.1',allowedHosts:['127.0.0.1','localhost']};
  const ai=new FakeAi();const app=createApp(config,ai);await new Promise<void>(r=>app.server.listen(0,'127.0.0.1',r));
  const address=app.server.address() as {port:number};const base=`http://127.0.0.1:${address.port}`;
  const api=async(path:string,method='GET',data?:unknown,headers:Record<string,string>={})=>{
    const res=await fetch(base+path,{method,headers:{'Content-Type':'application/json',...headers},body:data===undefined?undefined:JSON.stringify(data)});return {status:res.status,data:await res.json() as any};
  };
  const create=async()=>{const r=await api('/api/rooms','POST',{title:'評価の相談',reqId:'REQ-005'});assert.equal(r.status,201);return r.data.room.id as string;};
  const wait=async(id:string,runId:string)=>{for(let i=0;i<100;i++){const s=await api(`/api/rooms/${id}`);const run=s.data.runs.find((r:any)=>r.id===runId);if(!['running','queued'].includes(run.status))return s.data;await delay(20);}throw new Error('run timeout');};
  return {...app,root,base,api,ai,create,wait,cleanup:async()=>{await app.close();await rm(root,{recursive:true,force:true});}};
}
test('room history, idempotent input, discussion roles, references, SSE replay',async()=>{
  const f=await fixture();try{
    const id=await f.create();
    const one=await f.api(`/api/rooms/${id}/messages`,'POST',{content:'みんなで議論して',clientId:'same'});
    const duplicate=await f.api(`/api/rooms/${id}/messages`,'POST',{content:'みんなで議論して',clientId:'same'});
    assert.equal(one.data.runId,duplicate.data.runId);assert.equal(duplicate.data.duplicate,true);
    const s=await f.wait(id,one.data.runId);assert.equal(s.runs[0].status,'complete');
    assert.deepEqual(s.messages.map((m:any)=>m.role),['user','pm','evaluator','reviewer','pm']);
    assert.ok(s.sources.some((r:any)=>r.path.includes('REQ-005')));
    assert.ok(s.runs[0].sources.some((r:any)=>r.path.includes('REQ-005')&&r.hash.length===64));
    const controller=new AbortController();const events=await fetch(`${f.base}/api/rooms/${id}/events?after=0`,{signal:controller.signal});
    const read=await events.body!.getReader().read();assert.match(new TextDecoder().decode(read.value),/event: update/);controller.abort();
    const restart=new Store(join(f.root,'data/chat.sqlite'));assert.equal(restart.list('message',id).length,5);restart.close();
  }finally{await f.cleanup();}
});
test('interruption, retry, queued-user isolation and request validation',async()=>{
  const f=await fixture();try{
    const id=await f.create();
    const one=await f.api(`/api/rooms/${id}/messages`,'POST',{content:'first-message',clientId:'one',recipients:['evaluator']});
    const two=await f.api(`/api/rooms/${id}/messages`,'POST',{content:'future-message',clientId:'two'});
    await delay(30);await f.api(`/api/runs/${one.data.runId}/stop`,'POST',{});
    const stopped=await f.wait(id,one.data.runId);assert.equal(stopped.runs[0].status,'interrupted');
    await f.wait(id,two.data.runId);
    assert.ok(!f.ai.calls[0].input.includes('future-message'));
    // Send another pair without interruption: B must see A even if B was persisted before A finished.
    const a=await f.api(`/api/rooms/${id}/messages`,'POST',{content:'third-message',clientId:'third'});
    const b=await f.api(`/api/rooms/${id}/messages`,'POST',{content:'fourth-message',clientId:'fourth'});
    await f.wait(id,b.data.runId);
    const aReply=f.store.list<any>('message',id).find(m=>m.runId===a.data.runId&&m.role==='pm');
    const bInput=f.ai.calls.find(c=>c.key.includes(b.data.runId));
    assert.ok(bInput?.input.includes(aReply.id),'Queued B must include A response in logical turn order');
    const retry=await f.api(`/api/runs/${one.data.runId}/retry`,'POST',{});const s=await f.wait(id,retry.data.runId);assert.equal(s.runs.at(-1).status,'complete');
    const invalid=await f.api(`/api/rooms/${id}/sources`,'POST',{paths:['../../.codex/auth.json']});assert.equal(invalid.status,400);
    const origin=await f.api('/api/rooms','POST',{title:'test'},{Origin:'https://untrusted.example'});assert.equal(origin.status,403);
    assert.equal((await f.api('/api/rooms/missing')).status,400);
    const invalidRole=await f.api(`/api/rooms/${id}/messages`,'POST',{content:'hello',clientId:'bad',recipients:['root']});assert.equal(invalidRole.status,400);
  }finally{await f.cleanup();}
});
test('draft review, save numbering and idempotency; changed document conflicts',async()=>{
  const f=await fixture();try{
    const id=await f.create();const sent=await f.api(`/api/rooms/${id}/messages`,'POST',{content:'評価を相談したい',clientId:'draft'});await f.wait(id,sent.data.runId);
    const d=(await f.api(`/api/rooms/${id}/drafts`,'POST',{target:'report'})).data.draft;
    assert.equal(d.status,'draft');
    const saved=await f.api(`/api/drafts/${d.id}/save`,'POST',{content:d.content});assert.equal(saved.status,200);
    assert.match(saved.data.draft.savedPath,/2026-\d{2}-\d{2}-\d{3}-REQ-005-discussion-note.md/);
    const text=await readFile(join(f.root,saved.data.draft.savedPath),'utf8');assert.match(text,/保存は研究方針/);assert.match(text,/未承認の提案/);assert.match(text,/発生日時/);assert.match(text,/静的検査は未実施/);assert.match(text,/SHA-256/);
    assert.equal((await f.api(`/api/drafts/${d.id}/save`,'POST',{content:d.content})).data.draft.savedPath,saved.data.draft.savedPath);
    const next=(await f.api(`/api/rooms/${id}/drafts`,'POST',{target:'report'})).data.draft;
    await writeFile(join(f.root,'reports/2026-10-25-099-other.md'),'別作業の変更');
    assert.equal((await f.api(`/api/drafts/${next.id}/save`,'POST',{content:next.content})).status,409);
    const cp=(await f.api(`/api/rooms/${id}/drafts`,'POST',{target:'checkpoint'})).data.draft;
    const cpSave=await f.api(`/api/drafts/${cp.id}/save`,'POST',{content:cp.content});assert.equal(cpSave.status,200);assert.match(cpSave.data.draft.savedPath,/checkpoints/);
  }finally{await f.cleanup();}
});
test('restart marks incomplete runs/messages interrupted without generation',async()=>{
  const f=await fixture();try{
    const room=f.store.room('再起動','REQ-005');const result=f.store.addInput(room,'未処理','restart',[],false);
    result.run.status='running';f.store.put('run',result.run);
    const other=new Store(join(f.root,'data/chat.sqlite'));assert.equal((other.get<any>('run',result.run.id)).status,'interrupted');other.close();assert.equal(f.ai.calls.length,0);
  }finally{await f.cleanup();}
});

test('configured proofreading runs and referenced document changes block stale draft save',async()=>{
  const f=await fixture();try {
    const id=await f.create();const sent=await f.api(`/api/rooms/${id}/messages`,'POST',{content:'文書に残したい',clientId:'proof'});await f.wait(id,sent.data.runId);
    const script=join(f.root,'lint.py');await writeFile(script,"import pathlib,sys\nassert pathlib.Path(sys.argv[1]).read_text()\nprint('PROOFREAD_OK')\n");
    // Exercise configured proofreading independently without depending on a developer's home directory.
    const {Documents}=await import('../src/documents.js');
    assert.equal((await new Documents(f.root,script).lint('校正対象')).trim(),'PROOFREAD_OK');
    const d=(await f.api(`/api/rooms/${id}/drafts`,'POST',{target:'report'})).data.draft;
    await writeFile(join(f.root,'AGENTS.md'),'参照指示が変更された');
    const saved=await f.api(`/api/drafts/${d.id}/save`,'POST',{content:d.content});assert.equal(saved.status,409);assert.match(saved.data.error,/参照資料/);
  }finally{await f.cleanup();}
});

test('a failed retry does not permanently suppress further retries of the same run',async()=>{
  const f=await fixture();try{
    const room=await f.create();f.ai.failures=2;
    const sent=await f.api(`/api/rooms/${room}/messages`,'POST',{content:'再試行の確認',clientId:'repeat-retry'});
    await f.wait(room,sent.data.runId);
    const first=await f.api(`/api/runs/${sent.data.runId}/retry`,'POST',{});const firstResult=await f.wait(room,first.data.runId);
    assert.equal(firstResult.runs.find((r:any)=>r.id===first.data.runId).status,'failed');
    const second=await f.api(`/api/runs/${first.data.runId}/retry`,'POST',{});assert.notEqual(second.data.runId,first.data.runId);
    const duplicate=await f.api(`/api/runs/${sent.data.runId}/retry`,'POST',{});assert.equal(second.data.runId,duplicate.data.runId);
    const final=await f.wait(room,second.data.runId);assert.equal(final.runs.find((r:any)=>r.id===second.data.runId).status,'complete');
  }finally{await f.cleanup();}
});

test('shared reference files reach every participant and drafts, refresh on change, and reject unsafe files',async()=>{
  const f=await fixture();try{
    const id=await f.create(); // Created before reference files exist: existing rooms must also gain them.
    await mkdir(join(f.root,'reference/nested'),{recursive:true});
    await writeFile(join(f.root,'reference/nested/context.md'),'SHARED_REFERENCE_VERSION_ONE');
    await writeFile(join(f.root,'reference/.private.md'),'HIDDEN_REFERENCE');
    await writeFile(join(f.root,'reference/ignore.bin'),'UNSUPPORTED_BINARY');
    const {symlink}=await import('node:fs/promises');
    await writeFile(join(f.root,'outside.md'),'OUTSIDE_REFERENCE');
    await symlink(join(f.root,'outside.md'),join(f.root,'reference/link.md'));
    const catalog=await f.api('/api/sources');const shared=catalog.data.sources.filter((s:any)=>s.path.startsWith('reference/'));
    assert.deepEqual(shared.map((s:any)=>s.path),['reference/nested/context.md']);assert.equal(shared[0].automatic,true);
    const first=await f.api(`/api/rooms/${id}/messages`,'POST',{content:'みんなで議論して',clientId:'reference-one'});const snap=await f.wait(id,first.data.runId);
    assert.equal(snap.runs[0].status,'complete');
    assert.ok(f.ai.calls.length>=4);assert.ok(f.ai.calls.every(c=>c.input.includes('SHARED_REFERENCE_VERSION_ONE')));
    assert.ok(!f.ai.calls.some(c=>/HIDDEN_REFERENCE|OUTSIDE_REFERENCE|UNSUPPORTED_BINARY/.test(c.input)));
    const source=snap.sources.find((s:any)=>s.path==='reference/nested/context.md');assert.ok(source.hash);
    await f.api(`/api/rooms/${id}/drafts`,'POST',{target:'report'});
    assert.ok(f.ai.calls.at(-1)!.input.includes('SHARED_REFERENCE_VERSION_ONE'));
    await writeFile(join(f.root,'reference/nested/context.md'),'SHARED_REFERENCE_VERSION_TWO');
    const second=await f.api(`/api/rooms/${id}/messages`,'POST',{content:'担当にも確認',clientId:'reference-two',recipients:['editor']});const updated=await f.wait(id,second.data.runId);
    assert.ok(f.ai.calls.at(-1)!.input.includes('SHARED_REFERENCE_VERSION_TWO'));
    assert.notEqual(updated.sources.find((s:any)=>s.path==='reference/nested/context.md').hash,source.hash);
    await f.api(`/api/rooms/${id}/sources`,'POST',{paths:['reference/nested/context.md']});
    await rm(join(f.root,'reference/nested/context.md'));
    await symlink(join(f.root,'outside.md'),join(f.root,'reference/nested/context.md'));
    const blocked=await f.api(`/api/rooms/${id}/messages`,'POST',{content:'古い選択の安全確認',clientId:'reference-symlink'});await f.wait(id,blocked.data.runId);
    assert.ok(!f.ai.calls.at(-1)!.input.includes('OUTSIDE_REFERENCE'));
  }finally{await f.cleanup();}
});

test('explicit reference selection keeps its context budget before large automatic shared files',async()=>{
  const f=await fixture();try{
    await mkdir(join(f.root,'reference'));
    for(let i=0;i<8;i++)await writeFile(join(f.root,`reference/a${i}.md`),'共有'.repeat(12000));
    await writeFile(join(f.root,'reference/z-priority.md'),'EXPLICIT_PRIORITY_CONTENT');
    const room=f.store.room('優先参照','REQ-005');room.paths=['reference/z-priority.md'];
    const {Context}=await import('../src/context.js');const result=await new Context(f.root,'','').load(room);
    assert.ok(result.sources.find(s=>s.path==='reference/z-priority.md')!.text.includes('EXPLICIT_PRIORITY_CONTENT'));
    assert.ok(result.sources.some(s=>s.unread&&s.text.length===0));assert.match(result.warning!,/未読/);
    assert.ok(result.sources.reduce((total,s)=>total+s.text.length,0)<=100000);
  }finally{await f.cleanup();}
});
