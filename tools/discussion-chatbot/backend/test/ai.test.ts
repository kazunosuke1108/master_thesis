import assert from 'node:assert/strict';
import { mkdtemp, writeFile, chmod, rm, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { createProvider, type AiProvider } from '../src/ai.js';

async function fixture(t: { after: (cleanup: () => Promise<void>) => void }, timeoutMs = 2_000) {
  const dir = await mkdtemp(join(tmpdir(), 'research-ai-'));
  const bin = join(dir, 'fake-codex');
  const log = join(dir, 'requests.jsonl');
  const source = `#!/usr/bin/env node
import {createInterface} from 'node:readline';
import {appendFileSync} from 'node:fs';
const log=${JSON.stringify(log)};
appendFileSync(log, JSON.stringify({method:'fixture/process',args:process.argv.slice(2),codexHome:process.env.CODEX_HOME,home:process.env.HOME})+'\\n');
const send=(m)=>process.stdout.write(JSON.stringify(m)+'\\n');
let threads=0;
const ids=new Map();
const ack=(m,result)=>send({id:m.id,result});
const finish=(threadId,turnId,text='本文')=>{
 send({method:'item/agentMessage/delta',params:{threadId,turnId,itemId:'i',delta:text}});
 send({method:'item/completed',params:{threadId,turnId,item:{type:'agentMessage',phase:'final_answer',text}}});
 send({method:'turn/completed',params:{threadId,turn:{id:turnId,status:'completed'}}});
};
createInterface({input:process.stdin}).on('line',line=>{
 const m=JSON.parse(line);appendFileSync(log,JSON.stringify(m)+'\\n');
 if(m.method==='initialize')return ack(m,{});
 if(m.method==='account/read')return ack(m,{account:{type:'chatgpt',email:'private@example.test'},requiresOpenaiAuth:true});
 if(m.method==='model/list')return ack(m,{data:[{model:'test-model',isDefault:true,hidden:false}]});
 if(m.method==='thread/start')return ack(m,{thread:{id:'thread-'+(++threads)}});
 if(m.method==='thread/resume')return ack(m,{thread:{id:m.params.threadId}});
 if(m.method==='turn/start'){
  if('access' in (m.params.sandboxPolicy??{}))return send({id:m.id,error:{code:-32600,message:'Invalid request: readOnly.access is no longer supported; use permissionProfile for restricted reads'}});
  const {threadId}=m.params;const input=m.params.input[0].text;const turnId='turn-'+m.id;
  ids.set(threadId,turnId);
  if(input==='crash')return process.exit(2);
  if(input==='protocol-error')return send({id:m.id,error:{code:-32600,message:'Invalid request: SECRET'}});
  if(input==='rpc-error')return send({id:m.id,error:{code:-1,message:'Authorization: Bearer SECRET 401'}});
  if(input==='fail'){
   ack(m,{turn:{id:turnId}});
   return send({method:'turn/completed',params:{threadId,turn:{id:turnId,status:'failed',error:{message:'authorization Bearer SECRET 401'}}}});
  }
  if(input!=='slow-start'&&input!=='early')ack(m,{turn:{id:turnId}});
  if(input==='hang'||input==='slow-start')return;
  if(input==='approval'){
   send({id:'command',method:'item/commandExecution/requestApproval',params:{threadId,turnId}});
   send({id:'permission',method:'item/permissions/requestApproval',params:{threadId,turnId}});
   send({id:'tool',method:'item/tool/call',params:{threadId,turnId}});
   return setTimeout(()=>finish(threadId,turnId),30);
  }
  if(input==='early'){
   // Turn notifications before the acknowledgement are valid for async clients.
   finish(threadId,turnId,'早い応答');return;
  }
  setTimeout(()=>finish(threadId,turnId),20);
 }
 if(m.method==='turn/interrupt'){
  ack(m,{});send({method:'turn/completed',params:{threadId:m.params.threadId,turn:{id:m.params.turnId,status:'interrupted'}}});
 }
});
`;
  await writeFile(bin, source); await chmod(bin, 0o700);
  const provider = createProvider({ provider: 'codex', codexBin: bin, codexHome: join(dir, 'home'), workspace: dir, timeoutMs });
  t.after(async () => { await provider.close(); await rm(dir, { recursive: true, force: true }); });
  return { provider, log };
}

const options = (input = 'hello') => ({ key: 'room:pm', instructions: '日本語で返してください。', input });

test('status exposes only authentication boolean; stream and role thread reuse', async (t) => {
  const { provider, log } = await fixture(t);
  assert.deepEqual(await provider.status(), { ready: true, authenticated: true });
  let streamed = '';
  const response = await provider.generate({ ...options(), onDelta: (part) => { streamed += part; } });
  assert.equal(response.text, '本文'); assert.equal(streamed, '本文'); assert.equal(response.model, 'test-model');
  const next = await provider.generate(options());
  assert.equal(next.threadId, response.threadId);
  const requests = (await readFile(log, 'utf8')).trim().split('\n').map((line) => JSON.parse(line));
  const startup = requests.find((m) => m.method === 'fixture/process');
  assert.ok(startup.args.includes('cli_auth_credentials_store="file"'));
  assert.equal(startup.home, startup.codexHome);
  assert.ok(startup.codexHome.endsWith('/home'));
  assert.equal(requests.filter((m) => m.method === 'thread/start').length, 1);
  assert.equal(requests.filter((m) => m.method === 'thread/resume').length, 1);
  assert.equal(requests.find((m) => m.method === 'thread/start').params.approvalPolicy, 'never');
  assert.deepEqual(requests.find((m) => m.method === 'turn/start').params.sandboxPolicy, {type:'readOnly'});
});

test('completion does not require late acknowledgement or include reasoning', async (t) => {
  const { provider } = await fixture(t);
  const result = await provider.generate(options('early'));
  assert.equal(result.text, '早い応答');
});

test('explicit cancellation interrupts active Codex turn', async (t) => {
  const { provider, log } = await fixture(t);
  await provider.status();
  const controller = new AbortController();
  const result = provider.generate({ ...options('hang'), signal: controller.signal });
  setTimeout(() => controller.abort(), 100);
  await assert.rejects(result, { name: 'AbortError' });
  await new Promise((resolve) => setTimeout(resolve, 30));
  const messages = (await readFile(log, 'utf8')).trim().split('\n').map((line) => JSON.parse(line));
  assert.ok(messages.some((m) => m.method === 'turn/interrupt'));
  assert.equal((await provider.generate(options())).text, '本文');
});

test('cancel before turn/start acknowledgement returns immediately', async (t) => {
  const { provider } = await fixture(t);
  await provider.status();
  const controller = new AbortController();
  const result = provider.generate({ ...options('slow-start'), signal: controller.signal });
  setTimeout(() => controller.abort(), 50);
  await assert.rejects(result, { name: 'AbortError' });
});

test('generation deadline rejects and a process exit rejects active work', async (t) => {
  const { provider } = await fixture(t, 180);
  await provider.status();
  await assert.rejects(provider.generate(options('hang')), { name: 'TimeoutError' });
  await assert.rejects(provider.generate(options('crash')), /接続が終了/);
  assert.equal((await provider.generate(options())).text, '本文');
});

test('all server-side execution/permission/dynamic tool requests are refused', async (t) => {
  const { provider, log } = await fixture(t);
  await provider.generate(options('approval'));
  const requests = (await readFile(log, 'utf8')).trim().split('\n').map((line) => JSON.parse(line));
  assert.equal(requests.find((m) => m.id === 'command').result.decision, 'decline');
  assert.deepEqual(requests.find((m) => m.id === 'permission').result.permissions, {});
  assert.equal(requests.find((m) => m.id === 'tool').result.success, false);
});

test('failed turns and RPC errors never expose arbitrary provider text', async (t) => {
  const { provider } = await fixture(t);
  for (const input of ['fail', 'rpc-error']) {
    await assert.rejects(provider.generate(options(input)), (error: unknown) => {
      assert.ok(error instanceof Error); assert.match(error.message, /認証/); assert.ok(!error.message.includes('SECRET'));
      return true;
    });
  }
});

test('close rejects work; already-aborted input never starts a turn', async (t) => {
  const { provider } = await fixture(t);
  await assert.rejects(provider.generate({ ...options(), signal: AbortSignal.abort() }), { name: 'AbortError' });
  await provider.status();
  const pending = provider.generate(options('hang'));
  await new Promise((resolve) => setTimeout(resolve, 30));
  await provider.close();
  await assert.rejects(pending);
  assert.equal((await provider.status()).ready, false);
});

test('mock is explicit, streams, supports planner schema and respects cancellation', async () => {
  const provider: AiProvider = createProvider({ provider: 'mock', codexBin: '', codexHome: '', workspace: '', timeoutMs: 1000 });
  const schema = { type: 'object', properties: { reply: { type: 'string' }, participants: { type: 'array' } } };
  const normal = await provider.generate({ ...options(), schema });
  assert.deepEqual(JSON.parse(normal.text).participants, []);
  const discuss = await provider.generate({ ...options('[DISCUSS=true]'), schema });
  assert.deepEqual(JSON.parse(discuss.text).participants, ['evaluator', 'reviewer']);
  const controller = new AbortController();
  const running = provider.generate({ ...options(), signal: controller.signal });
  controller.abort();
  await assert.rejects(running, { name: 'AbortError' });
  await provider.close();
});

test('protocol validation errors identify the RPC method without leaking provider text', async (t) => {
  const {provider}=await fixture(t);
  await assert.rejects(provider.generate(options('protocol-error')),(error:unknown)=>{
    assert.ok(error instanceof Error);assert.match(error.message,/turn\/start/);assert.match(error.message,/-32600/);assert.ok(!error.message.includes('SECRET'));return true;
  });
});
