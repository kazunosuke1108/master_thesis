import {createServer,type IncomingMessage,type ServerResponse} from 'node:http';
import {resolve} from 'node:path';
import {homedir} from 'node:os';
import {pathToFileURL} from 'node:url';
import {Store} from './store.js';
import {Context} from './context.js';
import {Documents,ConflictError} from './documents.js';
import {Chat,BusyError,safeError} from './chat.js';
import {createProvider,type AiProvider} from './ai.js';
import {roles,requireString,reqId,type Role,type Room} from './types.js';
export interface Config {workspace:string;dataDir:string;provider:'codex'|'mock';codexBin:string;codexHome:string;model?:string;timeoutMs:number;commonInstructions:string;commonWorkflow:string;port:number;host:string;allowedHosts:string[];lintScript?:string}
export function configFromEnv():Config {
  const provider=process.env.AI_PROVIDER??'codex';if(!['codex','mock'].includes(provider))throw new Error('AI_PROVIDERはcodexまたはmockです。');
  return {workspace:resolve(process.env.RESEARCH_WORKSPACE??'../../..'),dataDir:resolve(process.env.DATA_DIR??'../.local/data'),provider:provider as Config['provider'],codexBin:process.env.CODEX_BIN??'codex',codexHome:resolve(process.env.CODEX_HOME??'../.local/codex'),model:process.env.CODEX_MODEL||undefined,timeoutMs:Number(process.env.AI_TIMEOUT_MS??120000),commonInstructions:process.env.COMMON_AGENTS_PATH??resolve(homedir(),'.codex/AGENTS.md'),commonWorkflow:process.env.COMMON_WORKFLOW_PATH??resolve(homedir(),'.codex/research-docs/requirements-workflow.md'),lintScript:process.env.YOMIYASU_SCRIPT??resolve(homedir(),'.agents/skills/yomiyasu/scripts/yomiyasu_lint.py'),port:Number(process.env.PORT??8787),host:process.env.HOST??'127.0.0.1',allowedHosts:(process.env.ALLOWED_HOSTS??'localhost,127.0.0.1,[::1]').split(',').map(h=>h.trim())};
}
async function body(req:IncomingMessage):Promise<Record<string,unknown>> {
  if(!req.headers['content-type']?.startsWith('application/json'))throw new Error('Content-Type: application/jsonで送信してください。');
  let content='';for await(const part of req){content+=part.toString();if(Buffer.byteLength(content)>100000)throw new Error('入力が大きすぎます。');}
  let data;try{data=JSON.parse(content||'{}');}catch{throw new Error('JSON形式を確認してください。');}
  if(!data||typeof data!=='object'||Array.isArray(data))throw new Error('JSONオブジェクトを送信してください。');return data;
}
function json(res:ServerResponse,status:number,value:unknown) {res.writeHead(status,{'Content-Type':'application/json; charset=utf-8','Cache-Control':'no-store','X-Content-Type-Options':'nosniff'});res.end(JSON.stringify(value));}
export function createApp(config:Config,provider?:AiProvider) {
  const ai=provider??createProvider({provider:config.provider,codexBin:config.codexBin,codexHome:config.codexHome,workspace:config.workspace,model:config.model,timeoutMs:config.timeoutMs});
  const store=new Store(resolve(config.dataDir,'chat.sqlite'));const context=new Context(config.workspace,config.commonInstructions,config.commonWorkflow);const documents=new Documents(config.workspace,config.lintScript);const chat=new Chat(store,context,documents,ai);
  const server=createServer(async(req,res)=>{
    try {
      const host=req.headers.host;if(!host) {json(res,403,{error:'接続先を確認してください。'});return;}
      const url=new URL(req.url??'/',`http://${host}`);
      if(!config.allowedHosts.includes(url.hostname)){json(res,403,{error:'許可されていないホストです。'});return;}
      const origin=req.headers.origin;
      if(origin && origin!==`http://${host}` && origin!==`https://${host}`){json(res,403,{error:'別のサイトからは操作できません。'});return;}
      // Requests without Origin from an ordinary command line are supported; reject cross-site browser fetches.
      if(req.headers['sec-fetch-site']==='cross-site'){json(res,403,{error:'別のサイトからは操作できません。'});return;}
      const path=url.pathname;const method=req.method??'GET';let m:RegExpMatchArray|null;
      if(method==='GET'&&path==='/api/health'){json(res,200,{status:'ok',provider:config.provider,ai:await ai.status(),roles,version:'0.1.0'});return;}
      if(method==='GET'&&path==='/api/sources'){json(res,200,{sources:await context.catalog()});return;}
      if(method==='GET'&&path==='/api/rooms'){json(res,200,{rooms:store.list<Room>('room').reverse()});return;}
      if(method==='POST'&&path==='/api/rooms') {
        const data=await body(req),id=reqId(data.reqId??'REQ-005');
        const room=store.room(requireString(data.title,'話題',120),id);
        if((await context.catalog()).some(s=>s.path==='paper/latest/paper_full_draft_v18.pdf')) {room.paths=['paper/latest/paper_full_draft_v18.pdf'];store.put('room',room);}
        json(res,201,{room});return;
      }
      if((m=path.match(/^\/api\/rooms\/([\w-]+)$/))&&method==='GET'){json(res,200,await chat.snapshot(m[1]));return;}
      if((m=path.match(/^\/api\/rooms\/([\w-]+)\/events$/))&&method==='GET') {
        const roomId=m[1];chat.room(roomId);
        const after=Number(req.headers['last-event-id']??url.searchParams.get('after')??0);
        if(!Number.isSafeInteger(after)||after<0)throw new Error('イベントIDを確認してください。');
        res.writeHead(200,{'Content-Type':'text/event-stream; charset=utf-8','Cache-Control':'no-cache','Connection':'keep-alive','X-Accel-Buffering':'no'});
        const send=(e:unknown)=>{const event=e as {id:number};res.write(`id: ${event.id}\nevent: update\ndata: ${JSON.stringify(event)}\n\n`);};
        const unsubscribe=chat.subscribe(roomId,send);
        for(const e of store.events(roomId,after))send(e);
        res.write(': connected\n\n');const keepalive=setInterval(()=>res.write(': keepalive\n\n'),15000);
        req.on('close',()=>{clearInterval(keepalive);unsubscribe();});return;
      }
      if((m=path.match(/^\/api\/rooms\/([\w-]+)\/sources$/))&&method==='POST'){const data=await body(req);await chat.updateSources(m[1],data.paths);json(res,200,await chat.snapshot(m[1]));return;}
      if((m=path.match(/^\/api\/rooms\/([\w-]+)\/messages$/))&&method==='POST') {
        const data=await body(req);const content=requireString(data.content,'発言',12000);const clientId=requireString(data.clientId,'送信ID',128);
        if(!/^[\w-]+$/.test(clientId))throw new Error('送信IDを確認してください。');
        if(data.recipients!==undefined && (!Array.isArray(data.recipients)||!data.recipients.every(x=>roles.some(r=>r.id===x))))throw new Error('担当名を確認してください。');
        let recipients=(data.recipients??[]) as Role[];
        if(!recipients.length)recipients=[...content.matchAll(/@(pm|researcher|architect|implementer|evaluator|reviewer|editor)\b/g)].map(x=>x[1] as Role);
        recipients=[...new Set(recipients)];if(recipients.length>2)throw new Error('指名できる担当は2名までです。');
        if(data.discuss!==undefined&&typeof data.discuss!=='boolean')throw new Error('議論モードを確認してください。');
        const discuss=data.discuss===true||/みんな|皆の意見|議論して|複数の意見/.test(content);
        json(res,202,chat.enqueue(m[1],content,clientId,recipients,discuss));return;
      }
      if((m=path.match(/^\/api\/runs\/([\w-]+)\/(stop|retry)$/))&&method==='POST') {
        await body(req);json(res,200,m[2]==='stop'?{run:chat.stop(m[1])}:chat.retry(m[1]));return;
      }
      if((m=path.match(/^\/api\/rooms\/([\w-]+)\/drafts$/))&&method==='POST') {
        const data=await body(req);if(data.target!=='report'&&data.target!=='checkpoint')throw new Error('保存先を確認してください。');
        json(res,201,{draft:await chat.draft(m[1],data.target)});return;
      }
      if((m=path.match(/^\/api\/drafts\/([\w-]+)(\/save)?$/))&&((m[2]&&method==='POST')||(!m[2]&&method==='PUT'))) {
        const data=await body(req);const content=requireString(data.content,'下書き',20000);
        json(res,200,{draft:m[2]?await chat.saveDraft(m[1],content):await chat.updateDraft(m[1],content)});return;
      }
      json(res,404,{error:'見つかりません。'});
    }catch(e){if(!res.headersSent)json(res,e instanceof ConflictError||e instanceof BusyError?409:400,{error:safeError(e)});else res.end();}
  });
  return {server,chat,store,ai,async close(){server.closeAllConnections();await new Promise<void>(r=>server.close(()=>r()));await chat.close();store.close();}};
}
if(process.argv[1]&&import.meta.url===pathToFileURL(resolve(process.argv[1])).href) {
  const config=configFromEnv();const app=createApp(config);
  app.server.listen(config.port,config.host,()=>console.log(`Research room backend: http://${config.host}:${config.port} (${config.provider})`));
  let closing=false;for(const event of ['SIGINT','SIGTERM'] as const)process.on(event,()=>{if(closing)return;closing=true;void app.close().then(()=>process.exit(0));});
}
