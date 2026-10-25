import {readFile, readdir, realpath, stat, lstat} from 'node:fs/promises';
import {join, resolve, sep, extname} from 'node:path';
import {createHash} from 'node:crypto';
import {execFile,execFileSync} from 'node:child_process';
import {promisify} from 'node:util';
import type {Room, Source} from './types.js';
const exec=promisify(execFile);
export const hash=(s:string|Buffer)=>createHash('sha256').update(s).digest('hex');
export class Context {
  readonly codeRevision:string;

  constructor(readonly workspace:string,readonly commonInstructions:string,readonly commonWorkflow:string){
    try {this.codeRevision=process.env.RESEARCH_REVISION||execFileSync('git',['-C',workspace,'rev-parse','HEAD'],{encoding:'utf8',timeout:2000,stdio:['ignore','pipe','ignore']}).trim();}catch{this.codeRevision='unknown';}
  }
  private async safe(path:string) {
    const root=await realpath(this.workspace); const actual=await realpath(resolve(root,path));
    if(!actual.startsWith(root+sep)) throw new Error('参照先が許可範囲外です。');
    return actual;
  }
  async catalog():Promise<{path:string;label:string;automatic?:boolean}[]> {
    const result:{path:string;label:string;automatic?:boolean}[]=[];
    for(const dir of ['requests','reports']) {
      for(const name of await readdir(join(this.workspace,dir)).catch(()=>[])) if(name.endsWith('.md')) result.push({path:`${dir}/${name}`,label:name});
    }
    const references:string[]=[];
    const referenceRoot=join(this.workspace,'reference');
    // Reference folders and files are traversed without following symlinks.
    if((await lstat(referenceRoot).catch(()=>null))?.isDirectory()) {
      const walk=async(relative:string,depth:number):Promise<void>=>{
        if(depth>8)return;
        const entries=await readdir(join(this.workspace,relative),{withFileTypes:true}).catch(()=>[]);
        for(const entry of entries) {
          if(entry.name.startsWith('.')||entry.isSymbolicLink())continue;
          const path=`${relative}/${entry.name}`;
          if(entry.isDirectory())await walk(path,depth+1);
          else if(entry.isFile()&&['.md','.txt','.pdf'].includes(extname(entry.name).toLowerCase()))references.push(path);
        }
      };
      await walk('reference',0);
    }
    references.sort();
    references.forEach((path,index)=>result.push({path,label:path.slice('reference/'.length),automatic:index<8}));
    for(const path of ['AGENTS.md','.codex/research-docs/requirements-workflow.md','paper/latest/paper_full_draft_v18.pdf']) {
      if(await stat(join(this.workspace,path)).catch(()=>null))result.push({path,label:path.split('/').at(-1)!});
    }
    return result.sort((a,b)=>a.path.localeCompare(b.path));
  }
  async validate(paths:unknown):Promise<string[]> {
    if(!Array.isArray(paths)||paths.length>8||!paths.every(p=>typeof p==='string'))throw new Error('参照資料は8件まで選択してください。');
    const allowed=new Set((await this.catalog()).map(x=>x.path));
    for(const p of paths)if(!allowed.has(p))throw new Error('許可されていない参照資料です。');
    return [...new Set(paths)] as string[];
  }
  async load(room:Room):Promise<{sources:Source[];text:string;warning?:string}> {
    const sources:Source[]=[];const warnings:string[]=[];
    const add=async(label:string,path:string,isExternal=false)=>{
      try {
        if(!isExternal&&path.startsWith('reference/')) {
          let current=this.workspace;
          for(const part of path.split('/')) {current=join(current,part);if((await lstat(current)).isSymbolicLink())throw new Error('参照資料のsymlinkは読みません');}
        }
        const file=isExternal?path:await this.safe(path);
        const info=await stat(file);if(info.size>20_000_000)throw new Error('資料が大きすぎます');
        const raw=await readFile(file);
        let text:string;
        if(extname(file).toLowerCase()==='.pdf') {
          const result=await exec('pdftotext',['-layout',file,'-'],{maxBuffer:2_000_000,timeout:10000});text=result.stdout;
        }else text=raw.toString('utf8');
        const cap=label.startsWith('instructions/')?16000:22000;
        const truncated=text.length>cap;
        sources.push({path:label,hash:hash(raw),text:text.slice(0,cap),truncated});
        if(truncated)warnings.push(`${label} は先頭${cap}文字まで参照`);
      } catch {warnings.push(`${label} を読み込めませんでした。配置またはPDF抽出ツールを確認してください。`);}
    };
    if(this.commonInstructions)await add('instructions/common-AGENTS.md',this.commonInstructions,true);
    if(this.commonWorkflow)await add('instructions/requirements-workflow.md',this.commonWorkflow,true);
    await add('AGENTS.md','AGENTS.md');await add('.codex/research-docs/requirements-workflow.md','.codex/research-docs/requirements-workflow.md');
    const catalog=await this.catalog();
    const files=catalog.filter(x=>x.path.startsWith('requests/')&&x.path.includes(room.reqId));
    if(files.length!==1)warnings.push(`${room.reqId} の要求書を一意に特定できませんでした。`);
    for(const f of files.slice(0,1))await add(f.path,f.path);
    const cpDir='.codex/research-docs/checkpoints';
    const checkpoints=(await readdir(join(this.workspace,cpDir)).catch(()=>[])).filter(x=>x.endsWith('.md')&&x.includes(room.reqId)).sort();
    if(checkpoints.length)await add(`${cpDir}/${checkpoints.at(-1)!}`,`${cpDir}/${checkpoints.at(-1)!}`);
    const shared=catalog.filter(s=>s.path.startsWith('reference/')&&s.automatic);
    if(catalog.filter(s=>s.path.startsWith('reference/')).length>shared.length)warnings.push('reference/の共通参照はパス順で先頭8件です。残りは参照資料から選択できます。');
    // Explicitly selected material takes precedence over automatic shared files.
    for(const path of room.paths)if(!sources.some(s=>s.path===path)) {
      if(path.startsWith('reference/')&&!catalog.some(s=>s.path===path)){warnings.push(`${path} は現在の参照対象にないため読み込んでいません。`);continue;}
      await add(path,path);
    }
    for(const source of shared)if(!sources.some(s=>s.path===source.path))await add(source.path,source.path);
    let budget=100000;
    const bounded=sources.map(s=>{
      const text=s.text.slice(0,Math.max(0,budget));budget-=text.length;
      if(text.length<s.text.length){warnings.push(`${s.path} は会話の資料量上限により${text.length?'一部のみ参照':'未読'}`);return {...s,text,truncated:true,unread:text.length===0};}return s;
    });
    return {sources:bounded,text:bounded.map(s=>`\n<reference path=${JSON.stringify(s.path)} hash="${s.hash}" truncated="${s.truncated}">\n${s.text}\n</reference>`).join('\n'),warning:warnings.join('\n')||undefined};
  }
}
