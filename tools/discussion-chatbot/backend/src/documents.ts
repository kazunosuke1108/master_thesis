import {readdir,readFile,realpath,open,unlink,mkdir} from 'node:fs/promises';
import {join,sep} from 'node:path';
import {tmpdir} from 'node:os';
import {mkdtemp,writeFile,rm} from 'node:fs/promises';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
const exec=promisify(execFile);
import {hash} from './context.js';
import {now,type Draft,type Room} from './types.js';
export class ConflictError extends Error {}
export class Documents {
  constructor(readonly workspace:string,readonly lintScript?:string){}
  async lint(content:string):Promise<string> {
    if(!this.lintScript)return '校正ツール未設定。保存前に本文を確認してください。';
    const dir=await mkdtemp(join(tmpdir(),'research-note-'));
    try {
      const file=join(dir,'note.md');await writeFile(file,content,{mode:0o600});
      const result=await exec('python3',[this.lintScript,file],{timeout:10000,maxBuffer:100000});
      return result.stdout.slice(0,12000);
    }catch{throw new Error('日本語の静的検査を実行できません。校正スクリプトとpython3の配置を確認してください。');}
    finally{await rm(dir,{recursive:true,force:true});}
  }
  private async directory(dir:string) {
    const root=await realpath(this.workspace);const dest=join(root,dir);await mkdir(dest,{recursive:true});
    const actual=await realpath(dest);if(!actual.startsWith(root+sep))throw new Error('保存先が許可範囲外です。');return actual;
  }
  async baseline(target:Draft['target']) {
    const dirs=target==='report'?['requests','reports']:['.codex/research-docs/checkpoints'];
    const entries:string[]=[];
    for(const dir of dirs){const actual=await this.directory(dir);for(const n of (await readdir(actual)).filter(n=>n.endsWith('.md')).sort()) {
      const file=await realpath(join(actual,n));if(!file.startsWith(actual+sep))throw new Error('文書内のシンボリックリンクは保存検査の対象外です。');
      entries.push(`${dir}/${n}:${hash(await readFile(file))}`);
    }}return hash(entries.join('\n'));
  }
  async save(draft:Draft,room:Room,content:string) {
    const reports=await this.directory('reports');const lockPath=join(reports,'.discussion-export.lock');
    let lock;
    try{lock=await open(lockPath,'wx',0o600);}catch{throw new ConflictError('別の保存処理が動いています。終了を待って再試行してください。');}
    try {
      if(await this.baseline(draft.target)!==draft.baseline)throw new ConflictError('下書き作成後に文書が変更されました。資料を確認して下書きを作り直してください。');
      const time=now();const day=time.slice(0,10);let relative:string;
      if(draft.target==='report') {
        const names=[...await readdir(await this.directory('requests')),...await readdir(reports)];
        const numbers=names.map(n=>n.match(new RegExp(`^${day}-(\\d+)-`))).filter(Boolean).map(m=>Number(m![1]));
        const next=String(Math.max(0,...numbers)+1).padStart(3,'0');relative=`reports/${day}-${next}-${room.reqId}-discussion-note.md`;
      }else {
        const timePart=time.slice(11,19).replaceAll(':','');relative=`.codex/research-docs/checkpoints/${day}-${timePart}-${room.reqId}-discussion-${draft.id.slice(0,8)}.md`;
      }
      const proofreading=await this.lint(content);draft.proofreading=proofreading;
      if(await this.baseline(draft.target)!==draft.baseline)throw new ConflictError('校正中に文書が変更されました。下書きを作り直してください。');
      const refs=(draft.sources??[]).map(s=>`| ${s.path.replace(/[|\r\n]/g,' ')} | ${s.hash} | ${s.unread?'上限により未読':s.truncated?'一部を参照':'全体'} |`).join('\n');
      const document=`---\nrequest_id: ${room.reqId}\nstatus: discussion\nrevision: 1\nissued_at: "${time}"\nupdated_at: "${time}"\n---\n\n# ${room.reqId} 研究相談メモ\n\n本書は研究責任者が保存を指示した議論の要点です。保存は研究方針・設計・評価条件の承認を意味しません。採用には対象REQ・版を示した明示的な判断が必要です。\n\n${content}\n\n## 参照と来歴\n\n- 下書きID: ${draft.id}\n- 会話中の参照発言ID: ${draft.sourceIds.join(', ')}\n- コード基準版: ${draft.codeRevision??'unknown'}
- 参照資料の構成ハッシュ: ${draft.sourceHash}\n- 発生日時: ${draft.createdAt}\n- 記録日時: ${time}\n\n## 参照資料の版\n\n| パス | SHA-256 | 読取範囲 |\n| --- | --- | --- |\n${refs}\n\n## 校正\n\n${this.lintScript?'日本語の静的検査を実行しました。機械検査の指摘は候補であり、研究内容の妥当性を保証しません。':'日本語の静的検査は未実施です。校正ツールが設定されていません。'}研究責任者の内容確認操作により保存しました。\n\n## 変更履歴\n\n| 発生日時 | 記録日時 | 担当 | 対象版 | 内容 |\n| --- | --- | --- | --- | --- |\n| ${draft.createdAt} | ${time} | PM・研究責任者 | 1 | 議論の要点を下書きから確認して保存。採用は未決 |\n`;
      const file=await open(join(this.workspace,relative),'wx',0o644);
      try {await file.writeFile(document);await file.sync();}finally{await file.close();}
      return relative;
    }finally {await lock.close();await unlink(lockPath);}
  }
}
