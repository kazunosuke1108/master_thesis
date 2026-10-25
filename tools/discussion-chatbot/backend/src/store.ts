import {DatabaseSync} from 'node:sqlite';
import {mkdirSync} from 'node:fs';
import {dirname} from 'node:path';
import {randomUUID} from 'node:crypto';
import {now, type Room, type Message, type Run, type Draft} from './types.js';
export class Store {
  db: DatabaseSync;
  constructor(path: string) {
    mkdirSync(dirname(path), {recursive:true, mode:0o700});
    this.db = new DatabaseSync(path);
    this.db.exec(`PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;
      CREATE TABLE IF NOT EXISTS entities (kind TEXT NOT NULL,id TEXT NOT NULL, room_id TEXT NOT NULL, data TEXT NOT NULL, PRIMARY KEY(kind,id));
      CREATE INDEX IF NOT EXISTS entities_room ON entities(kind,room_id);
      CREATE TABLE IF NOT EXISTS dedup (room_id TEXT,client_id TEXT,run_id TEXT, PRIMARY KEY(room_id,client_id));
      CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, room_id TEXT NOT NULL,type TEXT NOT NULL,payload TEXT NOT NULL);
      CREATE INDEX IF NOT EXISTS events_room ON events(room_id,id);`);
    for (const run of this.list<Run>('run')) {
      if (run.status==='running' || run.status==='queued') { run.status='interrupted'; run.error='サーバー再起動により中断しました。再試行できます。'; this.put('run',run); }
    }
    for (const msg of this.list<Message>('message')) if (msg.status==='streaming') {msg.status='interrupted';this.put('message',msg);}
  }
  put(kind:string, data: {id:string;roomId?:string}) {
    this.db.prepare('INSERT INTO entities(kind,id,room_id,data) VALUES (?,?,?,?) ON CONFLICT(kind,id) DO UPDATE SET data=excluded.data,room_id=excluded.room_id').run(kind,data.id,data.roomId??data.id,JSON.stringify(data));
  }
  get<T>(kind:string,id:string):T|undefined {
    const row=this.db.prepare('SELECT data FROM entities WHERE kind=? AND id=?').get(kind,id);
    return row ? JSON.parse(String(row.data)) as T : undefined;
  }
  list<T>(kind:string,roomId?:string):T[] {
    const rows=roomId===undefined ? this.db.prepare('SELECT data FROM entities WHERE kind=? ORDER BY rowid').all(kind) : this.db.prepare('SELECT data FROM entities WHERE kind=? AND room_id=? ORDER BY rowid').all(kind,roomId);
    return rows.map(r=>JSON.parse(String(r.data)) as T);
  }
  room(title:string,reqId:string):Room {
    const room:Room={id:randomUUID(),title,reqId,createdAt:now(),updatedAt:now(),paths:[]};this.put('room',room);return room;
  }
  transaction<T>(f:()=>T):T { this.db.exec('BEGIN IMMEDIATE');try {const r=f();this.db.exec('COMMIT');return r;}catch(e){this.db.exec('ROLLBACK');throw e;} }
  addInput(room:Room,content:string,clientId:string,recipients:Run['recipients'],discuss:boolean):{run:Run;message:Message;duplicate:boolean} {
    return this.transaction(()=>{
      const prior=this.db.prepare('SELECT run_id FROM dedup WHERE room_id=? AND client_id=?').get(room.id,clientId);
      if(prior){const run=this.get<Run>('run',String(prior.run_id))!;const message=this.get<Message>('message',run.messageId)!;return {run,message,duplicate:true};}
      const run:Run={id:randomUUID(),roomId:room.id,messageId:randomUUID(),status:'queued',recipients,discuss,createdAt:now(),completedRoles:[]};
      const message:Message={id:run.messageId,roomId:room.id,role:'user',content,status:'complete',createdAt:now(),runId:run.id};
      this.put('run',run);this.put('message',message);room.updatedAt=now();this.put('room',room);
      this.db.prepare('INSERT INTO dedup VALUES (?,?,?)').run(room.id,clientId,run.id);return {run,message,duplicate:false};
    });
  }
  event(roomId:string,type:string,payload:unknown):number {
    const result=this.db.prepare('INSERT INTO events(room_id,type,payload) VALUES (?,?,?)').run(roomId,type,JSON.stringify(payload));
    const id=Number(result.lastInsertRowid);
    // Keep a bounded replay buffer. Snapshots remain the canonical history.
    this.db.prepare('DELETE FROM events WHERE room_id=? AND id < ?').run(roomId,id-2000);
    return id;
  }
  events(roomId:string,after:number) {
    return this.db.prepare('SELECT id,room_id,type,payload FROM events WHERE room_id=? AND id>? ORDER BY id LIMIT 2001').all(roomId,after).map(r=>({id:Number(r.id),roomId:String(r.room_id),type:String(r.type),payload:JSON.parse(String(r.payload))}));
  }
  close(){this.db.close();}
}
