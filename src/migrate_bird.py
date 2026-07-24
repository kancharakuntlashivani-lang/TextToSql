from __future__ import annotations
import argparse, csv, json, math, sqlite3, sys
from dataclasses import dataclass, asdict
from decimal import Decimal
from pathlib import Path
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values
from tqdm import tqdm

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src import config
from src.core import safe_identifier

@dataclass
class Result:
    db_id:str; schema:str; table:str; sqlite_rows:int; postgres_rows:int; status:str; message:str=''

def pg():
    return psycopg2.connect(host=config.DB_HOST,port=config.DB_PORT,dbname=config.DB_NAME,user=config.DB_USER,password=config.DB_PASSWORD)

def dbfile(folder):
    files=[]
    for p in ('*.sqlite','*.sqlite3','*.db'): files += list(folder.rglob(p))
    if not files: raise FileNotFoundError(f'No SQLite file in {folder}')
    return sorted(files,key=lambda x:len(str(x)))[0]

def pgtype(t):
    t=(t or '').upper()
    if 'INT' in t:return 'BIGINT'
    if any(x in t for x in ('CHAR','TEXT','CLOB','VARCHAR')):return 'TEXT'
    if 'BLOB' in t:return 'BYTEA'
    if any(x in t for x in ('REAL','FLOA','DOUB')):return 'DOUBLE PRECISION'
    if any(x in t for x in ('NUMERIC','DECIMAL')):return 'NUMERIC'
    if 'BOOL' in t:return 'BOOLEAN'
    return 'TEXT'

def clean(v,t):
    if v is None:return None
    if isinstance(v,float) and (math.isnan(v) or math.isinf(v)):return None
    if t=='TEXT':
        return (v.decode('utf-8',errors='replace') if isinstance(v,bytes) else str(v)).replace('\x00','')
    if t=='BYTEA':return psycopg2.Binary(v if isinstance(v,bytes) else str(v).encode())
    if t=='BIGINT':
        try:return int(v)
        except:return None
    if t=='DOUBLE PRECISION':
        try:return float(v)
        except:return None
    if t=='NUMERIC':
        try:return Decimal(str(v))
        except:return None
    return v

def migrate(folder,replace,batch_size):
    db_id=folder.name; schema=safe_identifier(db_id); sf=dbfile(folder)
    s=sqlite3.connect(str(sf)); p=pg(); results=[]
    try:
        with p.cursor() as c:
            if replace:c.execute(sql.SQL('DROP SCHEMA IF EXISTS {} CASCADE').format(sql.Identifier(schema)))
            c.execute(sql.SQL('CREATE SCHEMA IF NOT EXISTS {}').format(sql.Identifier(schema)))
        p.commit()
        tables=[r[0] for r in s.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name").fetchall()]
        meta={}
        for table in tables:
            esc=table.replace('"','""'); cols=s.execute(f'PRAGMA table_info("{esc}")').fetchall(); meta[table]=cols
            defs=[sql.SQL('{} {}').format(sql.Identifier(c[1]),sql.SQL(pgtype(c[2]))) for c in cols]
            with p.cursor() as cur:cur.execute(sql.SQL('CREATE TABLE {}.{} ({})').format(sql.Identifier(schema),sql.Identifier(table),sql.SQL(', ').join(defs)))
            p.commit()
        for table in tqdm(tables,desc=db_id):
            esc=table.replace('"','""'); cols=meta[table]; names=[c[1] for c in cols]; types=[pgtype(c[2]) for c in cols]
            source=int(s.execute(f'SELECT COUNT(*) FROM "{esc}"').fetchone()[0]); cursor=s.execute(f'SELECT * FROM "{esc}"')
            ins=sql.SQL('INSERT INTO {}.{} ({}) VALUES %s').format(sql.Identifier(schema),sql.Identifier(table),sql.SQL(', ').join(sql.Identifier(n) for n in names)).as_string(p)
            with p.cursor() as cur:
                while True:
                    batch=cursor.fetchmany(batch_size)
                    if not batch:break
                    execute_values(cur,ins,[tuple(clean(v,types[i]) for i,v in enumerate(row)) for row in batch],page_size=batch_size)
            p.commit()
            with p.cursor() as cur:
                cur.execute(sql.SQL('SELECT COUNT(*) FROM {}.{}').format(sql.Identifier(schema),sql.Identifier(table))); target=int(cur.fetchone()[0])
            results.append(Result(db_id,schema,table,source,target,'SUCCESS' if source==target else 'MISMATCH'))
    finally:s.close();p.close()
    return results

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--database');ap.add_argument('--replace',action='store_true');ap.add_argument('--batch-size',type=int,default=1000);args=ap.parse_args()
    root=config.BIRD_DATA_DIR/'dev_databases'; folders=sorted(p for p in root.iterdir() if p.is_dir())
    if args.database:folders=[p for p in folders if p.name==args.database]
    all=[]
    for folder in folders:all += migrate(folder,args.replace,args.batch_size)
    rows=[asdict(r) for r in all]
    (config.OUTPUT_DIR/'bird_migration_report.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    with (config.OUTPUT_DIR/'bird_migration_report.csv').open('w',newline='',encoding='utf-8') as f:
        wr=csv.DictWriter(f,fieldnames=Result.__annotations__.keys());wr.writeheader();wr.writerows(rows)
    print(json.dumps({'databases':len(set(r.db_id for r in all)),'tables':len(all),'rows':sum(r.postgres_rows for r in all),'failed':sum(r.status!='SUCCESS' for r in all)},indent=2))
if __name__=='__main__':main()
