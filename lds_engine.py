from __future__ import annotations
from dataclasses import dataclass
from io import BytesIO
import math, re
from typing import Any
import numpy as np
import pandas as pd

BATCH_COLUMNS=["Batch / Lot","Supplier","Material / Grade","Underlayer","Sample level","Production date","Pad drawing","Backplate drawing","Process status","Reference batch","4M change","Process deviation","Notes","Source sheet","Sequence"]
PHYSICAL_COLUMNS=["Batch / Lot","Test","Sample ID","Result","Result numeric","Unit","LSL","USL","Target","Status","Source sheet","Source row"]
DYNO_COLUMNS=["Batch / Lot","Test","Metric","Run","Result","Result numeric","Unit","Min","Max","Target","Status","Source sheet","Source row"]
UNMAPPED_COLUMNS=["Source sheet","Source row","Descriptor","Value","Reason"]

PHYSICAL_KEYS=["compress","k3","heat transfer","shear","peel","density","thickness","natural frequency","frequency","xrf","ph","tga","moisture","porosity","hardness","backplate","underlayer distribution","pull","corrosion"]
DYNO_KEYS=["ak-master","ak master","ams","hams","static mu","static μ","wear","mojacar","epb","holding torque","wltp","emission","krauss","ece","r13","salt spray","water spray","stiction","dtv","btv","flaming","mpu","fade","recovery","friction","mu "," μ","groan","judder","nvh"]
META_MAP={
    "supplier":"Supplier","material":"Material / Grade","material / grade":"Material / Grade","grade":"Material / Grade","friction material":"Material / Grade",
    "underlayer":"Underlayer","ul":"Underlayer","sample":"Sample level","sample level":"Sample level","production date":"Production date","date":"Production date",
    "pad drawing":"Pad drawing","drawing":"Pad drawing","backplate drawing":"Backplate drawing","process status":"Process status","process":"Process status",
}

@dataclass
class LDSData:
    batches: pd.DataFrame
    physicals: pd.DataFrame
    dyno: pd.DataFrame
    unmapped: pd.DataFrame
    diagnostics: pd.DataFrame

def s(v: Any)->str:
    if v is None: return ""
    try:
        if pd.isna(v): return ""
    except Exception: pass
    x=str(v).strip()
    return "" if x.lower() in {"nan","none"} else x

def num(v: Any):
    if v is None: return None
    if isinstance(v,(int,float,np.integer,np.floating)):
        try: return float(v) if math.isfinite(float(v)) else None
        except: return None
    t=s(v).replace("\u2212","-").replace(",",".")
    m=re.search(r"[-+]?\d+(?:\.\d+)?",t)
    if not m: return None
    try: return float(m.group())
    except: return None

def norm(t: Any)->str:
    return re.sub(r"\s+"," ",s(t).lower()).strip()

def is_batch_header(v: Any)->bool:
    t=s(v); l=t.lower()
    if not t or len(t)>40: return False
    rejects=["lsl","usl","target","unit","result","value","test","parameter","characteristic","metric","supplier","material","batch","lot","notes","status","min","max","mean","date","source"]
    if any(l==r or l.startswith(r+" ") for r in rejects): return False
    if re.fullmatch(r"\d{4,10}",t): return True
    if re.fullmatch(r"[A-Za-z]{1,6}[-_/]?\d{2,12}[A-Za-z0-9-_/]*",t): return True
    if re.fullmatch(r"\d{2,10}[-_/][A-Za-z0-9-_/]+",t): return True
    return False

def role_for(text: str, sheet_name: str=""):
    z=(norm(text)+" "+norm(sheet_name))
    if any(k in z for k in DYNO_KEYS): return "dyno"
    if any(k in z for k in PHYSICAL_KEYS): return "physical"
    sn=norm(sheet_name)
    if "dyno" in sn or "performance" in sn: return "dyno"
    if "phys" in sn or "labor" in sn or "lab" in sn: return "physical"
    return "unknown"

def status(value, lo=None, hi=None):
    if s(value)=="": return "OPEN"
    r=num(value); l=num(lo); h=num(hi)
    if r is None: return "RESULT"
    if l is not None and r<l: return "FAIL"
    if h is not None and r>h: return "FAIL"
    if l is not None or h is not None: return "PASS"
    return "RESULT"

def _trim(raw: pd.DataFrame)->pd.DataFrame:
    d=raw.copy().replace(r"^\s*$",np.nan,regex=True).dropna(how="all").dropna(axis=1,how="all")
    return d.reset_index(drop=True)

def _header_score(row)->float:
    vals=[norm(x) for x in row if s(x)]
    if not vals:return -1
    kws=["batch","lot","test","parameter","characteristic","result","value","unit","lsl","usl","min","max","target","supplier","material","metric","run"]
    kw=sum(any(k in v for k in kws) for v in vals)
    batches=sum(is_batch_header(x) for x in row)
    return kw*3+batches*2+min(len(vals),10)*0.15

def header_row(raw: pd.DataFrame)->int:
    n=min(30,len(raw))
    scores=[_header_score(raw.iloc[i].tolist()) for i in range(n)]
    return int(np.argmax(scores)) if scores else 0

def _unique_headers(vals):
    out=[]; seen={}
    for i,v in enumerate(vals):
        x=s(v) or f"Column {i+1}"
        if x in seen:
            seen[x]+=1; x=f"{x} {seen[x]}"
        else: seen[x]=1
        out.append(x)
    return out

def _col_find(cols, aliases):
    for c in cols:
        n=norm(c)
        if any(n==a or a in n for a in aliases): return c
    return None

def _batch_master_parse(table: pd.DataFrame,sheet:str,seq_start:int):
    cols=list(table.columns)
    batch=_col_find(cols,["batch / lot","batch","lot","batch no","batch number"])
    supplier=_col_find(cols,["supplier"])
    material=_col_find(cols,["material / grade","material","grade"])
    # A pure batch-master table does not need Test/Result columns.
    if not batch or (supplier is None and material is None and norm(sheet)!="batches"):
        return [],[],[],[],seq_start
    field_aliases={
        "Supplier":["supplier"],
        "Material / Grade":["material / grade","material","grade"],
        "Underlayer":["underlayer","ul"],
        "Sample level":["sample level","sample"],
        "Production date":["production date","date"],
        "Pad drawing":["pad drawing"],
        "Backplate drawing":["backplate drawing"],
        "Process status":["process status"],
        "Reference batch":["reference batch"],
        "4M change":["4m change","4m"],
        "Process deviation":["process deviation","deviation"],
        "Notes":["notes","comment","remarks"],
    }
    mapped={field:_col_find(cols,aliases) for field,aliases in field_aliases.items()}
    rows=[];seq=seq_start
    for _,row in table.iterrows():
        b=s(row.get(batch))
        if not b: continue
        seq+=1
        rec={"Batch / Lot":b,"Source sheet":sheet,"Sequence":seq}
        for field,col in mapped.items():
            rec[field]=s(row.get(col)) if col else ""
        rows.append(rec)
    return rows,[],[],[],seq

def _long_parse(table: pd.DataFrame,sheet:str,seq_start:int):
    cols=list(table.columns)
    batch=_col_find(cols,["batch","lot","batch / lot","batch no","batch number"])
    test=_col_find(cols,["test","parameter","characteristic"])
    value=_col_find(cols,["result","value","measured","measurement"])
    if not(batch and test and value): return [],[],[],[],seq_start
    unit=_col_find(cols,["unit"]); lsl=_col_find(cols,["lsl","lower"]); usl=_col_find(cols,["usl","upper"]); target=_col_find(cols,["target","nominal"])
    metric=_col_find(cols,["metric"]); run=_col_find(cols,["run","repeat"]); sample_id=_col_find(cols,["sample id","pad id","sample"]); supplier=_col_find(cols,["supplier"]); material=_col_find(cols,["material","grade"]); underlayer=_col_find(cols,["underlayer","ul"])
    batches=[]; phys=[]; dyno=[]; unm=[]; seq=seq_start; seen={}
    for ri,row in table.iterrows():
        b=s(row.get(batch)); t=s(row.get(test)); v=row.get(value)
        if not b or not t: continue
        if b not in seen:
            seq+=1; seen[b]=seq
            batches.append({"Batch / Lot":b,"Supplier":s(row.get(supplier)) if supplier else "","Material / Grade":s(row.get(material)) if material else "","Underlayer":s(row.get(underlayer)) if underlayer else "","Source sheet":sheet,"Sequence":seq})
        role=role_for(t,sheet)
        common={"Batch / Lot":b,"Test":t,"Sample ID":s(row.get(sample_id)) if sample_id else "","Result":v,"Result numeric":num(v),"Unit":s(row.get(unit)) if unit else "","Target":num(row.get(target)) if target else None,"Status":None,"Source sheet":sheet,"Source row":int(ri)+2}
        if role=="dyno":
            rec={**common,"Metric":s(row.get(metric)) if metric else t,"Run":s(row.get(run)) if run else "","Min":num(row.get(lsl)) if lsl else None,"Max":num(row.get(usl)) if usl else None}
            rec["Status"]=status(v,rec["Min"],rec["Max"]); dyno.append(rec)
        elif role=="physical":
            rec={**common,"LSL":num(row.get(lsl)) if lsl else None,"USL":num(row.get(usl)) if usl else None}
            rec["Status"]=status(v,rec["LSL"],rec["USL"]); phys.append(rec)
        else:
            unm.append({"Source sheet":sheet,"Source row":int(ri)+2,"Descriptor":t,"Value":v,"Reason":"Unclassified long-format test"})
    return batches,phys,dyno,unm,seq

def _row_table_parse(raw: pd.DataFrame,sheet:str,seq_start:int):
    h=header_row(raw); headers=_unique_headers(raw.iloc[h].tolist()); table=raw.iloc[h+1:].copy(); table.columns=headers; table=table.dropna(how="all")
    cols=list(table.columns); batch=_col_find(cols,["batch","lot","batch / lot","batch no","batch number"])
    if not batch: return [],[],[],[],seq_start
    idcols={batch}
    for aliases in [["supplier"],["material","grade"],["underlayer","ul"],["sample level"],["production date"],["status"],["notes"]]:
        c=_col_find(cols,aliases)
        if c:idcols.add(c)
    measure=[c for c in cols if c not in idcols]
    classed=[c for c in measure if role_for(c,sheet)!="unknown"]
    if len(classed)<1:return [],[],[],[],seq_start
    supplier=_col_find(cols,["supplier"]); material=_col_find(cols,["material","grade"]); underlayer=_col_find(cols,["underlayer","ul"]); sample=_col_find(cols,["sample level","sample"]); date=_col_find(cols,["production date","date"])
    batches=[];phys=[];dyno=[];unm=[];seq=seq_start
    for ri,row in table.iterrows():
        b=s(row.get(batch))
        if not b:continue
        seq+=1
        batches.append({"Batch / Lot":b,"Supplier":s(row.get(supplier)) if supplier else "","Material / Grade":s(row.get(material)) if material else "","Underlayer":s(row.get(underlayer)) if underlayer else "","Sample level":s(row.get(sample)) if sample else "","Production date":s(row.get(date)) if date else "","Source sheet":sheet,"Sequence":seq})
        for c in classed:
            v=row.get(c); role=role_for(c,sheet)
            if role=="physical":
                phys.append({"Batch / Lot":b,"Test":str(c),"Sample ID":"","Result":v,"Result numeric":num(v),"Unit":"","LSL":None,"USL":None,"Target":None,"Status":status(v),"Source sheet":sheet,"Source row":int(ri)+h+2})
            else:
                dyno.append({"Batch / Lot":b,"Test":str(c),"Metric":str(c),"Run":"","Result":v,"Result numeric":num(v),"Unit":"","Min":None,"Max":None,"Target":None,"Status":status(v),"Source sheet":sheet,"Source row":int(ri)+h+2})
    return batches,phys,dyno,unm,seq

def _wide_parse(raw: pd.DataFrame,sheet:str,seq_start:int):
    best=None
    for i in range(min(35,len(raw))):
        vals=raw.iloc[i].tolist(); idx=[j for j,v in enumerate(vals) if is_batch_header(v)]
        if len(idx)>=2 and (best is None or len(idx)>len(best[1])): best=(i,idx)
    if best is None:return [],[],[],[],seq_start

    h,bidx=best
    batch_names={j:s(raw.iat[h,j]) for j in bidx}

    # Astemo-style LDS uses batch IDs on one row and "Result / Status / Report Ref."
    # directly below. Requirement headers (Characteristic, LSL, Target, USL, Unit)
    # are therefore taken from the sub-header row when present.
    sub_h=h+1 if h+1 < len(raw) else h
    subheaders=raw.iloc[sub_h].tolist()

    # Keep only true result columns for each batch group. A batch ID should have a
    # Result sub-header below it; generic wide sheets without a second header still work.
    result_idx=[]
    filtered_names={}
    for j,b in batch_names.items():
        sh=norm(subheaders[j]) if j < len(subheaders) else ""
        if sh in {"result","value","measurement","measured"} or sub_h==h:
            result_idx.append(j); filtered_names[j]=b
        else:
            # In merged-cell imports the batch ID lands on the first column of the group,
            # which is normally the Result column even if the sub-header is blank.
            if j+1 < raw.shape[1] and norm(subheaders[j+1])=="status":
                result_idx.append(j); filtered_names[j]=b
    if result_idx:
        bidx=result_idx; batch_names=filtered_names

    pre=[j for j in range(raw.shape[1]) if j not in bidx]
    if not pre:return [],[],[],[],seq_start

    header_norm={j:norm(subheaders[j]) for j in pre}
    desc=next((j for j in pre if "characteristic" in header_norm[j] or header_norm[j] in {"test","parameter","characteristic / test"}),None)
    if desc is None:
        def text_score(j):
            vals=[s(v) for v in raw.iloc[sub_h+1:,j].tolist()]
            return sum(bool(v) and num(v) is None for v in vals)
        desc=max(pre,key=text_score)

    unit_col=next((j for j in pre if "unit" in header_norm[j]),None)
    lsl_col=next((j for j in pre if header_norm[j] in {"lsl","lsl / min","min","minimum","lower"} or "lsl" in header_norm[j]),None)
    usl_col=next((j for j in pre if header_norm[j] in {"usl","usl / max","max","maximum","upper"} or "usl" in header_norm[j]),None)
    target_col=next((j for j in pre if "target" in header_norm[j] or "nominal" in header_norm[j]),None)

    batch_meta={b:{"Batch / Lot":b,"Source sheet":sheet} for b in batch_names.values()}
    phys=[];dyno=[];unm=[]
    data_start=sub_h+1

    for ri in range(data_start,len(raw)):
        descriptor=s(raw.iat[ri,desc])
        if not descriptor: continue
        nd=norm(descriptor)

        # Batch/material passport rows.
        meta_field=None
        for k,v in META_MAP.items():
            if nd==k or nd.startswith(k+" "): meta_field=v;break
        if nd=="reference batch": meta_field="Reference batch"
        elif nd=="4m change": meta_field="4M change"
        elif nd=="process deviation": meta_field="Process deviation"
        elif nd in {"batch notes","notes"}: meta_field="Notes"

        if meta_field:
            for j,b in batch_names.items():
                batch_meta[b][meta_field]=s(raw.iat[ri,j])
            continue

        role=role_for(descriptor,sheet)
        if role=="unknown":
            # Engineering-assessment / commentary rows are preserved as unmapped,
            # rather than being forced into physical/dyno calculations.
            if any(s(raw.iat[ri,j]) for j in bidx):
                unm.append({
                    "Source sheet":sheet,"Source row":ri+1,"Descriptor":descriptor,
                    "Value":" | ".join(f"{batch_names[j]}={s(raw.iat[ri,j])}" for j in bidx if s(raw.iat[ri,j])),
                    "Reason":"Engineering/comment row not used in numeric analysis",
                })
            continue

        unit=s(raw.iat[ri,unit_col]) if unit_col is not None else ""
        lo=num(raw.iat[ri,lsl_col]) if lsl_col is not None else None
        hi=num(raw.iat[ri,usl_col]) if usl_col is not None else None
        tar=num(raw.iat[ri,target_col]) if target_col is not None else None

        for j,b in batch_names.items():
            v=raw.iat[ri,j]
            if role=="physical":
                phys.append({
                    "Batch / Lot":b,"Test":descriptor,"Sample ID":"","Result":v,
                    "Result numeric":num(v),"Unit":unit,"LSL":lo,"USL":hi,"Target":tar,
                    "Status":status(v,lo,hi),"Source sheet":sheet,"Source row":ri+1,
                })
            else:
                dyno.append({
                    "Batch / Lot":b,"Test":descriptor,"Metric":descriptor,"Run":"","Result":v,
                    "Result numeric":num(v),"Unit":unit,"Min":lo,"Max":hi,"Target":tar,
                    "Status":status(v,lo,hi),"Source sheet":sheet,"Source row":ri+1,
                })

    batches=[];seq=seq_start
    for b,rec in batch_meta.items():
        seq+=1;rec["Sequence"]=seq;batches.append(rec)
    return batches,phys,dyno,unm,seq

def _frame(rows,cols):
    if not rows:return pd.DataFrame(columns=cols)
    d=pd.DataFrame(rows)
    for c in cols:
        if c not in d:d[c]=None
    return d[cols]

def parse_lds_workbook(raw_bytes: bytes)->LDSData:
    xls=pd.ExcelFile(BytesIO(raw_bytes))
    all_b=[];all_p=[];all_d=[];all_u=[];diag=[];seq=0
    for sheet in xls.sheet_names:
        raw=pd.read_excel(BytesIO(raw_bytes),sheet_name=sheet,header=None,dtype=object)
        raw=_trim(raw)
        if raw.empty:
            diag.append({"Sheet":sheet,"Rows":0,"Columns":0,"Detected layout":"empty","Batches":0,"Physical records":0,"Dyno records":0,"Unmapped rows":0});continue
        h=header_row(raw);headers=_unique_headers(raw.iloc[h].tolist());table=raw.iloc[h+1:].copy();table.columns=headers
        candidates=[("batch-master",_batch_master_parse(table,sheet,seq)),("long",_long_parse(table,sheet,seq)),("row-table",_row_table_parse(raw,sheet,seq)),("wide-matrix",_wide_parse(raw,sheet,seq))]
        layout,res=max(candidates,key=lambda x: len(x[1][1])+len(x[1][2])+len(x[1][0])*0.2)
        b,p,d,u,newseq=res
        if not (b or p or d):
            u=[]
            for ri,row in raw.iterrows():
                vals=[s(v) for v in row.tolist() if s(v)]
                if vals:u.append({"Source sheet":sheet,"Source row":ri+1,"Descriptor":vals[0],"Value":" | ".join(vals[1:5]),"Reason":"Sheet not automatically mapped"})
        else: seq=newseq
        all_b+=b;all_p+=p;all_d+=d;all_u+=u
        diag.append({"Sheet":sheet,"Rows":len(raw),"Columns":raw.shape[1],"Detected layout":layout if (b or p or d) else "unmapped","Batches":len({x.get('Batch / Lot') for x in b}),"Physical records":len(p),"Dyno records":len(d),"Unmapped rows":len(u)})
    batches=_frame(all_b,BATCH_COLUMNS)
    if len(batches):
        batches=batches.sort_values("Sequence").drop_duplicates("Batch / Lot",keep="first").reset_index(drop=True)
    return LDSData(batches,_frame(all_p,PHYSICAL_COLUMNS),_frame(all_d,DYNO_COLUMNS),_frame(all_u,UNMAPPED_COLUMNS),pd.DataFrame(diag))
