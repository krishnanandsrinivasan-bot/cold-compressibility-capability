from __future__ import annotations
from io import BytesIO
from zipfile import ZipFile, ZIP_DEFLATED
import re
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from ui import hero
from lds_engine import parse_lds_workbook
from lds_report_pdf import build_lds_pdf, comparison_figure, completeness_figure, chart_png_bytes, status_heatmap

hero("LDS Friction Lab", "Upload one LDS workbook and automatically analyze batches, physicals, dyno results, trends, gaps and comparisons.")
st.info("**One LDS file = one engineering workspace.** Data is parsed in-session only and is not written to the public GitHub repository.")

@st.cache_data(show_spinner=False)
def parse_lds(raw: bytes):
    return parse_lds_workbook(raw)

def demo_bytes():
    rows=[
      ["Parameter","Unit","LSL","USL","440122","453321","453901","454204"],
      ["Supplier","","","","ITT","ITT","ITT","ITT"],
      ["Material","","","","GA7900-D04","GA7900-D21","GA7900-D21","GA7900-D21"],
      ["Underlayer","","","","UL80","UL65","UL65","UL65"],
      ["Sample level","","","","B","C1","C1","C2"],
      ["Cold Compressibility (K3)","µm",75,125,100,103,108,105],
      ["Heat Transfer","°C","",200,190,198,205,196],
      ["Shear Strength","kN",2.5,"",3.1,3.0,2.8,3.2],
      ["AK-Master minimum mu","",0.33,"",0.35,0.34,0.32,0.35],
      ["AMS max temperature","°C","",650,610,620,645,635],
      ["Wear pad","mm","",1.5,1.1,1.2,1.4,1.3],
      ["EPB Holding Torque","Nm",1500,"",1600,1580,"",1620],
      ["WLTP Brake Emission","mg/km","",3.0,2.6,2.8,3.2,2.7],
    ]
    out=BytesIO()
    with pd.ExcelWriter(out,engine="xlsxwriter") as w:
        pd.DataFrame(rows).to_excel(w,index=False,header=False,sheet_name="LDS")
    return out.getvalue()

def numeric(df,test,batches):
    d=df[(df["Batch / Lot"].astype(str).isin(batches)) & (df["Test"].astype(str)==str(test))].copy()
    d["Result numeric"]=pd.to_numeric(d["Result numeric"],errors="coerce")
    return d.dropna(subset=["Result numeric"])

def add_limits(fig,d):
    lo="LSL" if "LSL" in d.columns else ("Min" if "Min" in d.columns else None)
    hi="USL" if "USL" in d.columns else ("Max" if "Max" in d.columns else None)
    for col,label in [(lo,"Lower"),(hi,"Upper")]:
        if col:
            vals=pd.to_numeric(d[col],errors="coerce").dropna().unique()
            if len(vals)==1: fig.add_hline(y=float(vals[0]),line_dash="dash",annotation_text=f"{label} {vals[0]:g}")
    return fig

def matrix(df,batches):
    if df.empty:return pd.DataFrame()
    d=df[df["Batch / Lot"].astype(str).isin(batches)].copy()
    d["Display"]=d.apply(lambda r: f"{'' if pd.isna(r['Result']) else r['Result']} {r.get('Unit') or ''} [{r.get('Status') or ''}]".strip(),axis=1)
    return d.pivot_table(index="Test",columns="Batch / Lot",values="Display",aggfunc=lambda x:" | ".join(dict.fromkeys(map(str,x)))).reindex(columns=[b for b in batches if b in set(d["Batch / Lot"].astype(str))])

COMPLETENESS_COLUMNS=["Batch / Lot","Records","Completed","Open","Fail","Completeness %"]

def completeness(p,d,batches):
    rows=[]
    p=p.copy() if isinstance(p,pd.DataFrame) else pd.DataFrame()
    d=d.copy() if isinstance(d,pd.DataFrame) else pd.DataFrame()
    for b in list(batches or []):
        pb=p[p["Batch / Lot"].astype(str)==str(b)] if "Batch / Lot" in p.columns else pd.DataFrame()
        db=d[d["Batch / Lot"].astype(str)==str(b)] if "Batch / Lot" in d.columns else pd.DataFrame()
        p_status=pb["Status"].astype(str).str.upper() if "Status" in pb.columns else pd.Series(dtype=str)
        d_status=db["Status"].astype(str).str.upper() if "Status" in db.columns else pd.Series(dtype=str)
        total=len(pb)+len(db)
        open_n=int((p_status=="OPEN").sum())+int((d_status=="OPEN").sum())
        fail=int((p_status=="FAIL").sum())+int((d_status=="FAIL").sum())
        rows.append({"Batch / Lot":str(b),"Records":total,"Completed":max(total-open_n,0),"Open":open_n,"Fail":fail,"Completeness %":round(100*(total-open_n)/total,1) if total else np.nan})
    return pd.DataFrame(rows,columns=COMPLETENESS_COLUMNS)

def excel_export(name,batches,p,d,unmapped,diag,selected):
    out=BytesIO(); gaps=pd.concat([p[p.Status.astype(str).str.upper().isin(["OPEN","FAIL"])].assign(Area="Physicals"),d[d.Status.astype(str).str.upper().isin(["OPEN","FAIL"])].assign(Area="Dyno")],ignore_index=True,sort=False)
    with pd.ExcelWriter(out,engine="xlsxwriter") as w:
        pd.DataFrame([["Source LDS",name],["Selected batches",len(selected)],["Physical records",len(p)],["Dyno records",len(d)],["Unmapped rows",len(unmapped)]],columns=["Metric","Value"]).to_excel(w,index=False,sheet_name="Summary")
        completeness(p,d,selected).to_excel(w,index=False,sheet_name="Completeness")
        batches[batches["Batch / Lot"].astype(str).isin(selected)].to_excel(w,index=False,sheet_name="Batches")
        p.to_excel(w,index=False,sheet_name="Physicals"); d.to_excel(w,index=False,sheet_name="Dyno"); gaps.to_excel(w,index=False,sheet_name="Open Gaps")
        matrix(p,selected).to_excel(w,sheet_name="Physical Matrix"); matrix(d,selected).to_excel(w,sheet_name="Dyno Matrix")
        diag.to_excel(w,index=False,sheet_name="Import Diagnostics"); unmapped.to_excel(w,index=False,sheet_name="Unmapped")
        wb=w.book; hdr=wb.add_format({"bold":True,"bg_color":"#1F4E78","font_color":"white"})
        for ws in w.sheets.values(): ws.set_row(0,None,hdr); ws.freeze_panes(1,0); ws.set_column(0,20,20)
    return out.getvalue()

def graphic_zip(p,d,batches,pt,dt):
    out=BytesIO()
    with ZipFile(out,"w",ZIP_DEFLATED) as z:
        z.writestr("01_Completeness.png",chart_png_bytes(completeness_figure(p,d,batches)))
        z.writestr("02_Physicals_Status.png",chart_png_bytes(status_heatmap(p,batches,"Physical Status Matrix")))
        z.writestr("03_Dyno_Status.png",chart_png_bytes(status_heatmap(d,batches,"Dyno Status Matrix")))
        if pt:z.writestr("04_Physical_Comparison.png",chart_png_bytes(comparison_figure(p,batches,pt,f"Physical Comparison - {pt}")))
        if dt:z.writestr("05_Dyno_Comparison.png",chart_png_bytes(comparison_figure(d,batches,dt,f"Dyno Comparison - {dt}")))
    return out.getvalue()

u1,u2=st.columns([5,1])
uploaded=u1.file_uploader("Upload LDS workbook",type=["xlsx","xls"],help="The importer scans all sheets and detects batch IDs, physical characteristics and dyno tests.")
if u2.button("Use demo LDS",use_container_width=True): st.session_state["lds_demo_v4"]=demo_bytes()
raw=uploaded.getvalue() if uploaded else st.session_state.get("lds_demo_v4")
name=uploaded.name if uploaded else ("Demo_LDS.xlsx" if raw else "")
if raw is None:
    st.markdown("### What happens after upload")
    st.write("The app creates a normalized batch database from the LDS and unlocks batch profiles, physical/dyno comparisons, trend charts, histograms, gap analysis and exports.")
    st.stop()

with st.spinner("Parsing LDS workbook..."): data=parse_lds(raw)
batches,phys,dyno,unmapped,diag=data.batches.copy(),data.physicals.copy(),data.dyno.copy(),data.unmapped.copy(),data.diagnostics.copy()
all_batches=list(dict.fromkeys([str(x) for x in pd.concat([batches.get("Batch / Lot",pd.Series(dtype=object)),phys.get("Batch / Lot",pd.Series(dtype=object)),dyno.get("Batch / Lot",pd.Series(dtype=object))],ignore_index=True).dropna() if str(x).strip()]))
if not all_batches:
    st.error("The workbook opened, but no batch IDs were detected. Check Import Diagnostics below or provide one representative LDS file so the parser can be adapted exactly.")
    st.dataframe(diag,use_container_width=True,hide_index=True); st.stop()

st.markdown("### Select batches")
selected=st.multiselect("Batches / lots included in analysis",all_batches,default=all_batches if len(all_batches)<=6 else all_batches[-5:])
if not selected:st.stop()
p=phys[phys["Batch / Lot"].astype(str).isin(selected)].copy(); d=dyno[dyno["Batch / Lot"].astype(str).isin(selected)].copy(); b=batches[batches["Batch / Lot"].astype(str).isin(selected)].copy()
status=pd.concat([p.get("Status",pd.Series(dtype=object)),d.get("Status",pd.Series(dtype=object))],ignore_index=True).astype(str).str.upper()
open_n=int((status=="OPEN").sum()); fail_n=int((status=="FAIL").sum())
metrics=st.columns(6); metrics[0].metric("Batches",len(selected)); metrics[1].metric("Physical records",len(p)); metrics[2].metric("Dyno records",len(d)); metrics[3].metric("Open gaps",open_n); metrics[4].metric("Failed criteria",fail_n); metrics[5].metric("Unmapped rows",len(unmapped))
ptests=sorted({str(x) for x in p.get("Test",pd.Series(dtype=object)).dropna() if str(x).strip()}); dtests=sorted({str(x) for x in d.get("Test",pd.Series(dtype=object)).dropna() if str(x).strip()})

tabs=st.tabs(["Dashboard","Batch Profile","Physicals","Dyno","Batch Comparison","Charts & Trends","Gaps / Quality","Export","Import Diagnostics"])
with tabs[0]:
    comp=completeness(p,d,selected); c1,c2=st.columns([1.4,1])
    with c1:
        required={"Batch / Lot","Completed","Open"}
        if comp.empty or not required.issubset(comp.columns):
            st.info("No completeness data could be calculated for the selected batches. Review Import Diagnostics to see how the LDS was interpreted.")
        else:
            m=comp[["Batch / Lot","Completed","Open"]].melt(id_vars=["Batch / Lot"],value_vars=["Completed","Open"],var_name="Evidence",value_name="Records")
            fig=px.bar(m,x="Batch / Lot",y="Records",color="Evidence",barmode="stack",title="LDS evidence completeness")
            fig.update_layout(template="plotly_white",legend_title_text="")
            st.plotly_chart(fig,use_container_width=True)
    with c2:
        st.markdown("#### Engineering attention")
        if fail_n:st.error(f"{fail_n} numeric result(s) fail imported criteria.")
        if open_n:st.warning(f"{open_n} LDS result(s) are open / blank.")
        if len(unmapped):st.info(f"{len(unmapped)} row(s) could not be classified automatically.")
        if not(fail_n or open_n or len(unmapped)):st.success("No automatic issues detected.")
    st.dataframe(comp,use_container_width=True,hide_index=True)
    if len(b):st.dataframe(b,use_container_width=True,hide_index=True)
with tabs[1]:
    batch=st.selectbox("Batch / Lot",selected,key="lds_profile")
    meta=b[b["Batch / Lot"].astype(str)==batch]
    if len(meta):
        row=meta.iloc[0]; cs=st.columns(4)
        for i,f in enumerate(["Supplier","Material / Grade","Underlayer","Sample level","Production date","Pad drawing","Backplate drawing","Process status"]):cs[i%4].metric(f,str(row.get(f) or "-"))
    c1,c2=st.columns(2); c1.dataframe(p[p["Batch / Lot"].astype(str)==batch],use_container_width=True,hide_index=True);c2.dataframe(d[d["Batch / Lot"].astype(str)==batch],use_container_width=True,hide_index=True)
with tabs[2]:
    if not ptests:st.info("No physical records detected.")
    else:
        test=st.selectbox("Physical characteristic",ptests,key="lds_phys_test"); x=numeric(p,test,selected); st.dataframe(p[p.Test.astype(str)==test],use_container_width=True,hide_index=True)
        if len(x):
            fig=px.box(x,x="Batch / Lot",y="Result numeric",color="Batch / Lot",points="all",title=f"{test} - batch distribution") if x.groupby("Batch / Lot").size().max()>1 else px.line(x,x="Batch / Lot",y="Result numeric",markers=True,title=f"{test} - batch comparison")
            st.plotly_chart(add_limits(fig,x),use_container_width=True)
with tabs[3]:
    if not dtests:st.info("No dyno records detected.")
    else:
        test=st.selectbox("Dyno test",dtests,key="lds_dyno_test"); x=numeric(d,test,selected); st.dataframe(d[d.Test.astype(str)==test],use_container_width=True,hide_index=True)
        if len(x):
            fig=px.box(x,x="Batch / Lot",y="Result numeric",color="Batch / Lot",points="all",title=f"{test} - repeats / batch distribution") if x.groupby("Batch / Lot").size().max()>1 else px.line(x,x="Batch / Lot",y="Result numeric",markers=True,title=f"{test} - batch comparison")
            st.plotly_chart(add_limits(fig,x),use_container_width=True)
with tabs[4]:
    c1,c2=st.columns(2); c1.markdown("#### Physical matrix");c1.dataframe(matrix(p,selected),use_container_width=True);c2.markdown("#### Dyno matrix");c2.dataframe(matrix(d,selected),use_container_width=True)
    for title,df in [("Physicals",p),("Dyno",d)]:
        if len(df):
            pv=df.pivot_table(index="Test",columns="Batch / Lot",values="Status",aggfunc=lambda z:list(z)[-1]).reindex(columns=[x for x in selected if x in set(df["Batch / Lot"].astype(str))]); code=pv.applymap(lambda x:{"FAIL":-1,"OPEN":0,"RESULT":1,"PASS":2,"CARRYOVER":2,"N.A.":1}.get(str(x).upper(),1)); fig=px.imshow(code,x=code.columns,y=code.index,aspect="auto",color_continuous_scale=[(0,"#f8c9c5"),(.33,"#fff1bf"),(.66,"#dbe9f6"),(1,"#cce8cf")],zmin=-1,zmax=2,title=f"{title} status matrix");fig.update_layout(coloraxis_showscale=False,height=max(320,30*len(code)+130));st.plotly_chart(fig,use_container_width=True)
with tabs[5]:
    a,bx,c=st.columns([1,2,1]); area=a.radio("Dataset",["Physicals","Dyno"]); source=p if area=="Physicals" else d; tests=ptests if area=="Physicals" else dtests
    if tests:
        test=bx.selectbox("Test / characteristic",tests,key="lds_studio"); typ=c.selectbox("Graphic",["Trend","Box plot","Histogram"]); x=numeric(source,test,selected)
        if len(x):
            if typ=="Histogram":fig=px.histogram(x,x="Result numeric",color="Batch / Lot",barmode="overlay",marginal="box",title=f"{test} - distribution")
            elif typ=="Box plot":fig=add_limits(px.box(x,x="Batch / Lot",y="Result numeric",color="Batch / Lot",points="all",title=f"{test} - batch distribution"),x)
            else:
                ag=x.groupby("Batch / Lot",as_index=False)["Result numeric"].mean(); fig=add_limits(px.line(ag,x="Batch / Lot",y="Result numeric",markers=True,title=f"{test} - trend"),x)
            fig.update_layout(template="plotly_white");st.plotly_chart(fig,use_container_width=True)
with tabs[6]:
    gaps=pd.concat([p[p.Status.astype(str).str.upper().isin(["OPEN","FAIL"])].assign(Area="Physicals"),d[d.Status.astype(str).str.upper().isin(["OPEN","FAIL"])].assign(Area="Dyno")],ignore_index=True,sort=False)
    if len(gaps):st.dataframe(gaps,use_container_width=True,hide_index=True)
    else:st.success("No OPEN or FAIL records for selected batches.")
    if len(unmapped):st.warning("Unmapped workbook rows are excluded from calculations until the parser is adapted.");st.dataframe(unmapped,use_container_width=True,hide_index=True)
with tabs[7]:
    e1,e2=st.columns(2); rpt_p=e1.selectbox("Physical chart in PDF",[None]+ptests,index=1 if ptests else 0); rpt_d=e2.selectbox("Dyno chart in PDF",[None]+dtests,index=1 if dtests else 0)
    pdf=build_lds_pdf(file_name=name,batches_df=batches,phys=p,dyno=d,unmapped=unmapped,selected_batches=selected,physical_test=rpt_p,dyno_test=rpt_d); xlsx=excel_export(name,batches,p,d,unmapped,diag,selected); z=graphic_zip(p,d,selected,rpt_p,rpt_d);safe=re.sub(r"[^A-Za-z0-9_-]+","_",name.rsplit(".",1)[0]) or "LDS"
    cols=st.columns(3);cols[0].download_button("Engineering report (.pdf)",pdf,f"{safe}_Engineering_Report.pdf","application/pdf",use_container_width=True,type="primary");cols[1].download_button("Analysis package (.xlsx)",xlsx,f"{safe}_Analysis.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",use_container_width=True);cols[2].download_button("Graphics pack (.zip)",z,f"{safe}_Graphics.zip","application/zip",use_container_width=True)
with tabs[8]:
    st.caption("Automatic import is auditable. Unknown rows are kept here instead of being forced into a wrong category.");st.dataframe(diag,use_container_width=True,hide_index=True)
    if len(unmapped):st.dataframe(unmapped,use_container_width=True,hide_index=True)

st.divider();st.caption("Automatic PASS/FAIL is applied only where numeric criteria exist in the LDS. Final acceptance remains an engineering decision under the applicable project/customer requirements.")
