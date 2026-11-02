from __future__ import annotations
from io import BytesIO
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether

NAVY=colors.HexColor('#183A5A'); BLUE=colors.HexColor('#1F4E78'); LIGHT=colors.HexColor('#EAF2F8'); GREY=colors.HexColor('#F3F5F7'); MID=colors.HexColor('#D7DCE1'); GREEN=colors.HexColor('#2E7D32'); RED=colors.HexColor('#B3261E'); AMBER=colors.HexColor('#9A6700')

def _t(v):
    if v is None: return '-'
    try:
        if pd.isna(v): return '-'
    except: pass
    z=str(v).strip(); return z if z else '-'

def _img(fig,width=174):
    b=BytesIO(); fig.savefig(b,format='png',dpi=170,bbox_inches='tight',facecolor='white'); plt.close(fig); b.seek(0)
    im=Image(b); ratio=im.imageHeight/im.imageWidth; im.drawWidth=width*mm; im.drawHeight=width*mm*ratio; return im

def _status_code(x):
    z=str(x).upper(); return {'FAIL':-1,'OPEN':0,'RESULT':1,'PASS':2,'CARRYOVER':2,'N.A.':1,'NA':1}.get(z,1)

def status_heatmap(df:pd.DataFrame,batches:list[str],title:str,label_col='Test'):
    if df.empty:
        fig,ax=plt.subplots(figsize=(10,2.5)); ax.axis('off'); ax.text(.5,.5,'No data available',ha='center',va='center'); ax.set_title(title,loc='left',fontweight='bold'); return fig
    d=df[df['Batch / Lot'].astype(str).isin(batches)].copy()
    if d.empty:
        fig,ax=plt.subplots(figsize=(10,2.5)); ax.axis('off'); ax.text(.5,.5,'No data for selected batches',ha='center',va='center'); ax.set_title(title,loc='left',fontweight='bold'); return fig
    pv=d.pivot_table(index=label_col,columns='Batch / Lot',values='Status',aggfunc=lambda x: list(x)[-1])
    pv=pv.reindex(columns=[b for b in batches if b in pv.columns])
    score=pv.applymap(_status_code)
    priority=score.min(axis=1).sort_values().index[:18]
    score=score.loc[priority]
    fig,ax=plt.subplots(figsize=(10.3,max(3.2,0.30*len(score)+1.8)))
    cmap=plt.matplotlib.colors.ListedColormap(['#f8c9c5','#fff1bf','#dbe9f6','#cce8cf'])
    ax.imshow(score.values,aspect='auto',vmin=-1,vmax=2,cmap=cmap)
    ax.set_xticks(range(len(score.columns))); ax.set_xticklabels(score.columns,rotation=35,ha='right',fontsize=8)
    ax.set_yticks(range(len(score.index))); ax.set_yticklabels(score.index,fontsize=8)
    ax.set_title(title,loc='left',fontweight='bold')
    for i,row in enumerate(score.index):
        for j,col in enumerate(score.columns):
            val=pv.loc[row,col] if row in pv.index and col in pv.columns else ''
            ax.text(j,i,_t(val),ha='center',va='center',fontsize=6.5)
    ax.tick_params(length=0); fig.tight_layout(); return fig

def comparison_figure(df:pd.DataFrame,batches:list[str],test:str,title:str):
    d=df[(df['Batch / Lot'].astype(str).isin(batches)) & (df['Test'].astype(str)==str(test))].copy()
    d=d[pd.to_numeric(d['Result numeric'],errors='coerce').notna()]
    fig,ax=plt.subplots(figsize=(10.3,3.7))
    if d.empty:
        ax.axis('off'); ax.text(.5,.5,'No numeric data available',ha='center',va='center'); ax.set_title(title,loc='left',fontweight='bold'); return fig
    d['Result numeric']=pd.to_numeric(d['Result numeric'],errors='coerce')
    order=[b for b in batches if b in set(d['Batch / Lot'].astype(str))]
    groups=[d.loc[d['Batch / Lot'].astype(str)==b,'Result numeric'].to_numpy() for b in order]
    if any(len(g)>1 for g in groups):
        ax.boxplot(groups,labels=order,showmeans=True)
        for i,g in enumerate(groups,1): ax.scatter(np.full(len(g),i),g,s=18,alpha=.55)
    else:
        vals=[g[0] if len(g) else np.nan for g in groups]
        ax.plot(order,vals,marker='o',linewidth=2)
    lo_col='LSL' if 'LSL' in d else ('Min' if 'Min' in d else None); hi_col='USL' if 'USL' in d else ('Max' if 'Max' in d else None)
    if lo_col:
        lows=pd.to_numeric(d[lo_col],errors='coerce').dropna().unique()
        if len(lows)==1: ax.axhline(lows[0],linestyle='--',linewidth=1,label=f'Lower {lows[0]:g}')
    if hi_col:
        highs=pd.to_numeric(d[hi_col],errors='coerce').dropna().unique()
        if len(highs)==1: ax.axhline(highs[0],linestyle='--',linewidth=1,label=f'Upper {highs[0]:g}')
    unit=next((str(x) for x in d['Unit'] if str(x).strip() and str(x).lower()!='nan'),'') if 'Unit' in d else ''
    ax.set_ylabel(unit); ax.set_title(title,loc='left',fontweight='bold'); ax.grid(True,axis='y',alpha=.2); ax.tick_params(axis='x',rotation=25)
    if ax.get_legend_handles_labels()[0]: ax.legend(frameon=False)
    fig.tight_layout(); return fig

def completeness_figure(phys:pd.DataFrame,dyno:pd.DataFrame,batches:list[str]):
    rows=[]
    for b in batches:
        for area,df in [('Physicals',phys),('Dyno',dyno)]:
            x=df[df['Batch / Lot'].astype(str)==b]
            rows.append((b,area,int((x['Status'].astype(str).str.upper()!='OPEN').sum()),int((x['Status'].astype(str).str.upper()=='OPEN').sum()),int((x['Status'].astype(str).str.upper()=='FAIL').sum())))
    fig,ax=plt.subplots(figsize=(10.3,3.6))
    if not rows:
        ax.axis('off'); ax.text(.5,.5,'No data available',ha='center',va='center'); return fig
    lab=[];done=[];openv=[]
    for b,a,d,o,f in rows: lab.append(f'{b}\n{a}');done.append(d);openv.append(o)
    x=np.arange(len(lab)); ax.bar(x,done,label='Completed');ax.bar(x,openv,bottom=done,label='Open')
    ax.set_xticks(x);ax.set_xticklabels(lab,rotation=35,ha='right',fontsize=8);ax.set_ylabel('Records');ax.set_title('LDS Completeness by Batch',loc='left',fontweight='bold');ax.legend(frameon=False);ax.grid(True,axis='y',alpha=.15);fig.tight_layout();return fig

def chart_png_bytes(fig)->bytes:
    b=BytesIO(); fig.savefig(b,format='png',dpi=190,bbox_inches='tight',facecolor='white'); plt.close(fig); return b.getvalue()

def build_lds_pdf(*,file_name:str,batches_df:pd.DataFrame,phys:pd.DataFrame,dyno:pd.DataFrame,unmapped:pd.DataFrame,selected_batches:list[str],physical_test:str|None=None,dyno_test:str|None=None)->bytes:
    out=BytesIO();doc=SimpleDocTemplate(out,pagesize=A4,leftMargin=15*mm,rightMargin=15*mm,topMargin=16*mm,bottomMargin=16*mm,title='LDS Friction Lab Report',author='Pad Development Tools')
    styles=getSampleStyleSheet();styles.add(ParagraphStyle(name='T',parent=styles['Title'],fontName='Helvetica-Bold',fontSize=18,textColor=NAVY,leading=22,spaceAfter=4));styles.add(ParagraphStyle(name='S',parent=styles['Normal'],fontSize=9.2,textColor=colors.HexColor('#5B6570'),leading=12,spaceAfter=8));styles.add(ParagraphStyle(name='H',parent=styles['Heading2'],fontName='Helvetica-Bold',fontSize=11.5,textColor=NAVY,spaceBefore=5,spaceAfter=5));styles.add(ParagraphStyle(name='B',parent=styles['BodyText'],fontSize=8.6,leading=12));styles.add(ParagraphStyle(name='Sm',parent=styles['BodyText'],fontSize=7.5,leading=9,textColor=colors.HexColor('#5B6570')))
    def footer(canvas,document):
        canvas.saveState();canvas.setStrokeColor(MID);canvas.line(15*mm,12*mm,A4[0]-15*mm,12*mm);canvas.setFont('Helvetica',7.2);canvas.setFillColor(colors.HexColor('#6B7280'));canvas.drawString(15*mm,7.5*mm,'Pad Development Tools | LDS Friction Lab');canvas.drawRightString(A4[0]-15*mm,7.5*mm,f'Page {document.page}');canvas.restoreState()
    p=phys[phys['Batch / Lot'].astype(str).isin(selected_batches)].copy();d=dyno[dyno['Batch / Lot'].astype(str).isin(selected_batches)].copy()
    open_n=int((pd.concat([p['Status'],d['Status']],ignore_index=True).astype(str).str.upper()=='OPEN').sum()) if len(p)+len(d) else 0;fail_n=int((pd.concat([p['Status'],d['Status']],ignore_index=True).astype(str).str.upper()=='FAIL').sum()) if len(p)+len(d) else 0
    story=[Paragraph('PAD DEVELOPMENT TOOLS',styles['Sm']),Paragraph('LDS Friction Lab - Engineering Report',styles['T']),Paragraph(f'Source: {_t(file_name)} | Selected batches: {len(selected_batches)}',styles['S'])]
    kpi=[[Paragraph(f'<b>{len(selected_batches)}</b><br/>Batches',styles['B']),Paragraph(f'<b>{len(p)}</b><br/>Physical records',styles['B']),Paragraph(f'<b>{len(d)}</b><br/>Dyno records',styles['B']),Paragraph(f'<b>{open_n}</b><br/>Open gaps',styles['B']),Paragraph(f'<b>{fail_n}</b><br/>Failed criteria',styles['B'])]]
    kt=Table(kpi,colWidths=[34*mm]*5,rowHeights=[17*mm]);kt.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),LIGHT),('BOX',(0,0),(-1,-1),.5,BLUE),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('ALIGN',(0,0),(-1,-1),'CENTER')])) ;story+=[kt,Spacer(1,5*mm),Paragraph('Batch Overview',styles['H'])]
    bsel=batches_df[batches_df['Batch / Lot'].astype(str).isin(selected_batches)].copy(); cols=[c for c in ['Batch / Lot','Supplier','Material / Grade','Underlayer','Sample level','Production date'] if c in bsel.columns]
    data=[cols]+[[_t(row[c]) for c in cols] for _,row in bsel.head(15).iterrows()]
    if len(data)==1:data.append(['-' for _ in cols])
    tb=Table(data,colWidths=[170*mm/max(1,len(cols))]*len(cols),repeatRows=1);tb.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NAVY),('TEXTCOLOR',(0,0),(-1,0),colors.white),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('GRID',(0,0),(-1,-1),.3,MID),('FONTSIZE',(0,0),(-1,-1),7.2),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]));story+=[tb,Spacer(1,4*mm),_img(completeness_figure(p,d,selected_batches)),PageBreak(),Paragraph('Physical Properties',styles['T']),Paragraph('Status matrix and selected numeric comparison. FAIL/OPEN rows are prioritized in the matrix.',styles['S']),_img(status_heatmap(p,selected_batches,'Physicals - Status Matrix'))]
    if physical_test:story += [Spacer(1,4*mm),_img(comparison_figure(p,selected_batches,physical_test,f'Physical Comparison - {physical_test}'))]
    story += [PageBreak(),Paragraph('Dyno Results',styles['T']),Paragraph('Status matrix and selected dyno comparison across the chosen batches.',styles['S']),_img(status_heatmap(d,selected_batches,'Dyno - Status Matrix'))]
    if dyno_test:story += [Spacer(1,4*mm),_img(comparison_figure(d,selected_batches,dyno_test,f'Dyno Comparison - {dyno_test}'))]
    story += [PageBreak(),Paragraph('Open Gaps & Data Quality',styles['T'])]
    gaps=pd.concat([p[p['Status'].astype(str).str.upper().isin(['OPEN','FAIL'])].assign(Area='Physicals'),d[d['Status'].astype(str).str.upper().isin(['OPEN','FAIL'])].assign(Area='Dyno')],ignore_index=True,sort=False)
    rows=[['Area','Batch / Lot','Test','Status','Result','Source']]
    for _,r in gaps.head(40).iterrows():rows.append([_t(r.get('Area')),_t(r.get('Batch / Lot')),_t(r.get('Test')),_t(r.get('Status')),_t(r.get('Result')),_t(r.get('Source sheet'))])
    if len(rows)==1:rows.append(['-','-','No open/failed records detected','-','-','-'])
    gt=Table(rows,colWidths=[20*mm,25*mm,58*mm,20*mm,23*mm,25*mm],repeatRows=1);gt.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NAVY),('TEXTCOLOR',(0,0),(-1,0),colors.white),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('GRID',(0,0),(-1,-1),.3,MID),('FONTSIZE',(0,0),(-1,-1),7),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]));story.append(gt)
    if len(unmapped):story += [Spacer(1,5*mm),Paragraph('Import diagnostics',styles['H']),Paragraph(f'{len(unmapped)} row(s) were not automatically classified. Review the Import Diagnostics tab before using the report for a formal decision.',styles['B'])]
    story += [Spacer(1,5*mm),Paragraph('Usage note',styles['H']),Paragraph('This report is generated from the uploaded LDS workbook. Automatic PASS/FAIL is applied only where numeric limits are available. Qualitative results, carry-over decisions and engineering acceptance remain subject to the applicable project/customer requirements and engineering judgement.',styles['Sm'])]
    doc.build(story,onFirstPage=footer,onLaterPages=footer);return out.getvalue()
