"""Generate all figures for the barley HEB-25 GPFN benchmark."""
import csv, statistics as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family']='DejaVu Sans'
mpl.rcParams['axes.spines.top']=False
mpl.rcParams['axes.spines.right']=False

# palette matching Paper 1
C = dict(bayesb="#2a9d8f", gblup="#264f8e", pcr="#9aa7c7", gpfn="#e63946")

def col(path,c):
    out=[]
    for row in csv.DictReader(open(path)):
        try: out.append(float(row[c]))
        except: pass
    return out

targets = ["Dundee_2014_N0","Dundee_2014_N1","Dundee_2015_N0","Dundee_2015_N1",
           "Halle_2014_N0","Halle_2014_N1","Halle_2015_N0","Halle_2015_N1","FT"]
labels  = ["Dun\n14_N0","Dun\n14_N1","Dun\n15_N0","Dun\n15_N1","Hal\n14_N0","Hal\n14_N1","Hal\n15_N0","Hal\n15_N1","FT\n(flow)"]

# gather means + CIs
data={m:[] for m in ['bayesb','gblup','pcr','gpfn']}
err ={m:[] for m in ['bayesb','gblup','pcr','gpfn']}
for t in targets:
    py=f"results/raw_results/py_{t}.csv"; r=f"results/raw_results/r_{t}.csv"
    vals=dict(gpfn=col(py,"gpfn_pearson"),gblup=col(py,"gblup_pearson"),
              pcr=col(py,"pcr_pearson"),bayesb=col(r,"bayesb_pearson"))
    for m in data:
        v=vals[m]; data[m].append(st.mean(v)); err[m].append(1.96*st.pstdev(v)/np.sqrt(len(v)))

# ---------- FIG 1: 4-method accuracy bars ----------
fig,ax=plt.subplots(figsize=(11,5))
x=np.arange(len(targets)); w=0.2
for i,m in enumerate(['bayesb','gblup','pcr','gpfn']):
    ax.bar(x+(i-1.5)*w, data[m], w, yerr=err[m], capsize=2,
           label=m.upper() if m!='gpfn' else 'GPFN', color=C[m], error_kw=dict(lw=0.8))
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("Prediction accuracy (Pearson $r$)")
ax.set_title("Barley HEB-25: prediction accuracy across 9 targets (100 seeds each)", fontweight='bold', fontsize=12)
ax.legend(ncol=4, frameon=False, loc='upper left')
ax.axhline(0,color='k',lw=0.6)
plt.tight_layout(); plt.savefig("figures/Fig1_accuracy_by_target.png",dpi=200,bbox_inches='tight'); plt.close()
print("Fig1 done")

# ---------- FIG 2: architecture split (the headline) ----------
fig,ax=plt.subplots(figsize=(7,5))
tgw_gap=[]; 
for t in targets[:-1]:
    py=f"results/raw_results/py_{t}.csv"; r=f"results/raw_results/r_{t}.csv"
    tgw_gap.append(st.mean(col(r,"bayesb_pearson"))-st.mean(col(py,"gpfn_pearson")))
ft_py=f"results/raw_results/py_FT.csv"; ft_r=f"results/raw_results/r_FT.csv"
ft_gap=st.mean(col(ft_r,"bayesb_pearson"))-st.mean(col(ft_py,"gpfn_pearson"))

bp=ax.boxplot([tgw_gap],[0],positions=[0],widths=0.5,patch_artist=True)
for b in bp['boxes']: b.set_facecolor("#9aa7c7")
ax.scatter([0]*len(tgw_gap),tgw_gap,color="#264f8e",zorder=3,s=30,label="polygenic (grain weight, 8 targets)")
ax.scatter([1],[ft_gap],color="#e63946",zorder=3,s=90,marker='D',label="oligogenic (flowering, 1 target)")
ax.axhline(0,color='grey',ls='--',lw=1)
ax.set_xticks([0,1]); ax.set_xticklabels(["Polygenic\n(grain weight)","Oligogenic\n(flowering)"])
ax.set_ylabel("BayesB $-$ GPFN  (Pearson $r$)")
ax.set_title("The projection bottleneck's cost scales with trait architecture", fontweight='bold', fontsize=11)
ax.legend(frameon=False, fontsize=8, loc='upper left')
ax.annotate(f"+{ft_gap:.3f}",(1,ft_gap),xytext=(1.1,ft_gap),fontsize=10,fontweight='bold',va='center')
ax.annotate(f"mean +{st.mean(tgw_gap):.3f}",(0,st.mean(tgw_gap)),xytext=(0.15,0.02),fontsize=9)
plt.tight_layout(); plt.savefig("figures/Fig2_architecture_split.png",dpi=200,bbox_inches='tight'); plt.close()
print("Fig2 done")

# ---------- FIG 3: GPFN tracks its projection (GPFN-PCR vs GPFN-BayesB) ----------
fig,ax=plt.subplots(figsize=(7,5))
gp_pcr=[]; gp_bay=[]
for t in targets:
    py=f"results/raw_results/py_{t}.csv"; r=f"results/raw_results/r_{t}.csv"
    g=st.mean(col(py,"gpfn_pearson")); p=st.mean(col(py,"pcr_pearson")); b=st.mean(col(r,"bayesb_pearson"))
    gp_pcr.append(abs(g-p)); gp_bay.append(abs(g-b))
yy=np.arange(len(targets))
ax.scatter(gp_pcr,yy,color="#9aa7c7",s=50,label="|GPFN $-$ PCR|",zorder=3)
ax.scatter(gp_bay,yy,color="#2a9d8f",s=50,label="|GPFN $-$ BayesB|",zorder=3)
for i in yy: ax.plot([gp_pcr[i],gp_bay[i]],[i,i],color='lightgrey',zorder=1)
ax.set_yticks(yy); ax.set_yticklabels([t.replace("_","\\_") for t in targets],fontsize=7)
ax.set_xlabel("Absolute difference in mean accuracy from GPFN")
ax.set_title("GPFN behaves like its projection (PCR), not like BayesB", fontweight='bold', fontsize=11)
ax.legend(frameon=False, fontsize=9)
plt.tight_layout(); plt.savefig("figures/Fig3_behavioural_tracking.png",dpi=200,bbox_inches='tight'); plt.close()
print("Fig3 done")

print("ALL FIGURES DONE")


# ---------- FIG 4: forest plot, BayesB - GPFN per target (Holm-corrected) ----------
def make_forest():
    import csv as _csv
    rows=list(_csv.DictReader(open("results/statistical_tests.csv")))
    # order by n_train, FT last-ish by architecture
    rows=sorted(rows, key=lambda r:(r['architecture']=='oligogenic', int(r['n_train'])))
    fig,ax=plt.subplots(figsize=(8,5.5))
    yy=np.arange(len(rows))
    for i,r in enumerate(rows):
        md=float(r['bayesb_minus_gpfn']); lo=float(r['ci_low']); hi=float(r['ci_high'])
        sig=r['significant_holm']=='yes'
        color="#e63946" if sig else "#9aa7c7"
        ax.plot([lo,hi],[i,i],color=color,lw=2,zorder=2)
        ax.scatter([md],[i],color=color,s=55,zorder=3)
        tag="***" if sig else "n.s."
        ax.annotate(tag,(hi,i),xytext=(6,0),textcoords='offset points',va='center',fontsize=8,color='#555')
    ax.axvline(0,color='grey',ls='--',lw=1)
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{r['target'].replace('_','\\_')}  (N={r['n_train']})" for r in rows], fontsize=8)
    ax.set_xlabel("Accuracy difference:  BayesB $-$ GPFN  (Pearson $r$)")
    ax.set_title("BayesB beats GPFN only on the oligogenic trait (flowering)", fontweight='bold', fontsize=11)
    ax.text(0.98,0.02,"red = significant after Holm-Bonferroni",transform=ax.transAxes,
            ha='right',fontsize=8,color='#888')
    plt.tight_layout(); plt.savefig("figures/Fig4_forest_bayesb_vs_gpfn.png",dpi=200,bbox_inches='tight'); plt.close()
    print("Fig4 (forest) done")

make_forest()
