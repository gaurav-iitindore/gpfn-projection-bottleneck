#!/usr/bin/env python3
import csv, glob, os, re
import numpy as np
from scipy import stats
METHODS=['gpfn','gblup','pcr','bayesb']
LABEL={'gpfn':'GPFN','gblup':'GBLUP','pcr':'PCR','bayesb':'BayesB'}
combos={}
for f in sorted(glob.glob("final_results/final_*_2012_*.csv")):
    m=re.match(r'final_([A-Za-z]+)_2012_(yield|protein|oil)\.csv$', os.path.basename(f))
    loc,tr=m.group(1),m.group(2); d={k:[] for k in METHODS}; nt=None
    for r in csv.DictReader(open(f)):
        nt=int(r['n_train'])
        for k in METHODS: d[k].append(float(r[f'{k}_pearson']))
    combos[(loc,tr)]={'n_train':nt, **{k:np.array(v) for k,v in d.items()}}
order=sorted(combos.items(), key=lambda x:x[1]['n_train']); n_comp=len(order)

print("="*116)
print("TABLE 1.  BayesB vs GPFN per combo (100 paired seeds). Holm-Bonferroni across 17 combos.")
print("="*116)
print(f"{'Location':10s}{'Trait':9s}{'N_train':>8s}  {'GPFN':>8s}{'BayesB':>8s}  {'Delta':>8s}{'95% CI':>20s}  {'d':>6s}{'p_raw':>10s}{'p_holm':>10s} sig")
raw_p=[]; rows=[]
for (loc,tr),c in order:
    a,b=c['gpfn'],c['bayesb']; d=b-a
    t,p=stats.ttest_rel(b,a); se=d.std(ddof=1)/np.sqrt(len(d))
    ci=(d.mean()-1.984*se, d.mean()+1.984*se); dz=d.mean()/d.std(ddof=1)
    raw_p.append(p); rows.append((loc,tr,c['n_train'],a.mean(),b.mean(),d.mean(),ci,dz,p))
idx=np.argsort(raw_p); holm=np.empty(n_comp); run=0.0
for rank,i in enumerate(idx):
    run=max(run,(n_comp-rank)*raw_p[i]); holm[i]=min(run,1.0)
for i,(loc,tr,nt,ga,bb,dm,ci,dz,p) in enumerate(rows):
    sig='***' if holm[i]<0.001 else '**' if holm[i]<0.01 else '*' if holm[i]<0.05 else 'ns'
    print(f"{loc:10s}{tr:9s}{nt:>8d}  {ga:8.4f}{bb:8.4f}  {dm:>+8.4f}  [{ci[0]:+.4f},{ci[1]:+.4f}]  {dz:>6.2f}{p:>10.1e}{holm[i]:>10.1e} {sig}")
print(f"\n  BayesB higher in {sum(1 for r in rows if r[5]>0)}/17 combos; significant after Holm in {int((holm<0.05).sum())}/17.")

print("\n"+"="*116)
print("TABLE 2.  All pairwise comparisons, pooled across 1700 paired evaluations")
print("="*116)
print(f"{'Comparison':22s}{'Mean diff':>11s}{'95% CI':>21s}{'Cohen d':>9s}{'p':>12s}{'Wins':>13s}")
for i,a in enumerate(METHODS):
    for b in METHODS[i+1:]:
        A=np.concatenate([c[a] for c in combos.values()]); B=np.concatenate([c[b] for c in combos.values()])
        d=A-B; t,p=stats.ttest_rel(A,B); se=d.std(ddof=1)/np.sqrt(len(d)); dz=d.mean()/d.std(ddof=1)
        print(f"{LABEL[a]+' - '+LABEL[b]:22s}{d.mean():>+11.4f}  [{d.mean()-1.96*se:+.4f},{d.mean()+1.96*se:+.4f}]{dz:>9.2f}{p:>12.1e}{int((A>B).sum()):>8d}/1700")

print("\n"+"="*116)
print("TABLE 3.  Method ranking")
print("="*116)
allm={k:np.concatenate([c[k] for c in combos.values()]) for k in METHODS}
rank=sorted(METHODS,key=lambda k:allm[k].mean(),reverse=True)
best_ct={k:0 for k in METHODS}; seedwin={k:0 for k in METHODS}
for c in combos.values():
    mm={k:c[k].mean() for k in METHODS}; best_ct[max(mm,key=mm.get)]+=1
    mat=np.vstack([c[k] for k in METHODS]); w=np.argmax(mat,axis=0)
    for i,k in enumerate(METHODS): seedwin[k]+=int((w==i).sum())
print(f"{'Rank':6s}{'Method':10s}{'Mean r':>9s}{'SD':>8s}{'Best in':>10s}{'Seed wins':>13s}")
for i,k in enumerate(rank,1):
    print(f"{i:<6d}{LABEL[k]:10s}{allm[k].mean():>9.4f}{allm[k].std():>8.4f}{best_ct[k]:>7d}/17{seedwin[k]:>9d}/1700")

print("\n"+"="*116)
print("TABLE 4.  Does the GPFN gap depend on training population size?")
print("="*116)
N=np.array([c['n_train'] for _,c in order],float)
for comp in ['gblup','bayesb','pcr']:
    gap=np.array([(c['gpfn']-c[comp]).mean() for _,c in order])
    r,p=stats.pearsonr(N,gap); rs,ps=stats.spearmanr(N,gap); sl,ic,rv,pv,se=stats.linregress(N,gap)
    print(f"  GPFN - {LABEL[comp]:7s}:  Pearson r={r:+.3f} (p={p:.4f})   Spearman={rs:+.3f} (p={ps:.4f})   slope={sl:+.2e}")
print("\n  (negative r = GPFN falls further behind as training N grows)")
yv=sorted([(c['n_train'],(c['gpfn']-c['gblup']).mean(),(c['gpfn']-c['bayesb']).mean()) for (l,t),c in combos.items() if t=='yield'])
yN=np.array([x[0] for x in yv],float)
print("\n  YIELD only (9 combos, the trait used in the original paper):")
for lab,i in [('vs GBLUP ',1),('vs BayesB',2)]:
    dd=np.array([x[i] for x in yv]); r,p=stats.pearsonr(yN,dd)
    print(f"    GPFN {lab}: Pearson r={r:+.3f} (p={p:.4f})")
print()
