"""Compute the 4-method ranking, architecture split, and total compute time
for the barley HEB-25 benchmark from the raw per-seed CSVs."""
import csv, statistics as st

targets = ["Dundee_2014_N0","Dundee_2014_N1","Dundee_2015_N0","Dundee_2015_N1",
           "Halle_2014_N0","Halle_2014_N1","Halle_2015_N0","Halle_2015_N1","FT"]

def col(path,c):
    out=[]
    for row in csv.DictReader(open(path)):
        try: out.append(float(row[c]))
        except: pass
    return out

summary={}
for t in targets:
    py=f"results/raw_results/py_{t}.csv"; r=f"results/raw_results/r_{t}.csv"
    summary[t]=dict(N=int(col(py,"n_train")[0]),
        gpfn=st.mean(col(py,"gpfn_pearson")), gblup=st.mean(col(py,"gblup_pearson")),
        pcr=st.mean(col(py,"pcr_pearson")), bayesb=st.mean(col(r,"bayesb_pearson")))

meth=['gpfn','gblup','pcr','bayesb']
overall={m:st.mean(summary[t][m] for t in summary) for m in meth}
print("RANKING:"," > ".join(f"{m}({overall[m]:.4f})" for m in sorted(overall,key=lambda x:-overall[x])))
print("BayesB > GPFN in", sum(1 for t in summary if summary[t]['bayesb']>summary[t]['gpfn']),"/9")
tgw=[t for t in summary if t!='FT']
print("Polygenic BayesB-GPFN gap:", round(st.mean(summary[t]['bayesb']-summary[t]['gpfn'] for t in tgw),4))
print("Oligogenic (FT) gap:", round(summary['FT']['bayesb']-summary['FT']['gpfn'],4))

# compute time
tot=0.0; bay=0.0
for t in targets:
    tot+=sum(col(f"results/raw_results/py_{t}.csv","wall_clock_seconds"))
    tot+=sum(col(f"results/raw_results/r_{t}.csv","wall_clock_seconds"))
    bay+=sum(col(f"results/raw_results/r_{t}.csv","bayesb_seconds"))
print(f"Total compute: {tot/3600:.1f} CPU-hours (BayesB {bay/3600:.1f}h)")
