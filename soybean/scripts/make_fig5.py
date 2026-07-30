import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':9})
fig, ax = plt.subplots(figsize=(9.6, 5.6))
ax.set_xlim(0,10); ax.set_ylim(0,7.6); ax.axis('off')
def box(x,y,w,h,label,sub,fc,ec,lw=1.1,fs=9):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.06,rounding_size=0.08",linewidth=lw,edgecolor=ec,facecolor=fc,alpha=.95))
    ax.text(x+w/2,y+h*0.66,label,ha='center',va='center',fontsize=fs,weight='bold',color='#1a1a1a')
    ax.text(x+w/2,y+h*0.27,sub,ha='center',va='center',fontsize=7.2,color='#444444')
def arrow(x1,y1,x2,y2,col='#777777'):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=11,lw=1.0,color=col,shrinkA=2,shrinkB=2))

ax.text(0.15,7.40,'What each method is allowed to see',fontsize=11.5,weight='bold',color='#1a1a1a',va='top')
ax.text(0.15,6.92,'GBLUP and BayesB read the full marker matrix. PCR and GPFN see only 100 principal components.',fontsize=8.0,color='#666666',va='top')
ax.text(0.15,6.58,'The input layer of the released GPFN is a fixed 2048 x 100 matrix, so the projection width cannot be changed without refitting the prior.',fontsize=8.0,color='#666666',va='top')

box(0.15,2.80,1.85,1.1,'Marker matrix','29,131 SNPs','#f2f2f2','#999999')
box(3.0,4.80,2.6,1.05,'Genomic relationship','all markers, VanRaden GRM','#e8eef4','#30638e')
box(6.7,4.80,3.1,1.05,'GBLUP','infinitesimal prior: every marker\ncontributes a small effect','#e8eef4','#30638e')
box(3.0,3.40,2.6,1.05,'Marker effects','all markers retained','#e6f4ef','#1b9e77')
box(6.7,3.40,3.1,1.05,'BayesB','sparse prior: a few markers\ncarry large effects','#e6f4ef','#1b9e77')

# PCA box: heavier red border marks it as the bottleneck. no background band.
box(3.0,1.45,2.6,1.15,'PCA projection','100 components\n(fixed by architecture)','#fdecef','#d1495b',lw=2.2)
ax.text(4.30,1.16,'information bottleneck',fontsize=8.0,color='#d1495b',ha='center',va='center',style='italic',weight='bold')

box(6.7,2.10,3.1,1.00,'PCR','linear regression on the\n100 components','#f0f0f2','#8d99ae')
box(6.7,0.30,3.1,1.20,'GPFN','transformer, amortized Bayesian\ninference on the 100 components','#fdecef','#d1495b')

arrow(2.0,3.35,3.0,5.30); arrow(2.0,3.35,3.0,3.90); arrow(2.0,3.35,3.0,2.10)
arrow(5.6,5.30,6.7,5.30); arrow(5.6,3.90,6.7,3.90)
arrow(5.6,2.30,6.7,2.60); arrow(5.6,1.90,6.7,1.15)
ax.text(6.20,2.00,'identical inputs',fontsize=7.2,color='#888888',ha='center',va='center',style='italic')

plt.tight_layout()
plt.savefig('fig5_information_flow.png', dpi=300, bbox_inches='tight')
plt.savefig('fig5_information_flow.pdf', bbox_inches='tight')
print("fig5 regenerated")
