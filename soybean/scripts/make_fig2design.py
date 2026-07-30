import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.lines import Line2D
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':9})

fig, ax = plt.subplots(figsize=(9.4, 4.8))
ax.set_xlim(0, 10); ax.set_ylim(0, 6.4); ax.axis('off')

TRAIN = '#30638e'; TEST = '#d1495b'; GREY = '#999999'

ax.text(0.1, 6.25, 'Between-families evaluation', fontsize=11.5, weight='bold', color='#1a1a1a', va='top')
ax.text(0.1, 5.80,
        'Whole families are held out, not random individuals. A test line has no full sibs in the training set and is\n'
        'connected to it only through the common recurrent parent. Roughly 20% of families are held out, and the\n'
        'partition is redrawn 100 times per location-trait combination.',
        fontsize=8.0, color='#666666', va='top')

# recurrent parent
ax.add_patch(Circle((5.0, 4.20), 0.24, facecolor='#f6d743', edgecolor='#8a7300', lw=1.2, zorder=5))
ax.text(5.0, 4.20, 'P', ha='center', va='center', fontsize=8.5, weight='bold', color='#5a4c00', zorder=6)
ax.text(5.34, 4.20, 'IA3023: recurrent parent, crossed into every family',
        ha='left', va='center', fontsize=7.4, color='#666666')

# families
n_fam = 13
test_idx = {2, 6, 10}      # 3 of 13 = 23%, matches the stated ~20%
x0, x1 = 0.55, 9.45
xs = np.linspace(x0, x1, n_fam)
top_y = 3.05
fam_h = 1.55
fam_w = (x1 - x0) / n_fam * 0.62

for i, x in enumerate(xs):
    is_test = i in test_idx
    col = TEST if is_test else TRAIN
    fc = '#fdecef' if is_test else '#e8eef4'
    ax.add_patch(FancyBboxPatch((x - fam_w/2, top_y - fam_h), fam_w, fam_h,
                 boxstyle="round,pad=0.02,rounding_size=0.05",
                 linewidth=1.4 if is_test else 0.9, edgecolor=col, facecolor=fc, alpha=.95, zorder=2))
    # line from parent
    ax.add_patch(FancyArrowPatch((5.0, 3.98), (x, top_y + 0.03), arrowstyle='-',
                 lw=0.6, color='#cccccc', shrinkA=1, shrinkB=1, zorder=1))
    # RILs inside the family
    rng = np.random.default_rng(i)
    for r in range(9):
        rx = x + (rng.random() - .5) * fam_w * 0.62
        ry = top_y - 0.22 - r * (fam_h - 0.42) / 8.5
        ax.plot(rx, ry, 'o', ms=2.4, color=col, alpha=.85, zorder=3)
    ax.text(x, top_y - fam_h - 0.20, f'{i+1}', ha='center', va='center',
            fontsize=6.6, color=col, weight='bold' if is_test else 'normal')

ax.text(x0 - 0.42, top_y - fam_h/2, 'families', ha='right', va='center',
        fontsize=8, color='#666666', rotation=90)
ax.text(5.0, top_y - fam_h - 0.55, 'about 39 families per location, each roughly 100 to 140 recombinant inbred lines',
        ha='center', va='center', fontsize=7.4, color='#888888')

handles = [
    Line2D([], [], marker='s', color=TRAIN, ls='', ms=8, label='training families (about 80%)'),
    Line2D([], [], marker='s', color=TEST,  ls='', ms=8, label='held-out families (about 20%)'),
]
ax.legend(handles=handles, frameon=False, fontsize=8, loc='lower center',
          bbox_to_anchor=(0.5, -0.02), ncol=2)

plt.tight_layout()
plt.savefig('fig2_design.png', dpi=300, bbox_inches='tight')
plt.savefig('fig2_design.pdf', bbox_inches='tight')
print("fig2_design written")
