import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import json
import os
import seaborn

success = {
    'left_turn':[0.88],
    'cross':[0.92],
    're':[0.84],
    'rm':[0.76],
    'r':[0.66],
}
f = '/home/haochen/TPDM_transformer/'
for name in success.keys():
    for t in [3, 5, 10]:
        sr = []
        for n in range(3):
            if not os.path.exists(f + f'ABLsac_map_aughier_{t}_{name}_{n}.json'):
                continue
            with open(f + f'ABLsac_map_aughier_{t}_{name}_{n}.json','r') as reader:
                sc = json.load(reader)[1]
            sr.append(np.mean(sc[-50:]))
        success[name].append(np.max(sr))

sr  = {'left_turn': [0.88, 0.94, 0.92, 0.88, 0.82], 'cross': [0.92, 0.96, 0.94, 0.94, 0.90], 're': [0.84, 0.88, 0.92, 0.86, 0.78], 'rm': [0.76, 0.82, 0.82, 0.76, 0.74], 'r': [0.66, 0.76, 0.82, 0.84, 0.80]}



plt.figure(figsize=(15,7))
seaborn.set(style="whitegrid", font_scale=2, rc={"lines.linewidth": 3.5})
plt.ylabel('Avg Success rate')
x = [0, 0.9, 1.5, 3, 6]
plt.xlim([0,6])
# plt.xticks(x)
plt.xlabel('Time Horizon (s)')
m = ['o', 'v', 's', '^', 'd']
c = ['crimson', 'tan', 'rosybrown', 'slategrey', (105/255, 111/255, 145/255)]
i = 0

for k,v in sr.items():
    plt.plot(x, v, marker=m[i], markersize=20,color=c[i],linewidth=7,markeredgewidth=3.5,
    markeredgecolor='k',clip_on=False, markevery=1,zorder=10)
    # for x_i,v_i in zip(x, v):
    #     plt.text(x_i*1.05, v_i+0.005, str(v_i), ha='center', va='bottom', fontsize=25)
    i += 1

plt.legend(['Left Turn','Merge','R-A','R-B','R-C'],loc='upper center', bbox_to_anchor=(0.5, 1.16),
          ncol=5, fancybox=True,labelspacing=0.1)
plt.ylim([0.65,1.00])
# plt.xlim([0.65,1.00])
plt.savefig('/home/haochen/TPDM_transformer/stat_pic/A.svg')